# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
import logging
import os
import shutil
import time
from typing import Dict, List, Set

from joblib import Parallel, delayed
from omegaconf import OmegaConf

import stopes.pipelines.prepare_data.data_types as data_types
from stopes.pipelines.prepare_data.cache import cache_step, cache_step_sync
from stopes.pipelines.prepare_data.encode_and_binarize import encode_and_binarize
from stopes.pipelines.prepare_data.prepare_vocab import get_vocab
from stopes.pipelines.prepare_data.retrieve_data import retrieve_data
from stopes.pipelines.prepare_data.sharding import (
    get_all_num_shards,
    write_to_all_shards,
)
from stopes.pipelines.prepare_data.utils import (
    async_noop,
    dedup_sharding,
    execute_in_shell,
    hash_parallel_data,
    setup_config,
)
from stopes.pipelines.prepare_data.validation import validate_data_config

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
    ],
    force=True,
)
logger = logging.getLogger(__name__)


@cache_step_sync("prepare_valid_test_direction")
def prepare_valid_test_direction(
    direction: str,
    valid_data: Dict[str, data_types.CorporaMap],
    test_data: Dict[str, data_types.CorporaMap],
    sampled_train_dict: Dict[str, Dict[str, data_types.CorporaMap]],
    src_vocab: data_types.BuiltVocab,
    tgt_vocab: data_types.BuiltVocab,
    all_num_shards: Dict[str, int],
    data_config: data_types.DataConfig,
    temp_dir: str,
    output_dir: str,
    custom_step_name: str,
) -> None:

    parent_dir = os.path.join(temp_dir, "temp_binarized")
    valid_test_dir = os.path.join(temp_dir, "encoded_valid_test")

    binarized_valid = None
    if valid_data is not None and direction in valid_data:
        binarized_valid = encode_and_binarize(
            direction=direction,
            parallel_data=valid_data[direction],
            tag="valid",
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            binarize_workers=data_config.binarization_config.binarize_workers,
            output_dir=output_dir,
            encoded_outdir=valid_test_dir,
            binarized_outdir=parent_dir,
            shard_id=0,
            custom_step_name=f"encode_and_binarize.valid.{direction}",
        )

    binarized_sampled_train_dict = {}
    if sampled_train_dict:
        for sampled_train_fold, sampled_train in sampled_train_dict.items():
            if direction in sampled_train:
                binarized_sampled_train_dict[sampled_train_fold] = encode_and_binarize(
                    direction=direction,
                    parallel_data=sampled_train[direction],
                    tag=sampled_train_fold,
                    src_vocab=src_vocab,
                    tgt_vocab=tgt_vocab,
                    binarize_workers=data_config.binarization_config.binarize_workers,
                    output_dir=output_dir,
                    encoded_outdir=valid_test_dir,
                    binarized_outdir=f"{temp_dir}/temp_binarized/{sampled_train_fold}",
                    shard_id=0,
                    custom_step_name=f"encode_and_binarize_{sampled_train_fold}.{direction}",
                )

    binarized_test = None
    if test_data is not None and direction in test_data:
        binarized_test = encode_and_binarize(
            direction=direction,
            parallel_data=test_data[direction],
            tag="test",
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            binarize_workers=data_config.binarization_config.binarize_workers,
            output_dir=output_dir,
            encoded_outdir=valid_test_dir,
            binarized_outdir=parent_dir,
            shard_id=0,
            custom_step_name=f"encode_and_binarize.test.{direction}",
        )

    source, target = direction.split("-")
    if binarized_sampled_train_dict:
        for (
            sampled_train_fold,
            binarized_sampled_train,
        ) in binarized_sampled_train_dict.items():
            # Move sampled train data into the main folder
            sampled_train_dir = os.path.join(parent_dir, sampled_train_fold)
            for ext in ["bin", "idx"]:
                for lang in [source, target]:
                    execute_in_shell(
                        f"mv {sampled_train_dir}/valid.{source}-{target}.{lang}.{ext} {sampled_train_dir}/{sampled_train_fold}.{source}-{target}.{lang}.{ext}"
                    )

    if binarized_valid is not None:
        write_to_all_shards(
            binarized_valid,
            all_num_shards[direction],
            f"{output_dir}/data_bin",
        )
    if binarized_sampled_train_dict:
        for _, binarized_sampled_train in binarized_sampled_train_dict.items():
            write_to_all_shards(
                binarized_sampled_train,
                all_num_shards[direction],
                f"{output_dir}/data_bin",
            )

    if binarized_test is not None:
        write_to_all_shards(
            binarized_test,
            all_num_shards[direction],
            f"{output_dir}/data_bin",
        )


@cache_step_sync("prepare_train_directory")
def prepare_train_direction(
    direction: str,
    train_data_dict: Dict[str, Dict[str, data_types.CorporaMap]],
    src_vocab: data_types.BuiltVocab,
    tgt_vocab: data_types.BuiltVocab,
    all_num_shards: Dict[str, int],
    data_config: data_types.DataConfig,
    temp_dir: str,
    output_dir: str,
    seen: Set[str],
    custom_step_name: str,
) -> None:
    train_shards_dict = {}
    if train_data_dict:
        for train_fold, train_data in train_data_dict.items():
            if direction in train_data:
                train_shards_dict[train_fold] = dedup_sharding(
                    output_dir=output_dir,
                    custom_step_name=f"dedup_sharding_{train_fold}.{direction}",
                    direction=direction,
                    train_parallel=train_data[direction],
                    seen=seen,
                    num_shards=all_num_shards[direction],
                    binarization_config=data_config.binarization_config,
                    sharding_output_dir=f"{temp_dir}/deduped_train",
                    train_fold=train_fold,
                )

    parent_dir = os.path.join(temp_dir, "temp_binarized")
    binarized_train_data_shards = {}
    for train_fold, train_shards in train_shards_dict.items():
        binarized_train_data_shards[train_fold] = Parallel(n_jobs=8, verbose=100)(
            delayed(encode_and_binarize)(
                direction=direction,
                parallel_data=shard,
                tag=train_fold,
                src_vocab=src_vocab,
                tgt_vocab=tgt_vocab,
                binarize_workers=data_config.binarization_config.binarize_workers,
                output_dir=output_dir,
                encoded_outdir=f"{temp_dir}/encoded_train/shard{i:03d}",
                binarized_outdir=f"{output_dir}/data_bin/shard{i:03d}"
                if train_fold == "train"
                else f"{parent_dir}/shard{i:03d}/{train_fold}",
                shard_id=i,
                custom_step_name=f"encode_and_binarize_{train_fold}.{direction}.{i}",
                encoded_filtered_outdir=f"{temp_dir}/encoded_filtered_train/shard{i:03d}",
            )
            for i, shard in enumerate(train_shards)
        )

    # Copy over train_{fold}.bin/idx files from /tmp/temp_binarized to data_bin/
    for train_fold, binarized_train_data in binarized_train_data_shards.items():
        if train_fold != "train":
            for i, binarized_train in enumerate(binarized_train_data):
                shard_dir = f"{output_dir}/data_bin/shard{i:03d}"
                if not os.path.exists(shard_dir):
                    os.makedirs(shard_dir, exist_ok=True)
                src_path_prefix = binarized_train.source
                src_basename_prefix = os.path.basename(src_path_prefix)
                src_basename_prefix = src_basename_prefix.replace("train", train_fold)
                execute_in_shell(
                    f"cp {src_path_prefix}.bin {shard_dir}/{src_basename_prefix}.bin"
                )
                execute_in_shell(
                    f"cp {src_path_prefix}.idx {shard_dir}/{src_basename_prefix}.idx"
                )
                tgt_path_prefix = binarized_train.target
                tgt_basename_prefix = os.path.basename(tgt_path_prefix)
                tgt_basename_prefix = tgt_basename_prefix.replace("train", train_fold)
                execute_in_shell(
                    f"cp {tgt_path_prefix}.bin {shard_dir}/{tgt_basename_prefix}.bin"
                )
                execute_in_shell(
                    f"cp {tgt_path_prefix}.idx {shard_dir}/{tgt_basename_prefix}.idx"
                )


def prepare_data_direction(
    direction: str,
    train_data_dict: Dict[str, Dict[str, data_types.CorporaMap]],
    valid_data: Dict[str, data_types.CorporaMap],
    sampled_train_dict: Dict[str, Dict[str, data_types.CorporaMap]],
    test_data: Dict[str, data_types.CorporaMap],
    src_vocab: data_types.BuiltVocab,
    tgt_vocab: data_types.BuiltVocab,
    all_num_shards: Dict[str, int],
    data_config: data_types.DataConfig,
    temp_dir: str,
    output_dir: str,
) -> None:
    logger.info(f"Preparing data for {direction}")
    seen_valid = set()
    seen_test = set()
    reverse_direction = "-".join(direction.split("-")[::-1])
    if valid_data and test_data:
        if direction in valid_data and direction in test_data:
            seen_valid = hash_parallel_data(valid_data[direction])
            seen_test = hash_parallel_data(test_data[direction])
        elif reverse_direction in valid_data and reverse_direction in test_data:
            seen_valid = hash_parallel_data(valid_data[reverse_direction])
            seen_test = hash_parallel_data(test_data[reverse_direction])
    seen = seen_valid.union(seen_test)
    prepare_valid_test_direction(
        direction,
        valid_data,
        test_data,
        sampled_train_dict,
        src_vocab,
        tgt_vocab,
        all_num_shards,
        data_config,
        temp_dir,
        output_dir=output_dir,
        custom_step_name=f"prepare_valid_test_direction.{direction}",
    )
    prepare_train_direction(
        direction,
        train_data_dict,
        src_vocab,
        tgt_vocab,
        all_num_shards,
        data_config,
        temp_dir,
        output_dir=output_dir,
        seen=seen,
        custom_step_name=f"prepare_train_directory.{direction}",
    )

    return "Done prepare data"


async def main(data_config: data_types.DataConfig, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "progress"), exist_ok=True)
    temp_dir = os.path.join(output_dir, "tmp")
    os.makedirs(temp_dir, exist_ok=True)

    (
        train_folds,
        train_src_counts_map,
        train_tgt_counts_map,
        train_counts_map,
        executor,
    ) = await validate_data_config(data_config, output_dir=output_dir)

    logger.info(f"Running prepare data with:\n{OmegaConf.to_yaml(data_config)}")
    print(
        OmegaConf.to_yaml(data_config),
        file=open(os.path.join(output_dir, "config.yaml"), "w"),
    )

    # wait for all retrieval
    retrieve_outdir = f"{output_dir}/retrieved_data"
    if not os.path.exists(retrieve_outdir):
        os.makedirs(retrieve_outdir, exist_ok=True)

    @cache_step("retrieve_data")
    async def retrieve_data_step(
        data_config: data_types.DataConfig, output_dir: str, train_folds: List[str]
    ):
        tasks = [
            retrieve_data(
                all_corpora_map=data_config.train_corpora,
                output_prefix=os.path.join(retrieve_outdir, "train"),
                preprocess_config=data_config.preprocessing_config,
                tag="train",
                output_dir=output_dir,
                executor=executor,
            )
        ]
        if train_folds:
            for train_fold in train_folds:
                tasks.append(
                    retrieve_data(
                        all_corpora_map=getattr(data_config, f"{train_fold}_corpora"),
                        output_prefix=os.path.join(retrieve_outdir, train_fold),
                        preprocess_config=data_config.preprocessing_config,
                        tag=train_fold,
                        output_dir=output_dir,
                        executor=executor,
                    )
                )
        tasks.extend(
            [
                retrieve_data(
                    all_corpora_map=data_config.valid_corpora,
                    output_prefix=os.path.join(retrieve_outdir, "valid"),
                    preprocess_config=data_config.preprocessing_config,
                    tag="valid",
                    output_dir=output_dir,
                    executor=executor,
                )
                if data_config.valid_corpora
                else async_noop(),
                retrieve_data(
                    all_corpora_map=data_config.test_corpora,
                    output_prefix=os.path.join(retrieve_outdir, "test"),
                    preprocess_config=data_config.preprocessing_config,
                    tag="test",
                    output_dir=output_dir,
                    executor=executor,
                )
                if data_config.test_corpora
                else async_noop(),
            ]
        )
        return await asyncio.gather(*tasks)

    (
        full_train_data,
        *full_train_folds_data_list,
        valid_data,
        test_data,
    ) = await retrieve_data_step(
        data_config=data_config,
        output_dir=output_dir,
        train_folds=train_folds,
    )

    if full_train_data is not None:
        train_data_dict = {"train": full_train_data[0]}
        sampled_train_dict = {"sampled_train": full_train_data[1]}
        for train_fold, full_train_folds_data in zip(
            train_folds, full_train_folds_data_list
        ):
            train_data_dict[train_fold] = full_train_folds_data[0]
            sampled_train_dict[f"sampled_{train_fold}"] = full_train_folds_data[1]
    else:
        train_data_dict = None
        sampled_train_dict = None

    if valid_data is not None:
        valid_data = valid_data[0]
    if test_data is not None:
        test_data = test_data[0]

    src_vocab, tgt_vocab = await get_vocab(
        data_config=data_config,
        train_corpora_dict=train_data_dict,
        src_counts_map=train_src_counts_map,
        tgt_counts_map=train_tgt_counts_map,
        output_dir=output_dir,
    )

    all_num_shards = get_all_num_shards(train_counts_map, data_config)
    if train_data_dict is not None:
        directions = list(set(list(train_counts_map.keys())))
    else:
        directions = set(train_counts_map.keys())

    jobs = []
    with executor.batch():
        for direction in directions:
            logger.info(f"Preparing data for {direction}")
            job = executor.submit(
                prepare_data_direction,
                direction,
                train_data_dict,
                valid_data,
                sampled_train_dict,
                test_data,
                src_vocab,
                tgt_vocab,
                all_num_shards,
                data_config,
                temp_dir,
                output_dir,
            )
            jobs.append(job)
    logger.info(f"All jobs have been scheduled")
    results = [job.result() for job in jobs]
    logger.info(f"All jobs have finished.")

    shutil.rmtree(temp_dir)

    logger.info(f"Data preparation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-file", default="prepare_data.log")
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    config_path, config_name = os.path.split(args.data_config)

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else f"{config_name}_{int(time.time())}"
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    fh = logging.FileHandler(filename=os.path.join(output_dir, args.log_file))
    logger.addHandler(fh)

    data_config = setup_config(args.data_path, args.data_config)
    asyncio.run(main(data_config, output_dir))
