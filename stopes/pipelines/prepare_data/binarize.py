# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import asyncio
import re
import shutil
import typing as tp
from functools import partial
from pathlib import Path

from omegaconf import DictConfig

from stopes.core import utils
from stopes.core.launcher import Launcher
from stopes.core.stopes_module import Requirements
from stopes.modules.preprocess.multiproc_line_processor import (
    MultiprocLineProcessorConfig,
    MultiprocLineProcessorModule,
)
from stopes.modules.preprocess.train_spm import Vocab
from stopes.pipelines.filtering.dataset import Dataset


def get_binarizer(
    binarize_num_workers: int,
    output_dir: Path,
    tmp_dir: Path,
    shards: tp.List[str],
    vocab: Vocab,
):
    binarizer = MultiprocLineProcessorModule(
        config=MultiprocLineProcessorConfig(
            line_processor=DictConfig(
                {
                    "_target_": "stopes.modules.preprocess.multiproc_fairseq_binarizer_encoder.MultiProcFairSeqBinarizerEncoder",
                    "vocab_file_path": str(vocab.dict_file),
                    "spm_model_path": str(vocab.model_file),
                    "dataset_impl": "mmap",
                }
            ),
            custom_name="MultiProcFairSeqBinarizerEncoder",
            output_dir=str(output_dir),
            outfile_prefix="",
            shards=shards,
            requirements=Requirements(
                nodes=1,
                tasks_per_node=1,
                gpus_per_node=0,
                cpus_per_task=binarize_num_workers,
                timeout_min=24 * 60,
            ),
            tmp_dir=str(tmp_dir),
        )
    )
    return binarizer


async def binarize(
    sharded_train_datasets: tp.List[Dataset],
    retrieved_eval_datasets: tp.List[Dataset],
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    max_num_shards: int,
    binarize_num_workers: int,
    train_src_counts_map: tp.Dict[str, int],
    train_tgt_counts_map: tp.Dict[str, int],
    launcher: Launcher,
    tmp_dir: Path,
    output_dir: Path,
):
    src_train_shards = [dataset.src for dataset in sharded_train_datasets]
    tgt_train_shards = [dataset.tgt for dataset in sharded_train_datasets]
    src_eval_shards = [dataset.src for dataset in retrieved_eval_datasets]
    tgt_eval_shards = [dataset.tgt for dataset in retrieved_eval_datasets]

    binarizer_fn = partial(get_binarizer, binarize_num_workers, output_dir, tmp_dir)

    src_train_binarizer = binarizer_fn(src_train_shards, src_vocab)
    tgt_train_binarizer = binarizer_fn(tgt_train_shards, tgt_vocab)
    src_eval_binarizer = binarizer_fn(src_eval_shards, src_vocab)
    tgt_eval_binarizer = binarizer_fn(tgt_eval_shards, src_vocab)

    _, _, src_eval_binarized, tgt_eval_binarized = await asyncio.gather(
        launcher.schedule(src_train_binarizer),
        launcher.schedule(tgt_train_binarizer),
        launcher.schedule(src_eval_binarizer),
        launcher.schedule(tgt_eval_binarizer),
    )

    # Copy train metadata to binarized subfolder.
    for dataset in sharded_train_datasets:
        if dataset.metadata:
            shard_str = re.search(r"shard\d{3}", dataset.metadata).group()
            outfile_name = Path(dataset.metadata).name.replace(f".{shard_str}", "")
            target_path = output_dir / shard_str / outfile_name
            shutil.copy(dataset.metadata, target_path)

    # Copy eval metadata to all shards.
    for dataset in retrieved_eval_datasets:
        if dataset.metadata:
            for i in range(max_num_shards):
                if i == 0:
                    target_path = (
                        output_dir / f"shard{i:03d}" / Path(dataset.metadata).name
                    )
                    shutil.copy(dataset.metadata, target_path)
                else:
                    new_target_path = (
                        output_dir / f"shard{i:03d}" / Path(dataset.metadata).name
                    )
                    utils.symlink(new_target_path, target_path)

    # Symlink binarized eval files to all shards.
    for i in range(1, max_num_shards):
        for orig_path in src_eval_binarized + tgt_eval_binarized:
            target_path = Path(str(orig_path).replace("shard000", f"shard{i:03d}"))
            utils.symlink(target_path, orig_path)

    # Symlink dict files to all shards.
    for i in range(max_num_shards):
        for src_lang in train_src_counts_map:
            target_path = output_dir / f"shard{i:03d}" / f"dict.{src_lang}.txt"
            utils.symlink(target_path, src_vocab.dict_file)
        for tgt_lang in train_tgt_counts_map:
            target_path = output_dir / f"shard{i:03d}" / f"dict.{tgt_lang}.txt"
            utils.symlink(target_path, tgt_vocab.dict_file)
