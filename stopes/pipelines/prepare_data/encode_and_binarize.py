# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Optional, Tuple

import stopes.pipelines.prepare_data.data_types as data_types
from stopes.pipelines.prepare_data.cache import cache_step_sync
from stopes.pipelines.prepare_data.spm_tokenizer import SPMTokenizer
from stopes.pipelines.prepare_data.utils import execute_in_shell, split_direction

logger = logging.getLogger(__name__)


def _binarize(
    source: str,
    target: str,
    train_pref: str,
    valid_pref: str,
    test_pref: str,
    src_dict: str,
    tgt_dict: str,
    binarize_workers: int,
    output_dir: str,
) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    train_pref = f"--trainpref {train_pref}" if train_pref else ""
    valid_pref = f"--validpref {valid_pref}" if valid_pref else ""
    test_pref = f"--testpref {test_pref}" if test_pref else ""
    logger.info(f"Binarizing data: {train_pref} {valid_pref} {test_pref}\n")

    command = f"""export MKL_SERVICE_FORCE_INTEL=1 && fairseq-preprocess \
        --source-lang {source} \
        --target-lang {target} \
        {train_pref} \
        {valid_pref} \
        {test_pref} \
        --srcdict {src_dict} \
        --tgtdict {tgt_dict} \
        --destdir {output_dir} --workers {binarize_workers} \
        > {output_dir}/preprocess.{source}-{target}.log"""
    execute_in_shell(command)
    return


@cache_step_sync("length_based_filtering")
def _length_based_filtering(
    spm_path: str,
    src_lang: str,
    tgt_lang: str,
    tag: str,
    encoded_outdir: str,
    output_dir: str,
    custom_step_name: str,
) -> str:
    logger.info(f"Length based filtering for {tag} {spm_path} {src_lang} {tgt_lang}")
    if not os.path.exists(encoded_outdir):
        os.makedirs(encoded_outdir, exist_ok=True)
    if os.path.exists(
        f"{encoded_outdir}/spm_length_filtered_{tag}.{src_lang}-{tgt_lang}.{src_lang}"
    ):
        os.remove(
            f"{encoded_outdir}/spm_length_filtered_{tag}.{src_lang}-{tgt_lang}.{src_lang}"
        )
    if os.path.exists(
        f"{encoded_outdir}/spm_length_filtered_{tag}.{src_lang}-{tgt_lang}.{tgt_lang}"
    ):
        os.remove(
            f"{encoded_outdir}/spm_length_filtered_{tag}.{src_lang}-{tgt_lang}.{tgt_lang}"
        )
    command = (
        "perl examples/nllb/modeling/preprocessing/moses/clean-corpus-n.perl "
        "--ratio 3 "
        f"{spm_path} "
        f"{src_lang} "
        f"{tgt_lang} "
        f"{encoded_outdir}/spm_length_filtered_{tag}.{src_lang}-{tgt_lang} "
        "1 "
        "250 "
    )
    execute_in_shell(command)
    return f"{encoded_outdir}/spm_length_filtered_{tag}.{src_lang}-{tgt_lang}"


@cache_step_sync("binarize_step")
def _binarize_step(
    direction: str,
    src_vocab: data_types.BuiltVocab,
    tgt_vocab: data_types.BuiltVocab,
    encoded_prefix: str,
    tag: str,
    binarize_workers: int,
    binarized_outdir: str,
    output_dir: str,
    custom_step_name: str,
):
    source, target = split_direction(direction)
    _binarize(
        source,
        target,
        encoded_prefix if tag.split("_")[0] == "train" else None,
        encoded_prefix if tag == "valid" or tag.split("_")[0] == "sampled" else None,
        encoded_prefix if tag == "test" else None,
        src_vocab.dict_file,
        tgt_vocab.dict_file,
        binarize_workers,
        binarized_outdir,
    )
    file_tag = "train" if tag.split("_")[0] == "train" else tag
    # file_tag = "valid" if tag == "sampled_train" else tag
    binarize_results = data_types.ParallelDataset(
        source=f"{binarized_outdir}/{file_tag}.{direction}.{source}",
        target=f"{binarized_outdir}/{file_tag}.{direction}.{target}",
    )  # parallel_data for prefix
    return binarize_results


def encode_spm(
    raw_text_path: str,
    vocab: data_types.BuiltVocab,
    output_dir: str,
) -> Tuple[str, str]:
    logger.info(f"Encoding spm {raw_text_path} to {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    spm_path = SPMTokenizer()(raw_text_path, output_dir, vocab)
    return spm_path


@cache_step_sync("encode_spm_step")
def _encode_spm_step(
    parallel_data: data_types.ParallelDataset,
    src_vocab: data_types.BuiltVocab,
    tgt_vocab: data_types.BuiltVocab,
    encoded_outdir: str,
    output_dir: str,
    custom_step_name: str,
):
    return (
        encode_spm(
            parallel_data.source,
            src_vocab,
            f"{encoded_outdir}",
        ),
        encode_spm(
            parallel_data.target,
            tgt_vocab,
            f"{encoded_outdir}",
        ),
    )


@cache_step_sync("encode_and_binarize")
def encode_and_binarize(
    direction: str,
    parallel_data: data_types.ParallelDataset,
    tag: str,
    src_vocab: data_types.BuiltVocab,
    tgt_vocab: data_types.BuiltVocab,
    binarize_workers: int,
    output_dir: str,
    encoded_outdir: str,
    binarized_outdir: str,
    shard_id: int,
    custom_step_name: str,
    encoded_filtered_outdir: Optional[str] = None,
) -> data_types.ParallelDataset:
    """
    Encode and binarize parallel data
    Returns binarized file prefixes in parallel_data format
    """
    if not parallel_data:
        return None
    # encode source, target
    source, target = split_direction(direction)

    (encoded_src_path, encoded_tgt_path) = _encode_spm_step(
        output_dir=output_dir,
        parallel_data=parallel_data,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        encoded_outdir=encoded_outdir,
        custom_step_name=f"encode_spm_{tag}.{direction}.shard{shard_id}",
    )

    encoded_prefix = encoded_src_path.rsplit(".", 1)[0]

    # length based filtering
    if encoded_filtered_outdir is not None:
        encoded_prefix = _length_based_filtering(
            spm_path=encoded_prefix,
            src_lang=source,
            tgt_lang=target,
            tag=tag,
            encoded_outdir=encoded_filtered_outdir,
            output_dir=output_dir,
            custom_step_name=f"length_filtered_{tag}.{direction}.shard{shard_id}",
        )

    binarize_results = _binarize_step(
        direction=direction,
        output_dir=output_dir,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        encoded_prefix=encoded_prefix,
        tag=tag,
        binarized_outdir=binarized_outdir,
        binarize_workers=binarize_workers,
        custom_step_name=f"binarize_{tag}.{direction}.shard{shard_id}",
    )

    return binarize_results
