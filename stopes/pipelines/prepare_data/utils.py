# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import errno
import gzip
import hashlib
import logging
import os
import pickle
import re
import subprocess
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path
from typing import List, Set, Tuple, TypeVar

import numpy as np
from omegaconf import OmegaConf
from submitit import AutoExecutor, Job

import stopes.core.utils as utils
import stopes.pipelines.prepare_data.data_types as data_types
from stopes.pipelines.prepare_data.cache import cache_step_sync

JT = TypeVar("JT")

logger = logging.getLogger(__name__)


@lru_cache
def split_direction(direction: str) -> Tuple[str, str]:
    source, target = direction.split("-", maxsplit=1)
    return (source, target)


def setup_config(config_path: str) -> data_types.DataConfig:
    data_config = OmegaConf.load(config_path)
    schema = OmegaConf.structured(data_types.DataConfig)
    return OmegaConf.merge(schema, data_config)


def count_lines(fi: str, is_gzip: bool) -> int:
    open_func = gzip.open if is_gzip else open
    with open_func(fi, "rb") as f:
        count = sum(1 for line in f)
    return count


async def count_lines_async(
    file_name: str, is_gzip: bool, executor: AutoExecutor
) -> int:
    return count_lines(file_name, is_gzip)


def hash_sentence_pair(source_line: str, target_line: str) -> str:
    concated = f"{source_line} {target_line}"
    hashed = hashlib.md5(concated.encode()).hexdigest()
    return hashed


def hash_sentence(source_line: str) -> str:
    hashed = hashlib.md5(f"{source_line}".encode()).hexdigest()
    return hashed


def hash_parallel_data(parallel_data: data_types.CorporaMap) -> Set[str]:
    """
    Get the set of hashed lines in valid/test data
    """
    seen = set()
    with open(parallel_data.source) as s_f, open(parallel_data.target) as t_f:
        for source_line, target_line in zip(s_f, t_f):
            hashed = hash_sentence_pair(source_line, target_line)
            seen.add(hashed)
    return seen


@cache_step_sync("dedup_sharding")
def dedup_sharding(
    direction: str,
    train_parallel: data_types.ParallelDataset,
    seen: Set[str],
    num_shards: int,
    binarization_config: data_types.BinarizationConfig,
    sharding_output_dir: str,
    train_fold: str,
    output_dir: str,
    custom_step_name: str,
) -> List[data_types.ParallelDataset]:
    """
    Deduplicate training data appeared in valid & test data;
    execute random sharding to training data for one direction
    Returns list of sharded files prefixes
    """
    logger.info("dedup and sharding")
    logger.info(f"train_parallel: {train_parallel}\n")
    if train_parallel is None:
        logger.info(f"No data provided {custom_step_name}")
        return []

    if not os.path.exists(sharding_output_dir):
        os.makedirs(sharding_output_dir, exist_ok=True)
    src, tgt = direction.split("-")

    has_metadata = train_parallel.metadata is not None

    # sharding prepare
    source_shards = []
    target_shards = []
    metadata_shards = []

    for i in range(num_shards):
        execute_in_shell(f"mkdir -p {sharding_output_dir}/shard{i:03d}")
        source_shards.append(
            open(
                f"{sharding_output_dir}/shard{i:03d}/{train_fold}.shard{i:03d}.{src}-{tgt}.{src}",
                "wb",
            )
        )
        target_shards.append(
            open(
                f"{sharding_output_dir}/shard{i:03d}/{train_fold}.shard{i:03d}.{src}-{tgt}.{tgt}",
                "wb",
            )
        )
        if has_metadata:
            metadata_shards.append(
                open(
                    f"{sharding_output_dir}/shard{i:03d}/{train_fold}.shard{i:03d}.{src}-{tgt}.meta",
                    "w",
                )
            )
    np.random.seed(binarization_config.random_seed)
    num_lines = count_lines(
        train_parallel.source, train_parallel.is_gzip
    )  # using non-deduped data for random sharding
    idx_to_shards = np.random.randint(0, num_shards, num_lines)
    eval_hash_seen = set()

    idx = 0
    meta_f = open(train_parallel.metadata, "r") if has_metadata else None

    with open(train_parallel.source, "rb") as src_f, open(
        train_parallel.target, "rb"
    ) as tgt_f:
        for source_line, target_line in zip(src_f, tgt_f):
            if has_metadata:
                meta_line = meta_f.readline()
            # dedup
            source_hash = hash_sentence(source_line)
            target_hash = hash_sentence(target_line)
            hashed = hash_sentence_pair(source_line, target_line)
            if (
                hashed in seen
                or source_hash in eval_hash_seen
                or target_hash in eval_hash_seen
                or len(source_line.strip()) < 1
                or len(target_line.strip()) < 1
            ):
                pass
            else:
                seen.add(hashed)
                eval_hash_seen.add(source_hash)
                eval_hash_seen.add(target_hash)
                # sharding
                shard_id = idx_to_shards[idx]
                source_shards[shard_id].write(source_line)
                target_shards[shard_id].write(target_line)
                if has_metadata:
                    metadata_shards[shard_id].write(meta_line)
            idx += 1

    if has_metadata:
        meta_f.close()
    for fi in source_shards + target_shards + metadata_shards:
        fi.flush()
        fi.close()

    return [
        data_types.ParallelDataset(
            source=f"{sharding_output_dir}/shard{i:03d}/{train_fold}.shard{i:03d}.{src}-{tgt}.{src}",
            target=f"{sharding_output_dir}/shard{i:03d}/{train_fold}.shard{i:03d}.{src}-{tgt}.{tgt}",
            metadata=f"{sharding_output_dir}/shard{i:03d}/{train_fold}.shard{i:03d}.{src}-{tgt}.meta"
            if has_metadata
            else None,
        )
        for i in range(num_shards)
    ]


def clean_corpus_n(
    corpus: str,
    l1: str,
    l2: str,
    out: str,
    minc: int,
    maxc: int,
    meta: str = None,
    ratio: int = 9,
    retained: str = None,
    lc: bool = False,
    ignore_ratio: bool = False,
    ignore_xml: bool = False,
):
    """
    Filter a bitext by removing sentence pairs where a side is shorter than minc, longer than maxc or length ratio is larger than ratio

    Python port of the clean-corpus-n.perl script from Moses with some changes:
        - does not take factors into account

    Parameters:
        corpus (str): corpus name. The function will process corpus.{l1,l2}[.gz]
        l1 and l2 (str): the two languages
        out (str): output directory
        minc and maxc (int): min and max size for a segment
        ratio (int): length ratio between source and target lengths
        retained (str): file containing the retained lines numbers
        lc: should we lowercase the data? (default False)
        ignore_ratio and ignore_xml: ignore ratio or xml tags
    Outputs:
        Processed files are put in out/corpus.{l1,l2}[.gz]
    Returns:
        Nothing
    """
    logger.info(
        f"clean_corpus_n: processing {corpus}.{l1} & .{l2} to {out}, cutoff {minc}-{maxc}, ratio {ratio}, {'without' if meta is None else 'with'} metadata"
    )

    # Compile useful patterns
    space_pat = re.compile(r"\s+")

    with ExitStack() as stack:
        meta_file = (
            stack.enter_context(utils.open(Path(f"{corpus}.{meta}"))) if meta else None
        )
        l1_out = stack.enter_context(utils.open(f"{out}.{l1}", "w"))
        l2_out = stack.enter_context(utils.open(f"{out}.{l2}", "w"))
        meta_out = (
            stack.enter_context(utils.open(f"{out}.{meta}", "w")) if meta_file else None
        )
        retained_file = (
            stack.enter_context(utils.open(retained, "w")) if retained else None
        )
        innr = 0
        outnr = 0
        l1_file = stack.enter_context(utils.open(Path(f"{corpus}.{l1}")))
        l2_file = stack.enter_context(utils.open(Path(f"{corpus}.{l2}")))
        for l1_line, l2_line in zip(l1_file, l2_file):
            if meta_file:
                meta_line = meta_file.readline().strip()
            innr += 1
            # logger.info(".") if (innr%1000 == 0)
            # if lowercasing, lowercase
            if lc is True:
                l1_line = l1_line.lower()
                l2_line = l2_line.lower()

            l1_line = space_pat.sub(" ", l1_line)
            l1_line = l1_line.strip()
            l2_line = space_pat.sub(" ", l2_line)
            l2_line = l2_line.strip()

            if not l1_line or not l2_line:
                continue

            def word_count(line: str, ignore_xml=False):
                if ignore_xml is True:
                    line = re.sub(r"<\S[^>]*\S>", " ", line)
                    line = space_pat.sub(" ", line)
                    line = line.strip()
                words = line.split(" ")
                return len(words)

            l1_count = word_count(l1_line, ignore_xml=ignore_xml)
            l2_count = word_count(l2_line, ignore_xml=ignore_xml)
            if (l1_count < minc or l1_count > maxc) or (
                l2_count < minc or l2_count > maxc
            ):
                continue
            if (
                (ignore_ratio is False)
                and (l1_count / l2_count > ratio)
                or (l2_count / l1_count > ratio)
            ):
                continue

            outnr += 1
            l1_out.write(f"{l1_line}\n")
            l2_out.write(f"{l2_line}\n")
            if meta_out is not None:
                meta_out.write(f"{meta_line}\n")
            if retained_file is not None:
                retained_file.write(f"{innr}\n")


async def awaitable_job(job: Job[JT], poll_s: int = 1) -> JT:
    while not job.done():
        await asyncio.sleep(poll_s)
    return job.result()


def execute_in_shell(command, shell=True, dry_run=False, quiet=True):
    """Execute commands in the shell

    Args:
        command ([type]): str or list commands (type needs to correspond to the value of the shell)
        shell (bool, optional): controls the command type (True: str, False: list). Defaults to True.
        dry_run (bool, optional): print out commands without real execution. Defaults to False.
        quiet (bool, optional): controls whether to print information. Defaults to True.
    """
    if dry_run:
        if not quiet:
            logger.info(f"dry run command: {command}")
    else:
        with subprocess.Popen(command, stdout=subprocess.PIPE, shell=shell) as proc:
            if not quiet:
                logger.info(proc.stdout.read().decode("utf-8"))


async def async_noop():
    """
    useful to return none in awaitable ternaries
    """
    return None
