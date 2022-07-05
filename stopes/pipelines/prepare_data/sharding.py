# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import os
from typing import Dict

import stopes.pipelines.prepare_data.data_types as data_types
from stopes.pipelines.prepare_data.utils import execute_in_shell

logger = logging.getLogger(__name__)


def get_all_num_shards(
    train_counts_map: Dict[str, int],
    data_config: data_types.DataConfig,
):
    total_train_size = sum(train_counts_map.values())
    if total_train_size == 0:
        return {}
    directions = list(set(list(train_counts_map.keys())))
    max_num_shards = math.ceil(
        2
        ** math.ceil(
            math.log2(
                total_train_size
                / data_config.binarization_config.max_examples_per_shard
            )
        )
    )

    all_num_shards = {
        direction: get_num_shards(
            num_lines=train_counts_map.get(direction, 0),
            smallest_shard=data_config.binarization_config.smallest_shard,
            max_num_shards=max_num_shards,
        )
        for direction in directions
    }
    return all_num_shards


def get_num_shards(num_lines: int, smallest_shard: int, max_num_shards: int) -> int:
    """
    Calculate the num of shards based on total lines and config
    Returns the num of shards
    """
    if num_lines <= 0:
        return 0

    num_shards = math.ceil(2 ** math.ceil(math.log2(num_lines / smallest_shard)))
    return min(num_shards, max_num_shards)


def write_to_all_shards(
    parallel_pref: data_types.ParallelDataset, num_shards: int, output_dir: str
) -> None:
    """
    Copy binarized valid/test data to all shards
    """
    logger.info("write_to_all_shards")
    logger.info(f"parallel_pref: {parallel_pref}\n")

    for i in range(num_shards):
        shard_dir = f"{output_dir}/shard{i:03d}"
        if not os.path.exists(shard_dir):
            os.makedirs(shard_dir, exist_ok=True)
        src_path_prefix = parallel_pref.source
        src_basename_prefix = os.path.basename(src_path_prefix)
        execute_in_shell(
            f"cp {src_path_prefix}.bin {shard_dir}/{src_basename_prefix}.bin"
        )
        execute_in_shell(
            f"cp {src_path_prefix}.idx {shard_dir}/{src_basename_prefix}.idx"
        )
        tgt_path_prefix = parallel_pref.target
        tgt_basename_prefix = os.path.basename(tgt_path_prefix)
        execute_in_shell(
            f"cp {tgt_path_prefix}.bin {shard_dir}/{tgt_basename_prefix}.bin"
        )
        execute_in_shell(
            f"cp {tgt_path_prefix}.idx {shard_dir}/{tgt_basename_prefix}.idx"
        )
    return
