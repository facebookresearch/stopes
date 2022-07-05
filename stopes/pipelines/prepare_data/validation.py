# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

from submitit import AutoExecutor

import stopes.pipelines.prepare_data.data_types as data_types
from stopes.pipelines.prepare_data.cache import cache_step
from stopes.pipelines.prepare_data.utils import count_lines

logger = logging.getLogger(__name__)


def validate_parallel_path(
    direction: str, corpus_name: str, parallel_dataset: data_types.ParallelDataset
):
    src_path = parallel_dataset.source
    tgt_path = parallel_dataset.target
    is_gzip = parallel_dataset.is_gzip
    assert os.path.exists(src_path), f"nonexistent source path: {src_path}"
    assert os.path.exists(tgt_path), f"nonexistent target path: {tgt_path}"
    num_src_lines = count_lines(src_path, is_gzip)
    num_tgt_lines = count_lines(tgt_path, is_gzip)
    if num_src_lines == num_tgt_lines:
        return direction, corpus_name, num_src_lines
    return direction, corpus_name, None


def _validate_corpora_items(corpora, executor=None):
    data_map = []
    jobs = []
    with executor.batch():
        for direction, corpora_confs in corpora.items():
            for corpus_name, parallel_paths in corpora_confs["values"].items():
                logger.info(f"Validating corpora for {direction} {corpus_name} job")
                job = executor.submit(
                    validate_parallel_path, direction, corpus_name, parallel_paths
                )
                jobs.append(job)

    data_map = [job.result() for job in jobs]

    return data_map


@cache_step("validate_data_config")
async def validate_data_config(
    data_config: data_types.DataConfig,
    output_dir: str,
) -> Tuple[List[str], Dict[str, int], Dict[str, int], Dict[str, int], AutoExecutor]:
    """
    Validate the following aspects of data_config:
        correct schema
        same list of directions in train_corpora and valid_corpora
        existence of training files
        same number of lines of source and target file
    Returns validated data_config or raises error for invalid data_config
    """
    logger.info("Validating values in input data_config\n")
    executor = AutoExecutor(
        folder=os.path.join(output_dir, data_config.executor_config.log_folder),
        cluster=data_config.executor_config.cluster,
    )
    executor.update_parameters(
        slurm_partition=data_config.executor_config.slurm_partition,
        timeout_min=1000,
        nodes=1,  # we only need one node for this
        cpus_per_task=8,
    )

    train_folds: List[str] = []
    train_directions = set(data_config.train_corpora.keys())
    train_counts_map_list = {
        "train": _validate_corpora_items(data_config.train_corpora, executor)
    }
    for field_name in dir(data_config):
        # If the field is of the form "train_{fold}_corpora"
        result = re.search(r"(train_.+)_corpora", field_name)
        if result:
            train_fold_corpora = getattr(data_config, field_name)
            if train_fold_corpora:
                train_fold = result.group(1)
                train_folds.append(train_fold)
                train_directions.update(set(train_fold_corpora.keys()))
                train_counts_map_list[train_fold] = _validate_corpora_items(
                    train_fold_corpora, executor
                )

    train_src_counts_map = defaultdict(int)
    train_tgt_counts_map = defaultdict(int)
    train_counts_map = defaultdict(int)
    logger.info("Checking line counts in training data")
    for _, train_counts_list in train_counts_map_list.items():
        for direction, corpus, num_lines in train_counts_list:
            if num_lines is None:
                assert (
                    False
                ), f"{direction}.{corpus} has inconsistent number of lines between source and target"
            source, target = direction.split("-")
            train_src_counts_map[source] += num_lines
            train_tgt_counts_map[target] += num_lines
            train_counts_map[direction] += num_lines

    if data_config.valid_corpora is not None:
        logger.info("Checking line counts in validation data")
        _validate_corpora_items(data_config.valid_corpora, executor)
    if data_config.test_corpora is not None:
        logger.info("Checking line counts in test data")
        _validate_corpora_items(data_config.test_corpora, executor)

    return (
        train_folds,
        train_src_counts_map,
        train_tgt_counts_map,
        dict(train_counts_map),
        executor,
    )
