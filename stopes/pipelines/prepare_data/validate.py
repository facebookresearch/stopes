# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from collections import defaultdict
from dataclasses import dataclass

from omegaconf.omegaconf import MISSING

from stopes.core.launcher import Launcher
from stopes.core.stopes_module import Requirements, StopesModule
from stopes.core.utils import count_lines
from stopes.pipelines.filtering.dataset import Dataset


@dataclass
class ValidateDataConfig:
    datasets: tp.List[Dataset] = MISSING


class ValidateData(StopesModule):
    def __init__(
        self,
        config: ValidateDataConfig = ValidateDataConfig(),
    ):
        super().__init__(config, ValidateDataConfig)

    def array(self):
        return self.config.datasets

    def requirements(self) -> Requirements:
        return Requirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=1,
            timeout_min=24 * 60,
        )

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> tp.Tuple[tp.Optional[int], str, str]:
        assert iteration_value is not None, "iteration value is null"
        logger = logging.getLogger("stopes.prepare_data.validate")
        dataset = iteration_value
        logger.info(f"Validating dataset: {dataset}")

        num_src_lines = count_lines(dataset.src)
        num_tgt_lines = count_lines(dataset.tgt)
        num_metadata_lines = num_src_lines
        if dataset.metadata:
            num_metadata_lines = count_lines(dataset.metadata)
        if num_src_lines == num_tgt_lines == num_metadata_lines:
            return num_src_lines, dataset
        return None, dataset

    def validate(
        self,
        output: tp.Any,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> bool:
        num_lines, _ = output
        return num_lines is not None

    def version(self):
        return "0.0"


async def validate(
    datasets: tp.List[Dataset], launcher: Launcher
) -> tp.Tuple[tp.Dict[str, int], tp.Dict[str, int], tp.Dict[str, int]]:
    validation_module = ValidateData(ValidateDataConfig(datasets=datasets))
    validation_results = await launcher.schedule(validation_module)
    train_src_counts_map = defaultdict(int)
    train_tgt_counts_map = defaultdict(int)
    train_counts_map = defaultdict(int)

    for (num_lines, dataset) in validation_results:
        if dataset.fold.startswith("train"):
            src, tgt = dataset.lang_dir.split("-")
            train_src_counts_map[src] += num_lines
            train_tgt_counts_map[tgt] += num_lines
            train_counts_map[dataset.lang_dir] += num_lines

    return train_src_counts_map, train_tgt_counts_map, train_counts_map
