# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import typing as tp
from dataclasses import dataclass

from omegaconf.omegaconf import MISSING

from stopes.core.stopes_module import DistributedRequirements, StopesModule


@dataclass
class HelloWorldArrayConfig:
    iteration_values: tp.List[str] = MISSING


class HelloWorldArrayModule(StopesModule):
    def __init__(
        self,
        config: HelloWorldArrayConfig = HelloWorldArrayConfig(),
    ):
        super().__init__(config)
        self.logger = logging.getLogger("hello_world_array_module")

    def requirements(self):
        return DistributedRequirements(
            nodes=1,
            mem_gb=10,
            tasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=1,
            timeout_min=60,
        )

    def array(self):
        return self.config.iteration_values

    async def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        self.logger.info(f"On index: {iteration_index} & value: {iteration_value}")
        # Let the job sleep for a bit to simulate the module doing work
        await asyncio.sleep(60)
        return iteration_value
