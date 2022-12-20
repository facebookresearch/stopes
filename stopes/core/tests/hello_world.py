# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
import typing as tp
from dataclasses import dataclass

from omegaconf.omegaconf import MISSING, OmegaConf

from stopes.core.launcher import Launcher
from stopes.core.stopes_module import Requirements, StopesModule


@dataclass
class HelloWorldConfig:
    greet: str = "hello"
    person: str = "world"
    duration: tp.Optional[float] = 0.1


class HelloWorldModule(StopesModule):
    def __init__(self, config):
        super().__init__(config, HelloWorldConfig)

    def requirements(self):
        return Requirements(
            nodes=1,
            mem_gb=10,
            tasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=1,
            timeout_min=60,
        )

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
        launcher: tp.Optional[Launcher] = None,
    ):
        print(OmegaConf.to_yaml(self.config))
        res = " ".join([self.config.greet, self.config.person, "!"])
        # Let the job sleep for a bit to simulate the module doing work
        if self.config.duration:
            time.sleep(self.config.duration)
        return res


@dataclass
class HelloWorldArrayConfig:
    greet: str = "hello"
    persons: tp.List[str] = MISSING
    duration: float = 0.1


class HelloWorldArrayModule(StopesModule):
    def __init__(
        self,
        config: HelloWorldArrayConfig = HelloWorldArrayConfig(),
    ):
        super().__init__(config, HelloWorldArrayConfig)
        self.logger = logging.getLogger("hello_world_array_module")

    def requirements(self):
        return Requirements(
            nodes=1,
            mem_gb=10,
            tasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=1,
            timeout_min=60,
        )

    def array(self):
        return self.config.persons

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        self.logger.info(f"On index: {iteration_index} & value: {iteration_value}")
        res = " ".join([self.config.greet, str(iteration_value), "!"])
        # Let the job sleep for a bit to simulate the module doing work
        time.sleep(self.config.duration)
        return res
