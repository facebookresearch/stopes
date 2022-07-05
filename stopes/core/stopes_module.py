# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import dataclasses
import logging
import typing as tp
from abc import ABC, abstractmethod
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import stopes.core

logger = logging.getLogger("stopes.module")

################################################################################
#  Module definition
################################################################################

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(process)d:%(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M",
)


class Requirements(ABC):
    pass


@dataclasses.dataclass
class DistributedRequirements(Requirements):
    nodes: int = 1
    mem_gb: tp.Optional[int] = None
    tasks_per_node: int = 1
    gpus_per_node: int = 1
    cpus_per_task: int = 5
    timeout_min: int = 720
    constraint: tp.Optional[str] = None


class LocalOnlyRequirements(Requirements):
    pass


class StopesModule(ABC):
    @staticmethod
    def build(config: tp.Any, **kwargs):
        """
        given a loaded config with a _target_ and a config entry, build the
        correct module.
        """
        merged_conf = OmegaConf.merge(config, {"config": kwargs})

        # hydra is good at that.
        return hydra.utils.instantiate(
            merged_conf,
            _recursive_=False,
        )

    def __init__(self, config: tp.Any, config_class: tp.Type = None):
        if dataclasses.is_dataclass(config):
            config = OmegaConf.structured(config)
        if config_class is not None:
            # Note: don't use ._promote since it merges the other way around:
            # the proto into the config.
            proto = OmegaConf.structured(config_class)
            proto.merge_with(config)
            self.config = proto
        else:
            assert isinstance(config, DictConfig), (
                "stopes module configs must be either a dataclass or a omega.DictConfig."
                f" Received a {type(config)}"
            )
            self.config = config
        OmegaConf.resolve(self.config)
        OmegaConf.set_readonly(self.config, True)

    def __call__(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
        cache: tp.Optional["stopes.core.Cache"] = None,
    ) -> tp.Any:
        """
        called when the job starts running.
        please implement `run` instead as we might need to add generic stuff in here
        """
        res = self.run(iteration_value=iteration_value, iteration_index=iteration_index)
        if not isinstance(res, tp.Coroutine):
            # Return result value in case of synchronous method call
            if cache is not None:
                cache.save_cache(self, res, iteration_value, iteration_index)
            return res

        # Handle async `run` implementation
        have_event_loop = True
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # 'RuntimeError: There is no current event loop.
            have_event_loop = False

        # TODO: Explain more when we can return a coroutine
        # This is weird: depending on the context we either return a result
        # or a coroutine.
        if have_event_loop:
            # this should be awaited by whoever is calling the raw module
            return res
        else:
            # We are in a separate process, run it with asyncio
            actual_result = asyncio.run(res)
            if cache is not None:
                cache.save_cache(self, actual_result, iteration_value, iteration_index)
            return actual_result

    @abstractmethod
    async def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> tp.Any:
        """
        the core of your module, implement your logic here.

        If `array` returned an array, this is an array job and this `run` be
        called for each iteration, the value of the specific iteration in the array
        will be passed down to you. If it's not an array job, this will be None.
        """
        ...

    def array(self) -> tp.Optional[tp.List[tp.Any]]:
        """
        if you want to submit your job as an array job, you can compute
        the array of values you want to process here. This will be processed
        before the job is submitted to the particular cluster chosen.

        By default, we return None to indicate this is not an array job.
        """
        return None

    def requirements(self) -> Requirements:
        """
        return a set of Requirements for your module, like num of gpus etc.
        If you return None, this will be launched "inline" without scheduling any new job.
        """
        return LocalOnlyRequirements()

    def name(self):
        """
        implement this if you want to give a fancy name to your job
        """
        # TODO ideally use hydra override_dirname here
        return "_".join([self.__class__.__name__, self.sha_key()])

    def cache_key(self):
        return (
            self.__class__.__module__,
            self.__class__.__qualname__,
            self.version(),
            self.config,
        )

    # TODO: @functools.cached_property()
    # This is only safe if cache_key is not allowed to change, in particular if config is frozen.
    # Can we guarantee that ?
    def sha_key(self):
        return stopes.core.utils.sha_key(repr(self.cache_key()))

    def comment(self):
        """
        same as `name` but for the job comment
        """
        return None

    @classmethod
    def version(cls):
        """
        the version of the module. If you want to invalidate
        some cache, you can change that
        """
        return "0.0"

    def validate(
        self,
        output: tp.Any,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> bool:
        """
        Validate the output of this module (for a single step of the array if it's an array module).
        This is called to invalidate the cache when needed but also at the end of the module run for
        sanity check.
        You can either return False or Raise/throw an exception if the results are not valid.

        The default implementation checks if the content of pickle is a "Path",
        and invalidate cache if the corresponding file is gone.
        """
        if isinstance(output, Path) and not output.exists():
            logger.warning(
                f"Cache for: {self.name()} iteration {iteration_index}"
                f"points to missing file {output}, will invalidate it."
            )
            return False
        return True
