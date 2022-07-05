# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import pickle
import typing as tp
from abc import ABC, abstractmethod
from pathlib import Path

import submitit
import tqdm
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf
from submitit import AutoExecutor
from tqdm.contrib.logging import logging_redirect_tqdm

from stopes.core.jobs_registry.registry import JobsRegistry
from stopes.core.jobs_registry.submitit_slurm_job import SubmititJob
from stopes.core.stopes_module import DistributedRequirements, LocalOnlyRequirements
from stopes.core.utils import sha_key

if tp.TYPE_CHECKING:
    from stopes.core import StopesModule

logger = logging.getLogger("stopes.launcher")


################################################################################
#  Caching definition
################################################################################


class MissingCache(Exception):
    """Raised when we do not find the cache"""

    pass


class Cache(ABC):
    def get_cache_key(
        self,
        module: "StopesModule",
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> str:
        return sha_key(
            self._raw_key(
                module,
                iteration_value,
                iteration_index,
            )
        )

    def _raw_key(
        self,
        module: "StopesModule",
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> str:
        local_key = ".".join(
            [self.__class__.__module__, self.__class__.__qualname__, self.version()]
        )
        cache_key = module.cache_key() + (iteration_value, iteration_index, local_key)
        k = repr(cache_key)
        return k

    @abstractmethod
    def get_cache(
        self,
        module: "StopesModule",
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
        validate: bool = True,
    ) -> tp.Optional[tp.Any]:
        """
        gets the cached value for this module. If no cache is found
        raise MissingCache.
        """
        ...

    @abstractmethod
    def save_cache(
        self,
        module: "StopesModule",
        value: tp.Any,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        ...

    @abstractmethod
    def invalidate_cache(
        self,
        module: "StopesModule",
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        ...

    def invalidate_module_cache(
        self,
        module: "StopesModule",
    ) -> None:
        array = module.array()
        if array is None:
            array = [None]
        for idx, val in enumerate(array):
            self.invalidate_cache(module, val, idx)

    def version(self):
        return "0.0"


class NoCache(Cache):
    """
    cache that doesn't cache. Useful to keep the same logic even
    if we don't use caching
    """

    def get_cache(
        self,
        module: "StopesModule",
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
        validate: bool = True,
    ) -> tp.Optional[tp.Any]:
        raise MissingCache()

    def save_cache(
        self,
        module: "StopesModule",
        value: tp.Any,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        pass

    def invalidate_cache(
        self,
        module: "StopesModule",
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        pass


# TODO add LRU
class FileCache(Cache):
    def __init__(self, caching_dir: Path):
        self.caching_dir = Path(caching_dir)
        self.caching_dir.mkdir(exist_ok=True)

    def get_cache_file_path(
        self,
        module: "StopesModule",
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> Path:
        cache_key = self.get_cache_key(
            module, iteration_value=iteration_value, iteration_index=iteration_index
        )
        return self.caching_dir / f"{cache_key}.pickle"

    def get_config_file_path(
        self,
        module: "StopesModule",
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> Path:
        cache_key = self.get_cache_key(
            module, iteration_value=iteration_value, iteration_index=iteration_index
        )
        return (
            self.caching_dir / f"{module.name()}.{cache_key}.{iteration_index:03d}.yaml"
        )

    def get_cache(
        self,
        module: "StopesModule",
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
        validate: bool = True,
    ) -> tp.Optional[tp.Any]:
        cache_file_path = self.get_cache_file_path(
            module, iteration_value=iteration_value, iteration_index=iteration_index
        )

        if not cache_file_path.is_file():
            raise MissingCache()

        try:
            with cache_file_path.open("rb") as cache_file:
                cached = pickle.load(cache_file)
        except Exception as e:
            logger.warning(
                f"couldn't load cache from {cache_file_path} for: "
                f"{module.name()} iteration {iteration_index}",
                exc_info=e,
            )
            cache_file_path.unlink()
            raise MissingCache()

        if validate:
            try:
                valid_result = module.validate(cached, iteration_value, iteration_index)
            except Exception as e:
                logger.warning(f"cache is invalid for {module.name()}", exc_info=e)
                valid_result = False
            if not valid_result:
                self.invalidate_cache(module, iteration_value, iteration_index)
                raise MissingCache()

        return cached

    def save_cache(
        self,
        module: "StopesModule",
        result: tp.Any,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        try:
            cache_file_path = self.get_cache_file_path(
                module, iteration_value=iteration_value, iteration_index=iteration_index
            )
            with cache_file_path.open("wb") as cache_file:
                pickle.dump(result, cache_file)
            cache_cfg_path = self.get_config_file_path(
                module, iteration_value=iteration_value, iteration_index=iteration_index
            )
            OmegaConf.save(config=module.config, f=cache_cfg_path)
            logger.info(f"written cache for {module.name()}:{iteration_index}")
        except Exception as e:
            logger.warning(
                f"couldn't write cache for {cache_file_path} for: {module.name()}:{iteration_index}",
                exc_info=e,
            )

    def invalidate_cache(
        self,
        module: "StopesModule",
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> None:
        try:
            logger.info(f"removing cache for {module.name()}:{iteration_index}")
            cache_file_path = self.get_cache_file_path(
                module, iteration_value=iteration_value, iteration_index=iteration_index
            )
            cache_file_path.unlink()
        except Exception as e:
            logger.warning(
                f"couldn't invalidate cache for {cache_file_path} for: {module.name()}:{iteration_index}",
                exc_info=e,
            )


################################################################################
#  Launchers definition
################################################################################


class Launcher(ABC):
    def __init__(
        self, cache: tp.Optional[Cache], config_dump_dir: tp.Optional[str] = None
    ):
        """
        If this was started by another launcher, it will be passed down, other launcher is None.
        """
        self.cache = NoCache() if cache is None else cache
        self.config_dump_dir = (
            config_dump_dir
            if config_dump_dir is not None
            else os.path.join(os.getcwd(), "config_logs")
        )
        os.makedirs(self.config_dump_dir, exist_ok=True)
        self.jobs_registry: JobsRegistry = JobsRegistry()

    async def schedule(self, module: "StopesModule"):
        with logging_redirect_tqdm():
            OmegaConf.save(
                config=module.config,
                f=os.path.join(self.config_dump_dir, f"{module.name()}.yaml"),
            )
            value_array = module.array()
            if value_array is not None:
                result = await self._schedule_array(module, value_array)
            else:
                result = await self._schedule_single(module)
            self.jobs_registry.log_progress()
            return result

    async def _await_and_cache(
        self,
        module: "StopesModule",
        not_cached: tp.List[tp.Tuple[int, tp.Any]],
        value_array: tp.List[tp.Any],
    ) -> tp.AsyncIterator[tp.Tuple[int, tp.Any]]:
        reqs = module.requirements()
        # it is an iterator of awaitables that we can wait for in as_completed
        it = (
            [
                module(iteration_value=val, iteration_index=idx, cache=self.cache)
                for (idx, val) in not_cached
            ]  # if we are only running locally, let's just call module directly here
            if isinstance(reqs, LocalOnlyRequirements)
            else self.uncached_schedule_iterator(
                module, not_cached
            )  # otherwise delegate to the real scheduler
        )

        # now await for the results of each iteration, use tqdm to show a progress bar
        for completed_result in tqdm.asyncio.tqdm.as_completed(
            it,
            desc=module.name(),
            total=len(not_cached),
        ):
            # we cache as they complete
            idx, result = await completed_result
            if not module.validate(
                result, iteration_value=value_array[idx], iteration_index=idx
            ):
                raise ValueError(f"invalid result for {module.name()}:{idx}")
            yield (idx, result)

    async def _schedule_array(
        self, module: "StopesModule", value_array: tp.List[tp.Any]
    ) -> tp.List[tp.Any]:
        """
        check if any item is in the cache, for the rest compute it
        """
        not_cached = []
        already_cached = []
        for idx, val in enumerate(value_array):
            try:
                cached_result = self.cache.get_cache(
                    module, iteration_value=val, iteration_index=idx
                )
                already_cached.append((idx, cached_result))
            except MissingCache:
                not_cached.append((idx, val))

        logger.info(
            f"for {module.name()} found {len(already_cached)} already cached array results,"
            f"{len(not_cached)} left to compute out of {len(value_array)}"
        )

        computed = (
            [r async for r in self._await_and_cache(module, not_cached, value_array)]
            if len(not_cached) > 0
            else []
        )
        return [
            result
            for _, result in sorted(already_cached + list(computed), key=lambda r: r[0])
        ]

    async def _schedule_single(self, module: "StopesModule") -> tp.Any:
        try:
            cached_result = self.cache.get_cache(module)
            logger.info(f"{module.name()} done from cache")
            return cached_result
        except MissingCache:
            pass

        reqs = module.requirements()
        result = (
            # if we are only running locally, let's just call module directly here
            await module(cache=self.cache)
            if isinstance(reqs, LocalOnlyRequirements)
            else await self.uncached_schedule_single(module)
        )
        if not module.validate(result):
            raise ValueError(f"invalid result for {module.name()}")
        logger.info(f"{module.name()} done after full execution")
        return result

    @abstractmethod
    def uncached_schedule_iterator(
        self,
        module: "StopesModule",
        array_with_indexes: tp.Iterable[tp.Tuple[int, tp.Any]],
    ) -> tp.Iterable[tp.Coroutine[tp.Tuple[int, tp.Any], None, None]]:
        """
        this should execute a module for an array of values,
        wait for all results to execute and return all results.

        The values and their index is passed in and the return should be the values and their index as tupples.
        This is because with caching, this function might receive a parse array to execute.

        returns an iterator of coroutines that can be awaited
        """
        ...

    @abstractmethod
    async def uncached_schedule_single(self, module: "StopesModule"):
        """
        this should execute a module in the non array case (single execution),
        wait for it to execute and return its results.
        """
        ...


class SubmititLauncher(Launcher):
    def __init__(
        self,
        cache: tp.Optional[Cache] = None,
        config_dump_dir: tp.Union[Path, str, None] = None,
        log_folder: tp.Union[Path, str] = Path("logs"),
        cluster: str = "local",
        partition: str = "devaccel",
    ):
        super().__init__(cache, config_dump_dir)
        self.log_folder = log_folder
        self.cluster = cluster
        self.partition = partition

        self.executor = AutoExecutor(folder=self.log_folder, cluster=cluster)

    def _setup_parameters(self, module: "StopesModule"):
        reqs = module.requirements()
        assert isinstance(
            reqs, DistributedRequirements
        ), "unsupported requirement type."
        self.executor.update_parameters(
            name=module.name(),
            slurm_partition=self.partition,
        )
        comment = module.comment()
        if comment is not None:
            self.executor.update_parameters(
                slurm_comment=comment,
            )
        if reqs is not None:
            mapped_reqs = {
                "name": module.name(),
                "nodes": reqs.nodes,
                "tasks_per_node": reqs.tasks_per_node,
                "gpus_per_node": reqs.gpus_per_node,
                "cpus_per_task": reqs.cpus_per_task,
                "timeout_min": reqs.timeout_min,
            }
            if getattr(reqs, "mem_gb", None):
                mapped_reqs["mem_gb"] = reqs.mem_gb
            if hasattr(reqs, "contraints"):
                mapped_reqs["slurm_constraint"] = (reqs.constraint,)
            self.executor.update_parameters(**mapped_reqs)

    async def _wrap_jobindex(self, job, idx):
        return (idx, await job.awaitable().result())

    def _instantiate_and_register_submitit_job(
        self, submitit_job: submitit.Job, module: "StopesModule"
    ):
        """
        Takes parameters: submitit_job object and module
        Instantiates submitit_job as an StopesJob (specifically as a SubmititJob)
        Registers it to registry
        """
        # Instantiating Stopes Job object:
        stopes_job = SubmititJob(
            submitit_job,
            self.jobs_registry.get_total_job_count(),
            module,
        )
        # Registering job into registry
        self.jobs_registry.register_job(stopes_job)
        logger.debug(stopes_job.get_job_info_log())

    def uncached_schedule_iterator(
        self,
        module: "StopesModule",
        array_with_indexes: tp.Iterable[tp.Tuple[int, tp.Any]],
    ) -> tp.List[tp.Coroutine[tp.Tuple[int, tp.Any], None, None]]:
        jobs = []
        self._setup_parameters(module)
        with self.executor.batch():
            for idx, v in array_with_indexes:
                jobs.append(
                    (
                        idx,  # we need to keep the true index for mixing with cached results later
                        self.executor.submit(
                            module,
                            iteration_value=v,
                            iteration_index=idx,
                            cache=self.cache,
                        ).cancel_at_deletion(),
                    )
                )
        logger.info(
            f"submitted job array for {module.name()}: {[j.job_id for _, j in jobs]}"
        )
        # Once jobs have been submitted, must add them to registry
        for _, submitit_job in jobs:
            self._instantiate_and_register_submitit_job(submitit_job, module)

        try:
            return [self._wrap_jobindex(j, idx) for idx, j in jobs]
        except Exception as e:
            # one job at least failed, kill everything else
            logger.warning(
                f"Exception while gathering job array for {module.name()}, cancelling all job array."
            )
            for _, j in jobs:
                j.cancel(
                    check=False  # it's ok if cancelling fails, in particular we probably can't cancel the job that failed
                )
            raise e

    async def uncached_schedule_single(self, module: "StopesModule"):
        self._setup_parameters(module)
        job = self.executor.submit(module, cache=self.cache).cancel_at_deletion()
        logger.info(f"submitted single job for {module.name()}: {job.job_id}")

        self._instantiate_and_register_submitit_job(job, module)

        return await job.awaitable().result()
