# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import dataclasses
import hashlib
import logging
import typing as tp
from pathlib import Path

import submitit
import tqdm
from omegaconf import OmegaConf
from submitit import AutoExecutor
from tqdm.contrib.logging import logging_redirect_tqdm

from stopes.core import utils
from stopes.core.cache import Cache, MissingCache, NoCache
from stopes.core.jobs_registry.registry import JobsRegistry
from stopes.core.jobs_registry.submitit_slurm_job import SubmititJob

if tp.TYPE_CHECKING:
    from stopes.core import StopesModule

logger = logging.getLogger("stopes.launcher")


@dataclasses.dataclass
class Task:
    module: "StopesModule"
    iteration_index: int
    iteration_value: tp.Any
    launcher: "Launcher"

    # private stuff
    _done: bool = dataclasses.field(init=False)
    _job: tp.Optional[submitit.Job] = dataclasses.field(init=False)
    _result: tp.Any = dataclasses.field(init=False, repr=False)
    # _job keeps the current inflight job, _tries keeps the past tries
    _tries: tp.List[submitit.Job] = dataclasses.field(default_factory=list, init=False)

    def done_from_cache(
        self,
        result: tp.Any,
    ) -> "Task":
        self._done = True
        self._result = result
        self._job = None
        return self

    def waiting_on_job(
        self,
        job: submitit.Job,
    ) -> "Task":
        self._done = False
        self._job = job
        self._result = None

        return self

    async def wait(self) -> "Task":
        if self.done:
            return self
        assert self._job is not None, "No _job on pending task."
        for attempt in range(
            # 1 initial run + max_retries extra tries
            (1 + self.launcher.max_retries)
            + 1  # + 1 for range
        ):  # this finishes either with a raise or a return
            try:
                # for a multi-task job, get all results
                results = await self._job.awaitable().results()
                # but because they are all identical, just return the first
                self._result = results[0]
                self._done = True
                return self
            except Exception as ex:
                self._handle_exception(ex, attempt)

        assert False, f"Too many retries {attempt} and that wasn't handled"

    def _handle_exception(self, ex, attempt):
        """
        deal with retries. If we see an exception while waiting on the result from submitit, then
        something bad has happenened, we need to decide if we retry or not.
        """
        if (  # one first try + n retries = too many tries
            attempt >= (1 + self.launcher.max_retries)
        ) or not self.module.should_retry(
            ex=ex,
            attempt=attempt,
            iteration_index=self.iteration_index,
            iteration_value=self.iteration_value,
        ):  # we should fail hard here
            raise ex

        assert self._job is not None, "No _job on pending task."
        self._tries.append(self._job)
        self._job = self.launcher.submit_job(
            self.module,
            iteration_index=self.iteration_index,
            iteration_value=self.iteration_value,
        )
        self.module.retry_counts[self.iteration_index] = len(self._tries) - 1

    async def wait_and_validate(self):
        if self.done:
            return self._result
        await self.wait()
        self.validate()

    def validate(self) -> None:
        if not self.module.validate(
            self._result,
            iteration_value=self.iteration_value,
            iteration_index=self.iteration_index,
        ):
            raise ValueError(
                f"invalid result for {self.module.name()}:{self.iteration_index}"
            )

    @property
    def final_result(self) -> tp.Any:
        """
        you need to call `wait` on the task before calling final_result, unless you are sure it's a cached result, otherwise it will not work
        """
        assert (
            self._done
        ), "you need to make sure that Task is done before inspecting results"
        return self._result

    @property
    def done(self) -> bool:
        return self._done


class Launcher:
    def __init__(
        self,
        cache: tp.Optional[Cache] = None,
        config_dump_dir: tp.Optional[Path] = None,
        log_folder: tp.Union[Path, str] = Path("executor_logs"),
        cluster: str = "local",
        partition: tp.Optional[str] = None,
        supports_mem_spec: bool = True,  # some slurm clusters do not support mem_gb, if you set this to False, the Requirements.mem_gb coming from the module will be ignored
        disable_tqdm: bool = False,  # if you don't want tqdm progress bars
        max_retries: int = 0,
        max_jobarray_jobs: int = 1000,
    ):
        """
        If this was started by another launcher, it will be passed down, other launcher is None.
        """
        self.cache = NoCache() if cache is None else cache
        self.config_dump_dir = (
            Path(config_dump_dir)
            if config_dump_dir is not None
            else Path.cwd() / "config_logs"
        )
        self.config_dump_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_registry: JobsRegistry = JobsRegistry()

        self.log_folder = Path(log_folder)
        self.cluster = cluster
        self.partition = partition
        self.supports_mem_spec = supports_mem_spec
        self.disable_tqdm = disable_tqdm
        self.progress_bar: tqdm.tqdm = None
        self.cluster = cluster
        self.max_retries = max_retries
        self.max_jobarray_jobs = max_jobarray_jobs

    def dump_config(self, module: "StopesModule") -> Path:
        config_folder = Path(self.config_dump_dir) / module.name()
        config_folder.mkdir(exist_ok=True, parents=True)
        config_file = config_folder / f"{module.sha_key()}.yaml"
        OmegaConf.save(config=module.config, f=config_file)
        return config_file

    def progress_start_jobs(self, n: int) -> None:
        if self.disable_tqdm:
            return
        if self.progress_bar is None:
            self.progress_bar = tqdm.tqdm(total=n)
            return

        self.progress_bar.total += n
        self.progress_bar.refresh()

    def progress_job_end(self) -> None:
        if self.disable_tqdm:
            return
        self.progress_bar.update(1)

    async def schedule(self, module: "StopesModule"):
        with logging_redirect_tqdm():
            self.dump_config(module)
            value_array = module.array()
            if value_array is not None:
                self.progress_start_jobs(len(value_array))
                result = await self._schedule_array(module, value_array)
                # progress has already been incremented inside _schedule_array
                # once every job finishes
            else:
                self.progress_start_jobs(1)
                result = await self._schedule_single(module)
                self.progress_job_end()
            self.jobs_registry.log_progress()
            return result

    #######################################
    # setup the executor
    #######################################

    def _get_executor(self, module: "StopesModule") -> submitit.Executor:
        # we create a separate folder, under log_folder for each module so that logs are more organized
        name = module.name()
        module_log_folder = self.log_folder / name
        module_log_folder.mkdir(parents=True, exist_ok=True)
        executor = AutoExecutor(folder=module_log_folder, cluster=self.cluster)

        # setup parameters
        reqs = module.requirements()
        executor.update_parameters(
            name=module.name(),
            slurm_partition=self.partition,
        )
        if self.partition and self.cluster == "slurm":
            executor.update_parameters(
                slurm_partition=self.partition,
            )

        comment = module.comment()
        if comment is not None:
            executor.update_parameters(
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
            executor.update_parameters(**mapped_reqs)
            if hasattr(reqs, "mem_gb") and self.supports_mem_spec:
                executor.update_parameters(mem_gb=reqs.mem_gb)
            if hasattr(reqs, "contraints"):
                executor.update_parameters(slurm_constraint=reqs.constraint)

        return executor

    ########################################
    # schedule a single (non-array) job
    ########################################

    async def _schedule_single(self, module: "StopesModule") -> tp.Any:
        try:
            cached_result = self.cache.get_cache(module)
            logger.info(f"{module.name()} done from cache")
            return cached_result
        except MissingCache:
            pass

        job = self.submit_job(module)
        logger.info(f"submitted single job for {module.name()}: {job.job_id}")

        self._track_job(job, module)
        task = Task(module, 0, None, launcher=self).waiting_on_job(job)
        await task.wait_and_validate()

        logger.info(f"{module.name()} done after full execution")
        return task.final_result

    def submit_job(
        self,
        module: "StopesModule",
        iteration_index: int = 0,
        iteration_value: tp.Any = None,
    ) -> submitit.Job:
        executor = self._get_executor(module)
        job = executor.submit(
            module,
            iteration_index=iteration_index,
            iteration_value=iteration_value,
            cache=self.cache,
        ).cancel_at_deletion()
        self._track_job(job, module)
        return job

    ########################################
    # schedule multiple jobs at once
    ########################################

    async def _schedule_array(
        self, module: "StopesModule", value_array: tp.List[tp.Any]
    ) -> tp.List[tp.Any]:
        """
        check if any item is in the cache, for the rest compute it
        """
        module.retry_counts = [0] * len(value_array)
        executor = self._get_executor(module)
        tasks = []
        tasks_to_submit = []

        for idx, val in enumerate(value_array):
            task = Task(module, idx, val, launcher=self)
            # first, look up if this iteration has already been cached
            try:
                cached_result = self.cache.get_cache(
                    module,
                    iteration_index=idx,
                    iteration_value=val,
                )

                task = task.done_from_cache(cached_result)
                self.progress_job_end()
            except MissingCache:
                tasks_to_submit.append(task)
            tasks.append(task)

        for task_batch in utils.batch(tasks_to_submit, self.max_jobarray_jobs):
            with executor.batch():
                for task in task_batch:
                    job = executor.submit(
                        task.module,
                        iteration_index=task.iteration_index,
                        iteration_value=task.iteration_value,
                        cache=self.cache,
                    ).cancel_at_deletion()
                    task = task.waiting_on_job(job)

        not_cached = len(tasks_to_submit)
        already_cached = len(tasks) - not_cached
        logger.info(
            f"for {module.name()} found {already_cached} already cached array results,"
            f"{not_cached} left to compute out of {len(value_array)}"
        )
        if not_cached == 0:
            return [task.final_result for task in tasks]

        logger.info(
            f"submitted job array for {module.name()}: {[task._job.job_id for task in tasks if task._job is not None]}"
        )
        # keep tasks in the register, I'm not sure why we aren't tracking everything
        for task in tasks:
            if task._job is not None:
                self._track_job(task._job, module)

        # now await for the results of each iteration.
        # we want to show progress as soon as a result returns. Jobs might not return
        # in order, so we use asyncio.as_completed to catch them as they arrive
        # we do this over an iterator of awaitables.
        for _res in asyncio.as_completed(
            [task.wait_and_validate() for task in tasks if not task.done]
        ):
            try:
                await _res
            # TODO: handle exception, eg we could decide to cancel the full job array
            finally:
                self.progress_job_end()

        # return the results that were already cached with the ones that just got computed
        return [task.final_result for task in tasks]

    def _track_job(self, submitit_job: submitit.Job, module: "StopesModule"):
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
        logger.debug(stopes_job.get_job_info_log())
