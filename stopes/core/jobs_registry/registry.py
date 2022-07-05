# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import collections
import logging
import typing as tp

from stopes.core.jobs_registry.stopes_job import ManuallyKilledByLauncherStatuses
from stopes.core.jobs_registry.submitit_slurm_job import RegistryStatuses, StopesJob

logger = logging.getLogger("stopes.jobs")

################################################################################
#  Registry Exceptions
################################################################################
class JobNotInRegistry(Exception):
    """
    Exception raised when querying for a job id in the registry that doesn't exist
    """

    def __init__(
        self, job_id: str, message: str = "This Job doesn't exist in registry"
    ):
        self.job_id = job_id
        self.message = f"Job ID is: {job_id}. " + message
        super().__init__(self.message)


################################################################################
#  Registry definition
################################################################################
class JobsRegistry:
    """
    The JobsRegistry is a field in the launcher class that stores all past and present scheduled jobs.
    Implemented as an dictionary (key = job_id, and value = StopesJob type). The dictionary is ordered (stores jobs chronologically).
    """

    def __init__(self):
        self.registry: tp.OrderedDict[str, StopesJob] = tp.OrderedDict()

    # General Methods for Registry
    def get_total_job_count(self) -> int:
        return len(self.registry)

    def get_job(self, job_id: str) -> StopesJob:
        try:
            job = self.registry[job_id]
            return job
        except KeyError:  # job_id doesn't exist in registry
            raise JobNotInRegistry(job_id)

    def register_job(self, stopes_job: StopesJob):
        """
        Adds job to the registry. If job already exists, logs a warning.
        """
        job_id = stopes_job.job_id
        if job_id in self.registry:
            logger.warning(
                f"Tried to add job with id: {job_id} into registry, but it already exists previously"
            )
        else:
            self.registry[job_id] = stopes_job

    ###########################################################
    # Methods for killing job(s)
    ###########################################################

    async def kill_job(self, job_id: str):
        """
        Kills a job.
        We deal with this based on status. If a job status is not Completed nor Failed, this method kills the job
        """
        job: StopesJob = self.get_job(job_id)
        job_status = job.get_status()
        # If conditions are segregated due to difference in logs
        if job_status in [
            RegistryStatuses.COMPLETED.value,
            RegistryStatuses.FAILED.value,
        ]:
            job.killed_by_launcher_status = (
                ManuallyKilledByLauncherStatuses.JOB_FINISHED_BEFORE_ATTEMPTED_KILL
            )
            logger.warning(
                f"Tried to kill a job with id: {job_id}, but this job has already {job_status}"
            )
            return

        if job.triggered_dying_strategy:
            # Sometimes a job raises an exception but takes some time for the status to reflect as failed; So, the previous if condition isn't entered.
            # But the triggered_dying_strategy boolean flag is is up to date - it tells us immediately if this job failed/is failing
            job.killed_by_launcher_status = (
                ManuallyKilledByLauncherStatuses.JOB_FINISHED_BEFORE_ATTEMPTED_KILL
            )
            logger.warning(
                f"Tried to kill a job with id: {job_id}, but this has already failed on its own."
            )
            return

        if job_status == RegistryStatuses.KILLED_BY_LAUNCHER.value:
            logger.warning(
                f"Tried to kill a job with id: {job_id}, but this has already been killed by launcher."
            )
            return

        if job_status in [  # Job is unfinished
            RegistryStatuses.RUNNING.value,
            RegistryStatuses.PENDING.value,
            RegistryStatuses.UNKNOWN.value,
        ]:
            logger.info(f"Killing job: {job_id} with current status: {job_status}")
            await job.kill_job()
            return

        else:
            # This case should not be possible as we've tried every possible status
            # However, if new statuses are added in the future (unlikely), and their if cases are not added to this method function then this
            # else case serves as safety - it will raise an exception to ensure we have defined behaviour
            raise NotImplementedError(
                f"Job with id: {job_id} and status: {self.get_job(job_id).get_status()} can't be killed due to unidentifiable status. Please implement kill_job for the status: {job_status}"
            )

    async def kill_all_jobs_in_registry_except(
        self, set_of_job_ids_not_to_kill: tp.Set[str]
    ):
        """
        This function kills all jobs in the registry except for the id's specified in set_of_job_ids_not_to_kill
        If set_of_job_ids_not_to_kill is None, it kills all jobs.
        """
        logger.info("Starting to kill all jobs in registry")
        kill_jobs_list = []
        for job_id in self.registry:
            if job_id not in set_of_job_ids_not_to_kill:
                kill_jobs_list.append(self.kill_job(job_id))

        await asyncio.gather(*kill_jobs_list)

    def segment_registry_by_status(
        self,
    ) -> tp.Dict[RegistryStatuses, tp.Set[StopesJob]]:
        """
        This function returns all jobs segmented by the statuses in the RegistryStatuses enum
        (These are COMPLETED, FAILED, KILLED_BY_LAUNCHER etc.)

        Return type is a dictionary, where key is RegistryStatuses, and value is a set of StopesJob Objects
        Note this simply returns the jobs based on status at the time the function was called only; it does not update live based on changes in job statuses
        """
        registry_segmented_by_status: tp.Dict[RegistryStatuses, tp.Set[StopesJob]] = {
            RegistryStatuses.COMPLETED.value: set(),
            RegistryStatuses.FAILED.value: set(),
            RegistryStatuses.KILLED_BY_LAUNCHER.value: set(),
            RegistryStatuses.PENDING.value: set(),
            RegistryStatuses.RUNNING.value: set(),
            RegistryStatuses.UNKNOWN.value: set(),
        }

        for job_id in self.registry:
            stopes_job = self.get_job(job_id)
            job_status = stopes_job.get_status()
            registry_segmented_by_status[job_status].add(stopes_job)

        return registry_segmented_by_status

    def log_progress(self) -> None:
        if not self.registry:
            return
        progress: tp.Dict[RegistryStatuses, int] = collections.defaultdict(int)
        for job in self.registry.values():
            progress[job.get_status()] += 1
        progress["Total"] = len(self.registry)
        logger.info(f"Jobs progress: {dict.__repr__(progress)}")

    ###########################################################
    # Methods used for logging registry in different formats
    ###########################################################
    def get_log_for_registry_jobs_ordered_chronologically(self) -> str:
        """
        This method returns a string that can be used for logging.
        The string logs out each job in the registry chronologically
        Note: a similar method, get_log_for_registry_jobs_by_status, does the same but segments the jobs by status
        """
        # list_of_all_job_logs is an array of strings - each string contains the logs for one job in the registry
        list_of_all_job_logs = [
            "Entire Job Registry Ordered Chronologically is:",
            "#" * 200,
        ]

        for job_id in self.registry:
            current_job: StopesJob = self.get_job(job_id)
            list_of_all_job_logs.append(current_job.get_job_info_log())

        list_of_all_job_logs.append("#" * 200)

        entire_registry_log: str = "\n".join(list_of_all_job_logs)
        return entire_registry_log

    def get_log_for_registry_jobs_ordered_by_status(self) -> str:
        """
        This method returns a string that can be used for logging.
        The string logs out each job in the registry segmented by the statuses in RegistryStatuses
        This method is useful for dying strategies
        Note: a similar method, get_log_for_registry_jobs_ordered_chronologically, does the same but orders the job chronologically (by job index)
        """
        # list_of_all_job_logs is an array of strings - each string contains the logs for one job in the registry
        list_of_all_job_logs = [
            "#" * 200,
            "Entire Job Registry Ordered By Status is:",
        ]  # use separator of 200 #'s for visual clarity in logs

        # segment_registry_by_status() returns dict of [RegistryStatuses, set of StopesJobs]
        registry_segmented_by_status = self.segment_registry_by_status()

        for status in registry_segmented_by_status:
            list_of_all_job_logs.append(
                f"\nList of jobs with {status} status is below:"
            )
            for current_stopes_job in registry_segmented_by_status[status]:
                list_of_all_job_logs.append(
                    f"\t {current_stopes_job.get_job_info_log()}"
                )

        list_of_all_job_logs.append(
            f"\n{'#'*200}"
        )  # use separator of 200 #'s for visual clarity in logs

        entire_registry_log: str = "\n".join(list_of_all_job_logs)
        return entire_registry_log
