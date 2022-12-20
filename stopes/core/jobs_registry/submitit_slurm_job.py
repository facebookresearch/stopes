# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import time
from pathlib import Path

import submitit
from submitit.slurm.slurm import read_job_id

from stopes.core.jobs_registry.stopes_job import (
    ArrayJobInfo,
    JobType,
    ManuallyKilledByLauncherStatuses,
    NonIdentifiableJobType,
    RegistryStatuses,
    StopesJob,
)
from stopes.core.stopes_module import StopesModule

logger = logging.getLogger("stopes.jobs")

################################################################################
#  Submitit Launcher - SLURM Job Statuses Dictionary
################################################################################
# Submitit/Slurm jobs statuses can take a total of ~28 different values
# These are documented here: https://slurm.schedmd.com/squeue.html under the "Job State Codes" Heading
# The job registry however displays only 6 high-level essential statuses. This can be seen in the RegistryStatuses Enum in stopes/core/jobs_registry/stopes_job.py
# Below is a dictionary to map each of the submitit job statuses 24 SLURM job + a few local Job statuses to the 6 accepted registry statuses

submitit_state_to_registry_state_dict = {
    # ############## RegistryStatuses.COMPLETED below ###########################
    "COMPLETED": RegistryStatuses.COMPLETED.value,
    # ############## RegistryStatuses.FAILED below ##############################
    "FAILED": RegistryStatuses.FAILED.value,
    "STOPPED": RegistryStatuses.FAILED.value,
    "SUSPENDED": RegistryStatuses.FAILED.value,
    "TIMEOUT": RegistryStatuses.FAILED.value,
    "NODE_FAIL": RegistryStatuses.FAILED.value,
    "OUT_OF_MEMORY": RegistryStatuses.FAILED.value,
    "DEADLINE": RegistryStatuses.FAILED.value,
    "BOOT_FAIL": RegistryStatuses.FAILED.value,
    "RESV_DEL_HOLD": RegistryStatuses.FAILED.value,
    "REVOKED": RegistryStatuses.FAILED.value,
    "SIGNALING": RegistryStatuses.FAILED.value,  # I'm quite sure signalling means being cancelled
    # ############## RegistryStatuses.PENDING below ##############################
    "QUEUED": RegistryStatuses.PENDING.value,
    "PENDING": RegistryStatuses.PENDING.value,
    "REQUEUE_FED": RegistryStatuses.PENDING.value,
    "REQUEUE_HOLD": RegistryStatuses.PENDING.value,
    "REQUEUED": RegistryStatuses.PENDING.value,
    "RESIZING": RegistryStatuses.PENDING.value,
    "READY": RegistryStatuses.PENDING.value,  # from local submitit job statuses
    # ############## RegistryStatuses.RUNNING below ##############################
    "RUNNING": RegistryStatuses.RUNNING.value,
    "COMPLETING": RegistryStatuses.RUNNING.value,
    "CONFIGURING": RegistryStatuses.RUNNING.value,
    # ############## RegistryStatuses.UNKNOWN below ##############################
    "UNKNOWN": RegistryStatuses.UNKNOWN.value,
    "FINISHED": RegistryStatuses.UNKNOWN.value,
    # "FINISHED" is set to UNKNOWN because both successful and failed jobs are BOTH labelled as 'finished'
    # Must perform an additional check to see if the job finished due to success or failure. This check is done in convert_slurm_status_into_registry_job_status function
    "INTERRUPTED": RegistryStatuses.UNKNOWN.value,  # from local submitit job statuses
    # "INTERRUPTED" Status is a bit tricky because if the job was manually interrupted as part of dying strategies, then we set the status to be KILLED_BY_LAUNCHER
    # If it was interrupted otherwise, we leave it as FAILED
    "CANCELLED": RegistryStatuses.FAILED.value,
    # Same logic applies as for INTERRUPTED status. If the job was cancelled by the launcher, then we set the status to be KILLED_BY_LAUNCHER
    # Else, we leave it as FAILED
    # (INTERRUPTED [local cluster] and CANCELLED [slurm cluster] seem to be equivalents of each other)
    "STAGE_OUT": RegistryStatuses.UNKNOWN.value,
    # STAGE_OUT status: Occurs once the job has completed or been cancelled, but Slurm has not released resources for the job yet. Source: https://slurm.schedmd.com/burst_buffer.html
    # Hence, setting it as unknown as this state doesn't specifify completion or cancellation.
    "SPECIAL_EXIT": RegistryStatuses.UNKNOWN.value,
    # SPECIAL_EXIT status: This occurs when a job exits with a specific pre-defined reason (e.g a specific error case).
    # This is useful when users want to automatically requeue and flag a job that exits with a specific error case. Source: https://slurm.schedmd.com/faq.html
    # Hence, setting as unknown as the status may change depending on automatic re-queue; without this requeue, job should be FAILED.
    "PREEMPTED": RegistryStatuses.UNKNOWN.value,
    # PREEMPTED Status: Different jobs react to preemption differently - some may automatically requeue and other's may not
    # (E.g if checkpoint method is implemented or not). Hence, setting as Unknown for now; without automatic re-queue, job should be FAILED
}


################################################################################
#  Submitit Launcher Job Class
################################################################################
class SubmititJob(StopesJob):
    """
    Job class for the submitit launcher.
    """

    def __init__(
        self,
        submitit_job_object: submitit.Job,
        index_in_registry: int,
        module: StopesModule,
    ):

        assert isinstance(
            submitit_job_object, submitit.Job
        ), f"submitit_job_object must have type submitit.Job but was provided with type {type(submitit_job_object)}"
        self._launcher_specific_job_object: submitit.Job = submitit_job_object
        job_id = self._launcher_specific_job_object.job_id

        # Setting abstract properties:
        self._std_out_file = self._launcher_specific_job_object.paths.stdout
        self._std_err_file = self._launcher_specific_job_object.paths.stderr

        # Now setting job_type (either single or array)
        # The read_job_id function reads a job id and returns a tuple with format:
        # [(parent_job_id, array_index] where array_index is only for array jobs; Hence, tuple length is 1 if job is single, and 2 if its an array
        read_job_id_output = read_job_id(job_id)[0]
        read_job_id_output_length = len(read_job_id_output)
        if read_job_id_output_length == 1:  # single job
            self._job_type = JobType.SINGLE
            self._array_job_info = None

        elif read_job_id_output_length == 2:  # array job
            self._job_type = JobType.ARRAY
            # Since it's an array job, set ArrayJobInfo to store parent_job_id and job_index_in_array
            parent_job_id = read_job_id_output[0]
            job_index_in_array = int(read_job_id_output[1])
            self._array_job_info = ArrayJobInfo(parent_job_id, job_index_in_array)

        # Note: there's a small trick when it comes to job types with slurm/submitit
        # On the slurm cluster, scheduling it as an array / singular works as intended.
        # But on the local cluster, EVEN if you schedule a set of jobs as an array, it will still schedule them individually.
        else:
            raise NonIdentifiableJobType(job_id)

        # Calling parent class (StopesJob) constructor
        super().__init__(
            job_id=job_id,
            index_in_registry=index_in_registry,
            module=module,
        )

    @property
    def launcher_specific_job_object(self) -> submitit.Job:
        return self._launcher_specific_job_object

    # Implementations for Absract Properties:
    @property
    def std_out_file(self) -> Path:
        return self._std_out_file

    @property
    def std_err_file(self) -> Path:
        return self._std_err_file

    @property
    def job_type(self) -> JobType:
        """
        Returns either JobType.SINGLE or JobType.ARRAY.
        Should raise NonIdentifiableJobType(self._job_id) if job type is neither of these options
        """
        job_type = self._job_type
        if job_type != JobType.SINGLE and job_type != JobType.ARRAY:
            raise NonIdentifiableJobType(self._job_id)
        return job_type

    @property
    def array_job_info(self) -> ArrayJobInfo:
        if self._job_type == JobType.SINGLE:
            logger.info(
                f"array_job_info method called on Job of type {JobType.SINGLE.value}. This will necessarily return None"
            )
        return self._array_job_info

    # Implementation for Abstract Methods:
    async def kill_job(self):
        """
        Kills a job and waits till the job is killed
        Should only be called on: unfinished jobs and jobs with triggered_dying_strategy = False (can enforce this rule in future PRs)

        This method should:
        - set self.status_of_job_before_launcher_kill
        - kill the job
        - set self.killed_by_launcher_status
        - wait until job is finished (Completed, Failed, or Killed by Launcher)
        """

        self.status_of_job_before_launcher_kill = self.get_status()
        self.killed_by_launcher_status = (
            ManuallyKilledByLauncherStatuses.ATTEMPT_IN_PROGRESS
        )
        self._launcher_specific_job_object.cancel()

        self._launcher_specific_job_object.wait()  # waits for kill to finish
        # job.kill + job.wait + 60 second wait is a bit flaky: sometimes despite job.kill and the extra waiting the job completes successfully on its own.

        is_job_finished = self._has_job_finished_after_attempted_kill()
        if not is_job_finished:
            # Job must be pending, unknown or still running, meaning it wasn't killed yet
            # It's highly possible that the job was indeed killed, but the the status from the cluster isn't up to date
            # This is because the status that fetched from cluster is only updated ~ once per minute (to avoid overloading cluster)
            # Hence, put job to sleep for ~65 seconds to let the status update
            await asyncio.sleep(65)

            is_job_finished = self._has_job_finished_after_attempted_kill()
            if not is_job_finished:
                logger.warning(
                    "Somehow Job: {self.job_id} is unfinished despite attempt by launcher to kill and a wait of more than 60 seconds"
                )

    def get_status(self) -> RegistryStatuses:
        return self._convert_slurm_status_into_registry_job_status(
            self._launcher_specific_job_object.state, self.job_id
        )

    def _convert_slurm_status_into_registry_job_status(
        self, slurm_status: str, job_id: str
    ) -> RegistryStatuses:
        """
        This function maps slurm's/submitit's ~28 different values to return 6 essential statuses understood by the registry,
        And logs a warning for unrecognized statuses. For most of the statuses this is as simple as a dictionary lookup.
        For a couple of them (FINISHED, CANCELLED, INTERRUPTED) we have to deal with them in a bit of a special way.
        """
        # First, handle "FINISHED" status:
        if slurm_status in ("FINISHED", "DONE"):
            # Now determine if (local) job finished due to success or failure
            # If the job doesn't raise an exception, it succeeded. Else, it failed (failed on its own; wasn't killed by launcher)
            job_failed = self.launcher_specific_job_object.exception()

            # If launcher never attempted kill, keep killed_by_launcher_status as it is. Else, set it to JOB_FINISHED_DURING_ATTEMPTED_KILL
            self.killed_by_launcher_status = (
                self.killed_by_launcher_status
                if (
                    self.killed_by_launcher_status
                    == ManuallyKilledByLauncherStatuses.NEVER_ATTEMPTED_TO_KILL_JOB
                    or self.killed_by_launcher_status
                    == ManuallyKilledByLauncherStatuses.JOB_FINISHED_BEFORE_ATTEMPTED_KILL
                )
                else ManuallyKilledByLauncherStatuses.JOB_FINISHED_DURING_ATTEMPTED_KILL
            )
            return (
                RegistryStatuses.FAILED.value
                if job_failed
                else RegistryStatuses.COMPLETED.value
            )

        # Second, handle statuses: INTERRUPTED (local)/CANCELLED (slurm)
        if (
            slurm_status == "INTERRUPTED"
            or slurm_status == "CANCELLED"
            or slurm_status.find("CANCELLED") != -1
            # The last one is included because sometimes, if killed by the launcher, the status may come out to be "CANCELLED by <number>" (e.g. for number: 1713700156)
        ):
            # Check if the launcher attempted to kill this job
            has_launcher_attempted_kill = (
                self.killed_by_launcher_status
                == ManuallyKilledByLauncherStatuses.ATTEMPT_IN_PROGRESS
            ) or (
                self.killed_by_launcher_status
                == ManuallyKilledByLauncherStatuses.SUCCESSFULLY_KILLED
            )

            # If has_launcher_attempted_kill, then the interruption is due to the launcher. Hence return status KILLED_BY_LAUNCHER. Else, return FAILED
            if has_launcher_attempted_kill:
                self.killed_by_launcher_status = (
                    ManuallyKilledByLauncherStatuses.SUCCESSFULLY_KILLED
                )
                return RegistryStatuses.KILLED_BY_LAUNCHER.value
            else:
                return RegistryStatuses.FAILED.value

        # Now, we handle the general cases - simply do dictionary lookup:
        try:
            job_status = submitit_state_to_registry_state_dict[slurm_status]
            if (
                job_status == RegistryStatuses.COMPLETED.value
                or job_status == RegistryStatuses.FAILED.value
            ):
                if (
                    self.killed_by_launcher_status
                    == ManuallyKilledByLauncherStatuses.ATTEMPT_IN_PROGRESS
                ):
                    self.killed_by_launcher_status = (
                        ManuallyKilledByLauncherStatuses.JOB_FINISHED_DURING_ATTEMPTED_KILL
                    )

            return job_status

        except KeyError:  # Entering this except block means slurm_status doesn't exist in submitit_state_to_registry_state_dict
            logger.warning(
                f"Job with id: {job_id} has unrecognized slurm status: {slurm_status}. Please inspect and if suitable, add this status to the slurm_state_to_registry_state_map converter."
            )
            return RegistryStatuses.UNKNOWN.value

    # Private Helper Methods
    def _has_job_finished_after_attempted_kill(self):
        """
        This private helper:
            Returns True if job has finished (either Completed, Failed, or Killed by Launcher)
            Returns False if job is unfinished (either pending, running, or unknown)

        Also, this helper:
            Is to be called only after a job has been attempted to be killed AND waited (job.kill() AND job.wait())
            Will set the job's killed_by_launcher_status
        """
        job_status_after_attempted_kill = self.get_status()
        if job_status_after_attempted_kill == RegistryStatuses.KILLED_BY_LAUNCHER.value:
            # Job has been successfully killed by launcher
            self.killed_by_launcher_status = (
                ManuallyKilledByLauncherStatuses.SUCCESSFULLY_KILLED
            )
            return True
        elif job_status_after_attempted_kill in [
            RegistryStatuses.COMPLETED.value,
            RegistryStatuses.FAILED.value,
        ]:
            # Even though the launcher attempted to kill an unfinished job, the job finished on its own before it was killed by the launcher
            self.killed_by_launcher_status = (
                ManuallyKilledByLauncherStatuses.JOB_FINISHED_DURING_ATTEMPTED_KILL
            )
            return True
        else:
            # Job must be pending, unknown or still running, meaning it wasn't killed yet
            return False
