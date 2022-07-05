# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

if tp.TYPE_CHECKING:
    from stopes.core.stopes_module import StopesModule

logger = logging.getLogger("stopes.jobs")

################################################################################
#  Stopes Job - Required Helper Enums and Class Definitions
################################################################################


class RegistryStatuses(Enum):
    """
    The job registry displays 6 essential high-level status of a job, shown below.
    """

    COMPLETED = "Completed"
    FAILED = "Failed"
    PENDING = "Pending"
    RUNNING = "Running"
    UNKNOWN = "Unknown"
    KILLED_BY_LAUNCHER = "Killed by launcher"


class JobType(Enum):
    """
    Two types of jobs exist: single jobs and array jobs
    """

    SINGLE = "Single"
    ARRAY = "Array"


@dataclass
class ArrayJobInfo:
    """
    ArrayJobInfo stores additional info for jobs of array type (not single job). Stored info is:
        parent_job_id: (the id of the combined array of jobs; In slurm this is the job ID before the underscore _ (e.g for 1234567_00, it would be 1234567))
        array_index: (the index of this job within larger job array)
    """

    def __init__(self, parent_job_id: str, array_index: int):
        self.parent_job_id = parent_job_id
        self.array_index = array_index
        assert array_index >= 0, f"array_index: {array_index} must be >= 0"
        # could also be helpful to add total jobs in array in future


class ManuallyKilledByLauncherStatuses(Enum):
    """If a job is killed by the launcher, this enum helps to track the status of this kill.
    Sometimes the kill may succeed, fail, be in progress, etc. Below is a description of each status:

    NEVER_ATTEMPTED_TO_KILL_JOB: Launcher never attempted to kill this job
    JOB_FINISHED_BEFORE_ATTEMPTED_KILL: Launcher attempted to kill this job but the job had finished already
    JOB_FINISHED_DURING_ATTEMPTED_KILL: This has been added to deal with a bit of a flaky situation with the launcher, where: the Launcher attempted to kill this job (while the job was still unfinished), but in the time of trying to kill the job, the job had finished on its own
    ATTEMPT_IN_PROGRESS: Launcher currently attempting to kill this unfinished job
    SUCCESSFULLY_KILLED: Launcher successfully killed this unfinished job
    """

    NEVER_ATTEMPTED_TO_KILL_JOB = "Never attempted to kill job"
    JOB_FINISHED_BEFORE_ATTEMPTED_KILL = "Job finished before attempted kill"
    JOB_FINISHED_DURING_ATTEMPTED_KILL = "Job finished on its own during attempted kill by launcher"  # added to deal with flakiness described above
    ATTEMPT_IN_PROGRESS = "Attempt in progress"
    SUCCESSFULLY_KILLED = "Successfully killed by launcher"


################################################################################
#  Exceptions
################################################################################
class NonIdentifiableJobType(Exception):
    """
    Exception raised when trying to identify job type but unable to.
    Job types are specified in JobType Enum in stopes_job.py (array or singular)
    """

    def __init__(
        self,
        job_id: str,
        message: str = "The type of this job somehow can't be identified.",
    ):
        self.job_id = job_id
        self.message = f"Job ID is: {job_id}. " + message
        super().__init__(self.message)


################################################################################
#  Abstract StopesJob - A concrete subclass can be created for each new launcher type
#  Right now there's only one launcher: Submitit; the Concrete implementation for StopesJob is SubmititJob
################################################################################
class StopesJob(ABC):
    """
    Attributes:
    job_id (str)
    index_in_registry (int, 0-indexed)
    module (StopesModule)
    triggered_dying_strategy: (bool)
      - starts as False, set to True if and only if the job fails and triggers dying strategy.
      - This is helpful because it identifies job(s) that triggered the dying strategy.
      - Also, when killing all jobs in the registry, a job where this is marked as True wil not be killed

    killed_by_launcher_status:
      - status of the kill by launcher (ie kill in progress, kill succeeded, etc)
      - can take any value from within ManuallyKilledByLauncherStatuses enum

    status_of_job_before_launcher_kill:
      - status of job right before launcher attempted to kill it (ie Running, Pending, etc)
      - if killed_by_launcher_status is NEVER_ATTEMPTED_TO_KILL_JOB, this should be None.
    """

    def __init__(
        self,
        job_id: str,
        index_in_registry: int,
        module: "StopesModule",
    ):
        self._job_id = job_id
        self._index_in_registry = index_in_registry
        self._module = module

        self.triggered_dying_strategy = False
        self.killed_by_launcher_status: ManuallyKilledByLauncherStatuses = (
            ManuallyKilledByLauncherStatuses.NEVER_ATTEMPTED_TO_KILL_JOB
        )
        self.status_of_job_before_launcher_kill: RegistryStatuses = None

    # Property getters:
    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def index_in_registry(self) -> int:
        return self._index_in_registry

    @property
    def module(self) -> "StopesModule":
        return self._module

    # Abstract property getters:
    @property
    @abstractmethod
    def std_out_file(self) -> Path:
        pass

    @property
    @abstractmethod
    def std_err_file(self) -> Path:
        pass

    @property
    @abstractmethod
    def job_type(self) -> JobType:
        """
        Returns either JobType.SINGLE or JobType.ARRAY
        Should raise NonIdentifiableJobType(self._job_id) if job type is neither of these options
        """
        pass

    @property
    @abstractmethod
    def array_job_info(self) -> ArrayJobInfo:
        """
        If job type is single, returns None
        Else (job type is array), should return ArrayJobInfo object
        """
        pass

    # Methods
    def get_job_info_log(self) -> str:
        """
        This function returns a string that can be used to log job info in one line in an organized fashion.
        It logs fields like Job id, status, type, and array job info (if type is array).
        """
        list_of_log_info = [
            f"Job Info - Registry Index: {self._index_in_registry} [id: {self._job_id}",
            f"Status: {self.get_status()}",
            f"triggered_dying_strategy: {self.triggered_dying_strategy}",
            f"status of kill by launcher : {self.killed_by_launcher_status.name}",
            f"status of job before killed by launcher: {self.status_of_job_before_launcher_kill}",
            f"module: {self._module.name()}",
            f"type: {self._job_type.value}",
        ]

        if self._job_type == JobType.ARRAY:
            list_of_log_info.append(
                f"Extra Array Info: (Parent Job ID: {self._array_job_info.parent_job_id}, Array Index: {self._array_job_info.array_index})"
            )
        else:
            list_of_log_info.append(f"Extra Array Info: None")

        job_info_log = " | ".join(list_of_log_info)
        return job_info_log

    @abstractmethod
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
        pass

    @abstractmethod
    def get_status(self) -> RegistryStatuses:
        """
        Gets job status
        """
        pass
