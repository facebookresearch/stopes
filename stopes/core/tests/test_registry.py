# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from stopes.core.jobs_registry.registry import JobsRegistry
from stopes.core.jobs_registry.submitit_slurm_job import RegistryStatuses
from stopes.core.launcher import Launcher
from stopes.core.stopes_module import StopesModule
from stopes.core.tests.hello_world import (
    HelloWorldArrayConfig,
    HelloWorldArrayModule,
    HelloWorldConfig,
    HelloWorldModule,
)


async def schedule_module_and_check_registry(
    launcher: Launcher,
    instantiated_module: StopesModule,
    total_scheduled_jobs: int,
):
    """
    Schedules the module and performs periodic checks/asserts in the registry before, during and after the job is running

    This helper function takes in a launcher, module, config and total_scheduled_jobs.
    It schedules a Stopes Module with the launcher, and asserts that exactly x jobs are scheduled,
    where x = total_scheduled_jobs parameter.
    After this, it asserts that the job statuses must be RUNNING.
    By the end, it asserts that the all job statuses must be COMPLETE
    """
    jobs_registry: JobsRegistry = launcher.jobs_registry
    # Job registry must be empty initially
    assert 0 == jobs_registry.get_total_job_count()

    job = launcher.schedule(instantiated_module)
    await job

    for job_id in jobs_registry.registry:
        current_stopes_job = jobs_registry.get_job(job_id)
        assert current_stopes_job.get_status() == RegistryStatuses.COMPLETED.value


async def test_successful_single_job(tmp_path: Path):
    """
    Tests the registry's functionality on a single module.
    """
    launcher = Launcher(
        config_dump_dir=tmp_path / "conf", log_folder=tmp_path / "logs", cluster="local"
    )

    await schedule_module_and_check_registry(
        launcher,
        HelloWorldModule(
            HelloWorldConfig(
                greet="hello",
                person="world",
                duration=0.5,
            )
        ),
        total_scheduled_jobs=1,
    )


async def test_successful_array_jobs(tmp_path: Path):
    """
    Tests the registry's functionality on an array module
    """
    launcher = Launcher(
        config_dump_dir=tmp_path / "conf", log_folder=tmp_path / "logs", cluster="local"
    )
    team = ["Anna", "Bob", "Eve"]

    await schedule_module_and_check_registry(
        launcher,
        HelloWorldArrayModule(
            HelloWorldArrayConfig(
                greet="hello",
                persons=team,
                duration=0.5,
            )
        ),
        total_scheduled_jobs=len(team),
    )
