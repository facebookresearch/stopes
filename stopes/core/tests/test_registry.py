# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import unittest
from pathlib import Path

import omegaconf

from stopes.core.jobs_registry.registry import JobsRegistry
from stopes.core.jobs_registry.submitit_slurm_job import RegistryStatuses, SubmititJob
from stopes.core.launcher import Launcher, SubmititLauncher
from stopes.core.stopes_module import StopesModule

from . import hello_world


async def schedule_module_and_check_registry(
    launcher: Launcher,
    module: StopesModule,
    config: omegaconf.DictConfig,
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

    instantiated_module = module(config)
    job = launcher.schedule(instantiated_module)
    await job

    for job_id in jobs_registry.registry:
        current_stopes_job = jobs_registry.get_job(job_id)
        assert current_stopes_job.get_status() == RegistryStatuses.COMPLETED.value


async def test_successful_single_job(tmp_path: Path):
    """
    Tests the registry's functionality on a single module.
    """
    launcher = SubmititLauncher(
        config_dump_dir=tmp_path / "conf", log_folder=tmp_path / "logs"
    )

    await schedule_module_and_check_registry(
        launcher,
        hello_world.HelloWorldModule,
        config=omegaconf.OmegaConf.create(
            {"greet": "hello", "person": "world", "duration": 0.5}
        ),
        total_scheduled_jobs=1,
    )


async def test_successful_array_jobs(tmp_path: Path):
    """
    Tests the registry's functionality on an array module
    """
    launcher = SubmititLauncher(
        config_dump_dir=tmp_path / "conf", log_folder=tmp_path / "logs"
    )
    team = ["Anna", "Bob", "Eve"]

    await schedule_module_and_check_registry(
        launcher,
        hello_world.HelloWorldArrayModule,
        config=omegaconf.OmegaConf.create(
            {"greet": "hello", "persons": team, "duration": 0.5}
        ),
        total_scheduled_jobs=len(team),
    )
