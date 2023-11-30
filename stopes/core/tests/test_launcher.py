# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import asyncio
import contextlib
import datetime
import math
import time
import typing as tp
import uuid
from pathlib import Path
from unittest import mock

import pytest
from omegaconf import OmegaConf
from submitit import AutoExecutor

from stopes.core.cache import Cache, FileCache
from stopes.core.launcher import (
    ArrayTaskError,
    Launcher,
    TaskExecutionError,
    ThrottleConfig,
)
from stopes.core.stopes_module import Requirements, StopesModule
from stopes.core.tests.hello_world import (
    HelloWorldArrayConfig,
    HelloWorldArrayModule,
    HelloWorldConfig,
    HelloWorldModule,
)


@pytest.fixture
def cache(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    return FileCache(cache_dir)


@pytest.fixture
def launcher(tmp_path: Path, cache: Cache):
    return Launcher(
        cache=cache,
        config_dump_dir=tmp_path / "conf",
        log_folder=tmp_path / "logs",
        cluster="local",
        max_jobarray_jobs=30,
    )


@pytest.mark.parametrize("use_mem", [True, False])
async def test_supports_mem(tmp_path: Path, use_mem: bool):
    mod = HelloWorldModule(HelloWorldConfig(duration=None))
    launcher1 = Launcher(
        config_dump_dir=tmp_path / "conf",
        log_folder=tmp_path / "logs",
        cluster="local",
        supports_mem_spec=use_mem,
    )

    with mock.patch("submitit.auto.auto.AutoExecutor.update_parameters") as mock_update:
        await launcher1.schedule(mod)
        assert (mock.call(mem_gb=10) in mock_update.call_args_list) == use_mem


async def test_schedule_one(launcher: Launcher):
    mod = HelloWorldModule(HelloWorldConfig(greet="Hello", person="Foo", duration=None))
    result = await launcher.schedule(mod)

    assert result == "Hello Foo !"


async def test_schedule_array(launcher: Launcher):
    mod = HelloWorldArrayModule(
        HelloWorldArrayConfig(greet="Hello", persons=["Foo", "Bar", "Baz"], duration=0)
    )

    result = await launcher.schedule(mod)
    assert result == ["Hello Foo !", "Hello Bar !", "Hello Baz !"]


@contextlib.contextmanager
def _setup_mocked_launcher(tmp_path: Path):
    # we need a mock to count, and a new executor to do the right thing
    # ideally, we would want to count calls to HelloWorldModule.run, but the
    # submitit suprocess seems to be confusing for the mock framework
    executor = AutoExecutor(folder=tmp_path / "logs", cluster="local")
    # ideally, patch without side_effect would be enough, but it means that
    # we have MagicMocks all the way down, and jobs are mocked as well as other
    # submitit internals, which eventually break. Instead, we just call another
    # executor and get a valid job execution from that
    with mock.patch(
        "submitit.auto.auto.AutoExecutor.submit", side_effect=executor.submit
    ) as mock_submit:
        with mock.patch(
            "submitit.auto.auto.AutoExecutor.batch", side_effect=executor.batch
        ) as mock_batch:
            yield (mock_submit, mock_batch)


async def test_single_cache(tmp_path: Path, launcher: Launcher, cache: Cache):
    with _setup_mocked_launcher(tmp_path) as (mock_submit, _):
        mod = HelloWorldModule(
            HelloWorldConfig(greet="Hello", person="Foo", duration=None)
        )

        # first call
        result = await launcher.schedule(mod)

        assert result == "Hello Foo !"
        # we should submit the module when it's not cached
        assert mock_submit.call_count == 1

        assert cache.get_cache(mod) == result

        # second call is cached
        result2 = await launcher.schedule(mod)
        assert result2 == "Hello Foo !"
        # we should NOT submit the module once it's been cached
        assert mock_submit.call_count == 1

        # uncached module
        mod2_config = HelloWorldConfig(greet="Hello", person="Bar", duration=None)
        mod2 = HelloWorldModule(mod2_config)

        result3 = await launcher.schedule(mod2)
        assert result3 == "Hello Bar !"
        # different config, so we should submit again
        assert mock_submit.call_count == 2

        # third call is cached and not mixed up with other config
        result4 = await launcher.schedule(mod)
        assert result4 == "Hello Foo !"
        # cached, so no more call
        assert mock_submit.call_count == 2

        # another module instance with the same config except `timeout_min`,
        # should use cache
        mod2_config.requirements.timeout_min += 10
        result5 = await launcher.schedule(HelloWorldModule(mod2_config))
        assert result5 == "Hello Bar !"
        assert mock_submit.call_count == 2

        # another module instance with the same config except `gpus_per_node`,
        # should use cache
        mod2_config.requirements.gpus_per_node += 1
        result6 = await launcher.schedule(HelloWorldModule(mod2_config))
        assert result6 == "Hello Bar !"
        assert mock_submit.call_count == 2


async def test_array_cache(tmp_path: Path, launcher: Launcher, cache: Cache):
    with _setup_mocked_launcher(tmp_path) as (mock_submit, mock_batch):
        nb_persons = 50
        persons = [f"Person_{i:03d}" for i in range(nb_persons)]
        mod = HelloWorldArrayModule(
            HelloWorldArrayConfig(greet="Hello", persons=persons, duration=0)
        )
        results = await launcher.schedule(mod)
        expected = [f"Hello {p} !" for p in persons]
        assert results == expected

        for idx, val in enumerate(persons):
            assert (
                cache.get_cache(
                    mod,
                    iteration_value=val,
                    iteration_index=idx,
                )
                == expected[idx]
            )

        expected_submit = len(expected)
        expected_batch = math.ceil(mock_submit.call_count / launcher.max_jobarray_jobs)
        assert mock_submit.call_count == expected_submit
        assert mock_batch.call_count == expected_batch
        assert (
            mock_batch.call_count > 1
        )  # Checking this test actually tests that more than one job-array is submitted

        cache.invalidate_cache(mod, iteration_index=3, iteration_value=persons[3])
        results2 = await launcher.schedule(mod)
        assert results2 == expected

        # we should only call it for the forgotten cache value.
        expected_submit += 1
        expected_batch += 1
        assert mock_submit.call_count == expected_submit
        assert mock_batch.call_count == expected_batch

        # scheduling the same thing should not resubmit anything
        results3 = await launcher.schedule(mod)
        assert results3 == expected

        # no other calls should have gone through
        assert mock_submit.call_count == expected_submit
        assert mock_batch.call_count == expected_batch

        # invalidating more than max job-array size:
        nb_invalidate = launcher.max_jobarray_jobs + 1
        for ind in range(3, 3 + nb_invalidate):
            cache.invalidate_cache(
                mod, iteration_index=ind, iteration_value=persons[ind]
            )
        results4 = await launcher.schedule(mod)
        assert results4 == expected

        # scheduled jobs for invalid cache should have created 2 job arrays
        expected_submit += nb_invalidate
        expected_batch += 2
        assert mock_submit.call_count == expected_submit
        assert mock_batch.call_count == expected_batch


def test_cache_with_transient_config(tmp_path: Path, launcher: Launcher, cache: Cache):
    from dataclasses import field, make_dataclass

    TransientConfig = make_dataclass(
        "NewHelloWorldConfig",
        [("duration", tp.Optional[float], field(default=0.1, metadata={"transient": True}))],  # type: ignore
        bases=(HelloWorldConfig,),
    )
    with _setup_mocked_launcher(tmp_path) as (mock_submit, _):
        mod1 = HelloWorldModule(
            TransientConfig(greet="Hello", person="Foo", duration=None), TransientConfig
        )
        mod2 = HelloWorldModule(
            TransientConfig(greet="Hello", person="Foo", duration=0.3), TransientConfig
        )
        mod3 = HelloWorldModule(
            TransientConfig(
                greet="Hello",
                person="Foo",
                duration=None,
                requirements=Requirements(nodes=3),
            ),
            TransientConfig,
        )

        # first call
        result = asyncio.run(launcher.schedule(mod1))

        assert result == "Hello Foo !"
        # we should submit the module when it's not cached
        assert mock_submit.call_count == 1

        assert cache.get_cache(mod1) == result

        # second call is cached
        result2 = asyncio.run(launcher.schedule(mod2))
        assert result2 == "Hello Foo !"
        # we should NOT submit the module once it's been cached
        assert mock_submit.call_count == 1

        # third call is also cached
        result3 = asyncio.run(launcher.schedule(mod3))
        assert result3 == "Hello Foo !"
        # we should NOT submit the module once it's been cached
        assert mock_submit.call_count == 1


##########
# retries
##########


class RetryableModuleTest(StopesModule):
    def __init__(
        self, use_array: bool, do_retries: bool, eventually_succeeds: bool = False
    ):
        super().__init__(
            OmegaConf.create(
                {
                    "use_array": use_array,
                    "eventually_succeeds": eventually_succeeds,
                }
            )
        )
        self.do_retries = do_retries

    def array(self) -> tp.Optional[tp.List[tp.Any]]:
        return ["a" * i for i in range(1, 4)] if self.config.use_array else None

    def requirements(self) -> Requirements:
        return Requirements(cpus_per_task=1)

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> tp.Any:
        return_value = iteration_value or "DONE"
        # succeed after two tries
        if self.config.eventually_succeeds and self.retry_counts[iteration_index] == 2:
            return return_value

        # in an array test, fail for the 2nd element of the array
        if self.config.use_array:
            if iteration_index == 1:
                raise ValueError(f"retry is {self.retry_counts[iteration_index]}")
            else:
                return return_value
        else:
            # no array, just fail:
            raise ValueError(f"retry is {self.retry_counts[iteration_index]}")

    def should_retry(
        self,
        ex: Exception,
        attempt: int,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> bool:
        if not self.do_retries:
            return False
        str_ex = str(ex)
        return "ValueError" in str_ex and "retry is " in str_ex


async def test_retry_single(launcher: Launcher):
    launcher.max_retries = 3

    # # 1. we should not retry if we don't want to
    non_retriable = RetryableModuleTest(
        use_array=False,
        do_retries=False,
    )

    with pytest.raises(TaskExecutionError):
        await launcher.schedule(non_retriable)

    # 2. should retry at most max_retries
    retriable = RetryableModuleTest(
        use_array=False,
        do_retries=True,
    )

    with pytest.raises(TaskExecutionError):
        await launcher.schedule(retriable)
    assert retriable.retry_counts[0] == 3

    # 3. should retry and eventually succeed
    success = RetryableModuleTest(
        use_array=False,
        do_retries=True,
        eventually_succeeds=True,
    )
    val = await launcher.schedule(success)
    assert val == "DONE"


async def test_retry_array(launcher: Launcher):
    launcher.max_retries = 3
    # 1. we should not retry if we don't want to
    non_retriable = RetryableModuleTest(
        use_array=True,
        do_retries=False,
    )

    with pytest.raises(ArrayTaskError) as e_info:
        await launcher.schedule(non_retriable)
        assert [(0, "a"), (2, "aaa")] == e_info.value.results
        assert len(e_info.value.exceptions) == 1
        assert 1 == e_info.value.exceptions[0][0]
        assert isinstance(e_info.value.exceptions[0][1], TaskExecutionError)
    assert len(non_retriable.retry_counts) == 3
    assert all([cnt == 0 for cnt in non_retriable.retry_counts])

    # 2. should retry at most max_retries
    retriable = RetryableModuleTest(
        use_array=True,
        do_retries=True,
    )

    with pytest.raises(ArrayTaskError) as e_info:
        await launcher.schedule(retriable)
        assert [(0, "a"), (2, "aaa")] == e_info.value.results
        assert len(e_info.value.exceptions) == 1
        assert 1 == e_info.value.exceptions[0][0]
        assert isinstance(e_info.value.exceptions[0][1], TaskExecutionError)
    assert len(retriable.retry_counts) == 3
    assert retriable.retry_counts[1] == 3

    success = RetryableModuleTest(
        use_array=True,
        do_retries=True,
        eventually_succeeds=True,
    )
    val = await launcher.schedule(success)
    assert val == ["a" * i for i in range(1, 4)]


############
# throttling
############


class ThrottleModuleTest(StopesModule):
    def __init__(self, wait_time, idx=0):
        super().__init__(OmegaConf.create({"wait_time": wait_time, "idx": idx}))

    def requirements(self) -> Requirements:
        return Requirements(cpus_per_task=1)

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> tp.Any:
        time.sleep(self.config.wait_time)
        return datetime.datetime.now()

    def name(self):
        return f"throttle-{self.config.idx}"


async def test_throttling(tmp_path: Path):
    wait_time = 1.5
    shared_name = f"/test_throttling{uuid.uuid4()}"

    launcher1 = Launcher(
        cache=None,
        config_dump_dir=tmp_path / "conf",
        log_folder=tmp_path / "logs",
        cluster="local",
        max_jobarray_jobs=30,
        throttle=ThrottleConfig(limit=1, shared_name=shared_name, timeout=30),
    )
    launcher2 = Launcher(
        cache=None,
        config_dump_dir=tmp_path / "conf",
        log_folder=tmp_path / "logs",
        cluster="local",
        max_jobarray_jobs=30,
        throttle=ThrottleConfig(limit=1, shared_name=shared_name, timeout=30),
    )

    mod1 = ThrottleModuleTest(wait_time)
    mod2 = ThrottleModuleTest(wait_time)
    mod3 = ThrottleModuleTest(wait_time)

    ends = []
    for coro in asyncio.as_completed(
        [launcher1.schedule(mod1), launcher2.schedule(mod2), launcher1.schedule(mod3)]
    ):
        res = await coro
        ends.append(res)

    ends.sort()
    for end1, end2 in zip(ends, ends[1:]):
        t_diff = end2 - end1
        assert (
            t_diff.total_seconds() >= wait_time
        ), f"{t_diff} is too short, should have waited {wait_time} at least in the semaphore."


async def test_throttling_timeout(tmp_path: Path):
    launcher1 = Launcher(
        cache=None,
        config_dump_dir=tmp_path / "conf",
        log_folder=tmp_path / "logs",
        cluster="local",
        max_jobarray_jobs=30,
        throttle=ThrottleConfig(limit=2, timeout=1),
    )
    mod_slow = ThrottleModuleTest(1.5)
    # no time for the three to run under 1s
    with pytest.raises(RuntimeError):
        a = asyncio.create_task(launcher1.schedule(mod_slow))
        b = asyncio.create_task(launcher1.schedule(mod_slow))
        c = asyncio.create_task(launcher1.schedule(mod_slow))
        await a
        await b
        await c

    launcher2 = Launcher(
        cache=None,
        config_dump_dir=tmp_path / "conf",
        log_folder=tmp_path / "logs",
        cluster="local",
        max_jobarray_jobs=30,
        throttle=ThrottleConfig(limit=10, timeout=2),
    )
    # the three should run concurrently thus finish in the 2s
    mod_slow = ThrottleModuleTest(1)
    a = asyncio.create_task(launcher2.schedule(mod_slow))
    b = asyncio.create_task(launcher2.schedule(mod_slow))
    c = asyncio.create_task(launcher2.schedule(mod_slow))
    await a
    await b
    await c
