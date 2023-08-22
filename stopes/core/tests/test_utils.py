# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import datetime
import os
import random
import tempfile
from pathlib import Path

import posix_ipc
import pytest

from stopes.core import utils
from stopes.core.utils import gather_optionals, sort_file


def test_batch():
    items = list(range(10))
    listify = lambda items: [list(item) for item in items]  # noqa

    assert listify(utils.batch([], 1)) == []
    assert listify(utils.batch([], 10)) == []
    # fmt: off
    assert listify(utils.batch(items, 1)) == [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
    # fmt: on
    assert listify(utils.batch(items, 2)) == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    assert listify(utils.batch(items, 3)) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert listify(utils.batch(items, 4)) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


async def foo(v: str) -> str:
    return v


async def test_gather_optionals():
    x, y, z = await gather_optionals(
        foo("x"),
        foo("y"),
        foo("z"),
    )
    assert x == "x"
    assert y == "y"
    assert z == "z"

    a, b, c = await gather_optionals(
        foo("a"),
        foo("b"),
        None,
    )
    assert a == "a"
    assert b == "b"
    assert c is None


@pytest.mark.parametrize("no_duplicate", [False, True])
def test_sort_files(no_duplicate):
    # Force external merge sort with intermediate files
    os.environ["STOPES_SHARD_CHUNK_SIZE"] = "100"
    with tempfile.TemporaryDirectory() as tmp_dir:
        big_array = list(range(1000)) * 2
        if no_duplicate:
            expected_array = sorted(list(range(1000)), key=str)
        else:
            expected_array = sorted(big_array, key=str)
        random.shuffle(big_array)

        # Add one extra column that shouldn't be sorted
        irrelevant_col = list("hello") * (int(len(big_array) / 5))

        infile = Path(tmp_dir) / "input"
        with open(infile, "a+", encoding="utf-8") as fh:
            fh.writelines(
                [f"line {c} {i}\n" for c, i in zip(irrelevant_col, big_array)]
            )

        outfile = Path(tmp_dir) / "output"
        sort_file(infile, outfile, col=2, sep=" ", no_duplicate=no_duplicate)
        with open(outfile, encoding="utf-8") as fh:
            for idx, line in zip(expected_array, iter(fh)):
                assert int(line.rstrip().split()[2]) == idx


async def _sem_test_function(sem_name, wait_time=0):
    async with utils.AsyncIPCSemaphore(
        name=sem_name,
        flags=posix_ipc.O_CREAT,
        initial_value=1,
        timeout=1
        + 2
        * wait_time,  # the first instantiation with the name sets the value, subsequent ones will ignore this param
    ) as sem:
        if wait_time:
            await asyncio.sleep(wait_time)
        # inside here, we have aquired the semaphore, value should be 0
        return sem.value, datetime.datetime.now()


async def test_semaphore():
    sem = utils.AsyncIPCSemaphore(
        name="/stopes_test_semaphore_1",
        flags=posix_ipc.O_CREAT,
        initial_value=1,
        timeout=30,  # the first instantiation with the name sets the value, subsequent ones will ignore this param
    )

    await sem.acquire()
    assert sem.value == 0

    sem.release()
    assert sem.value == 1

    sleep_time = 1.5
    ends = []
    for coro in asyncio.as_completed(
        [_sem_test_function("/stopes_test_semaphore_1", sleep_time) for _ in range(3)]
    ):
        sem_value, end = await coro
        ends.append(end)
        assert (
            sem_value == 0
        ), f"semaphore count should be 0 when in the execution block, got {sem_value}."

    # make sure that the semaphore blocks execution
    ends.sort()
    for (end1, end2) in zip(ends, ends[1:]):
        t_diff = end2 - end1
        assert (
            t_diff.total_seconds() >= sleep_time
        ), f"{t_diff} is too short, should have waited {sleep_time} at least in the semaphore."
