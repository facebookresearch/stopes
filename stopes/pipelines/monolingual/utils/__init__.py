# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import typing as tp
from pathlib import Path


def split_list(list: tp.List[Path], chunk_size: int) -> tp.Iterator[Path]:
    for i in range(0, len(list), chunk_size):
        yield list[i : i + chunk_size]


def slurm_tmp_maybe(tmp_dir: Path) -> Path:
    slurm_env = os.environ.get("SLURM_JOB_ID", None)
    if slurm_env:
        tmp_dir = tmp_dir / slurm_env
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir
