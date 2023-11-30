# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class M4TEvalConfig:
    launcher: tp.Any
    output_dir: str = MISSING
    data_dir: str = MISSING
    task: str = MISSING
    dataset_split: str = MISSING
    audio_root_dir: str = MISSING
    lang_dirs: tp.List[str] = MISSING
    kwargs: tp.Dict[str, str] = MISSING


@dataclass
class M4TEvalJob:
    src_lang: str
    tgt_lang: str


@dataclass
class M4TEvalModuleConfig:
    output_dir: str = MISSING
    data_dir: str = MISSING
    dataset_split: str = MISSING
    task: str = MISSING
    audio_root_dir: str = MISSING
    kwargs: tp.Dict[str, str] = MISSING
    eval_jobs: tp.List[M4TEvalJob] = MISSING
