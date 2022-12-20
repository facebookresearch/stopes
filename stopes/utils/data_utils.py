# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class DataConfig:
    data_version: str = MISSING
    iteration: int = MISSING
    data_shard_dir: str = MISSING
    shard_type: str = MISSING
    bname: str = MISSING
    shard_list: tp.Optional[tp.List[str]] = MISSING
    shard_glob: tp.Optional[str] = MISSING
    meta_glob: tp.Optional[str] = MISSING
    # we can compute the number of lines as part of the pipeline instead of relying on it being precomputed
    # however, it is still possible to provide .nl files (one file per language, one line per shard in the file)
    nl_file_template: tp.Optional[str] = None
