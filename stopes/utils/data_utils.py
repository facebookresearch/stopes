# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from collections import UserDict
from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class DataConfig:
    data_version: str = MISSING
    iteration: int = MISSING
    data_shard_dir: str = MISSING
    shard_type: str = MISSING
    bname: str = MISSING
    shard_list: tp.Optional[str] = MISSING
    shard_glob: tp.Optional[str] = MISSING
    meta_glob: tp.Optional[str] = MISSING
    # we can compute the number of lines as part of the pipeline instead of relying on it being precomputed
    # however, it is still possible to provide .nl files (one file per language, one line per shard in the file)
    nl_file_template: tp.Optional[str] = None


class DictWithOverrides(UserDict):
    """
    A custom dictionary where key and value can be overwritten at runtime by user-defined function.

    Note that any newly added transformation function will not affect the
    existing key / value.
    """

    def __init__(self, **kwargs):
        self.type = dict
        self.key_lambdas = []
        self.value_lambdas = []
        UserDict.__init__(self, kwargs)

    def __getitem__(self, key: tp.Any) -> tp.Any:
        for func in self.key_lambdas:
            key = func(key)
        return super().__getitem__(key)

    def __setitem__(self, key: tp.Any, item: tp.Any) -> None:
        for kfunc in self.key_lambdas:
            key = kfunc(key)
        for vfunc in self.value_lambdas:
            item = vfunc(item)
        return super().__setitem__(key, item)

    def register_key_hooks(self, func: tp.Callable) -> None:
        self.key_lambdas.append(func)

    def register_value_hooks(self, func: tp.Callable) -> None:
        self.value_lambdas.append(func)
