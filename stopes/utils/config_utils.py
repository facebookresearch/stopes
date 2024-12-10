# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional

from omegaconf import ListConfig


def parse_hydra_list(lst_config: Any) -> Optional[List]:
    """
    robust parsing of hydra argumnts, accepting formats like
    [1,2,3,5] OR [1] OR 1 OR 1,2 OR ['src','tgt']
    """
    if lst_config is None:
        return None
    if isinstance(lst_config, (list, ListConfig)):
        lst_value: list = list(lst_config)
    elif isinstance(lst_config, str):
        lst_value = lst_config.strip().strip("[]").split(",")
    else:
        lst_value = [lst_config]
    return lst_value
