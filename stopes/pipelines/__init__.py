# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import UserDict

from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin


class StopesConfigRegistry(UserDict):
    def __setitem__(self, key: str, item: str) -> None:
        super().__setitem__(key, item)

        # Register key as config provider and value as config path
        stopes_config_plugin = type(
            "StopesConfigPath",
            (SearchPathPlugin, object),
            {
                "manipulate_search_path": lambda self, search_path: search_path.append(
                    provider=key, path=item
                )
            },
        )
        Plugins.instance().register(stopes_config_plugin)


config_registry = StopesConfigRegistry()


# Register the common pipeline configs in stopes
config_registry["stopes-common"] = "pkg://stopes/pipelines/conf"
config_registry["stopes-text-mining"] = "pkg://stopes/pipelines/bitext/conf"
config_registry["stopes-speech-mining"] = "pkg://stopes/pipelines/speech/conf"
