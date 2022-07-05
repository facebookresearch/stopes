# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import logging
import typing as tp
from pathlib import Path

import hydra

from stopes.core.stopes_module import LocalOnlyRequirements, StopesModule

logger = logging.getLogger("preprocess_encode")


class PreprocessEncodeModule(StopesModule):
    def __init__(
        self,
        config,
    ):
        super().__init__(config)
        self.launcher = hydra.utils.instantiate(config.launcher)
        self.shards = self.config.shards
        if isinstance(self.shards, str):
            # it's a glob instead of a list of files
            self.shards = list(glob.glob(self.shards))

        # Create output dir so that child module can use it
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def requirements(self):
        #  this just stiches other modules together
        #  so we really just want it to run inline with the local
        #  coordinator script
        return LocalOnlyRequirements()

    async def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        files_to_embed = self.shards

        if hasattr(self.config, "preprocess"):  # preprocess if specified
            preprocess_module = StopesModule.build(
                self.config.preprocess,
                lang=self.config.lang,
                shards=files_to_embed,
            )
            files_to_embed = await self.launcher.schedule(preprocess_module)

        encode_module = StopesModule.build(
            self.config.encode,
            outfile_prefix=f"{self.config.encode.config.outfile_prefix}",
            outfile_postfix=f"{self.config.lang}",
            shards=[str(f) for f in files_to_embed],
        )
        return await self.launcher.schedule(encode_module)

    def version(cls):
        return "0.2"

    def name(self):
        return f"preprocess_encode.{self.config.lang}.{len(self.shards)}"
