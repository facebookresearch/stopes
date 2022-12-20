# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import logging
import typing as tp
from pathlib import Path

from stopes.core.stopes_module import Requirements, StopesModule

logger = logging.getLogger("preprocess_encode")


class PreprocessEncodeModule(StopesModule):
    def __init__(
        self,
        config,
    ):
        super().__init__(config)
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
        reqs = self.config.encode.config.requirements
        if not isinstance(reqs, Requirements):
            # Performe conversion if needed
            return Requirements(**reqs)
        return reqs

    def array(self):
        return self.shards

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        lang_shard_name = getattr(
            self.config, "lang_shard_name", None
        )  # this is used if we split a language

        preprocessed_file = iteration_value
        if getattr(self.config, "preprocess", None):  # preprocess if specified
            preprocess_module = StopesModule.build(
                self.config.preprocess,
                lang=self.config.lang,
                lang_shard_name=lang_shard_name,
                shards=[str(iteration_value)],
            )
            preprocessed_file = preprocess_module.run(
                iteration_value=str(iteration_value),
                iteration_index=iteration_index,
            )

        encode_module = StopesModule.build(
            self.config.encode,
            outfile_prefix=f"{self.config.encode.config.outfile_prefix}",
            outfile_postfix=f"{lang_shard_name or self.config.lang}",
            shards=[str(preprocessed_file)],
        )
        return encode_module.run(
            iteration_value=str(preprocessed_file),
            iteration_index=iteration_index,
        )

    def version(cls):
        return "0.3"

    def name(self):
        return f"preprocess_encode.{self.config.lang}.{len(self.shards)}"
