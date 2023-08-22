# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import glob
import logging
import typing as tp
from pathlib import Path

import omegaconf
import submitit

from stopes.core import Requirements, StopesModule, utils
from stopes.modules.preprocess import (
    LineProcessorConfig,
    LineProcessorModule,
    MosesPreprocessConfig,
)

logger = logging.getLogger("preprocess_encode")


@dataclasses.dataclass
class PreprocessEncodeConfig:
    encode: LineProcessorConfig
    encoder: tp.Any
    lang: str
    lang_shard_name: tp.Optional[str]
    output_dir: Path
    # shards is either a list of files or a glob string
    # if only hydra allowed, the right type would be tp.Union[str, tp.List[str]]
    shards: tp.Any = None
    preprocess: tp.Any = None


class PreprocessEncodeModule(StopesModule, submitit.helpers.Checkpointable):
    config: PreprocessEncodeConfig

    def __init__(self, config: PreprocessEncodeConfig):
        super().__init__(config, PreprocessEncodeConfig)
        assert (
            self.config.encoder == self.config.encode.line_processor
        ), "Don't override embed_text.encode.line_processor, but just embed_text.encoder"

        self.shards = self.config.shards
        if isinstance(self.shards, str):
            # it's a glob instead of a list of files
            self.shards = sorted(list(glob.glob(self.shards)))
        self.config.output_dir.mkdir(exist_ok=True)

    def requirements(self) -> Requirements:
        # Encoding is the most expensive part, use those requirements.
        reqs = self.config.encode.requirements
        if not isinstance(reqs, Requirements):
            # Performe conversion if needed
            reqs = Requirements(**reqs)
        return reqs

    def array(self) -> tp.List[str]:
        return self.shards

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> Path:
        lang = self.config.lang
        lang_shard_name = self.config.lang_shard_name

        if getattr(self.config, "preprocess", None):
            preprocess_module = StopesModule.build(
                self.config.preprocess,
                lang=lang,
                lang_shard_name=lang_shard_name,
                shards=[str(iteration_value)],
            )
            preprocessed_file = preprocess_module.run(
                iteration_value=str(iteration_value),
                iteration_index=iteration_index,
            )
        else:
            preprocessed_file = iteration_value

        with utils.clone_config(self.config.encode) as encode_cfg:
            encode_cfg.outfile_postfix = lang_shard_name or lang
            encode_module = LineProcessorModule(encode_cfg, validate_config=True)

        return encode_module.run(  # type:ignore[no-any-return]
            iteration_value=str(preprocessed_file),
            iteration_index=iteration_index,
        )

    @staticmethod
    def version() -> str:
        return "0.3"

    def name(self) -> str:
        return f"preprocess_encode.{self.config.lang}.{len(self.shards)}"
