# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import logging
import typing as tp
from dataclasses import dataclass
from pathlib import Path

from omegaconf.omegaconf import MISSING

from stopes.core.stopes_module import Requirements, StopesModule

from .vocal_style_sim_tool import compute_cosine_similarity, get_embedder

logger = logging.getLogger(__name__)


@dataclass
class VocalStyleSimilarityConfig:
    input_file: tp.Optional[Path] = None
    input_files: tp.Optional[tp.List[Path]] = None
    output_file: Path = MISSING
    src_audio_column: tp.Union[int, str] = 1
    tgt_audio_column: tp.Union[int, str] = 2
    named_columns: bool = False
    model_type: str = "valle"
    # A model can be downloaded from https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification
    # Use the last one (WavLM large without fixed pre-train) to reproduce the evaluation we have done
    model_path: str = MISSING
    use_cuda: bool = True


class VocalStyleSimilarityModule(StopesModule):
    def __init__(self, config: tp.Any):
        super().__init__(config, VocalStyleSimilarityConfig)
        self.config: VocalStyleSimilarityConfig
        self.config.output_file.parent.mkdir(exist_ok=True, parents=True)
        assert self.config.input_files is not None or self.config.input_file is not None

        if self.config.named_columns:
            assert isinstance(self.config.src_audio_column, str)
            assert isinstance(self.config.tgt_audio_column, str)
        else:
            assert isinstance(self.config.src_audio_column, int)
            assert isinstance(self.config.tgt_audio_column, int)
        # TODO: maybe unify all the boilerplate code for reading inputs and outputs

    def requirements(self) -> Requirements:
        return Requirements(gpus_per_node=1)

    def array(self):
        if self.config.input_files is not None:
            return self.config.input_files
        return None

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> tp.Any:
        if iteration_value is not None:
            input_file = iteration_value
            output_file = Path(str(self.config.output_file) + f".{iteration_index}")
        else:
            input_file = self.config.input_file
            output_file = self.config.output_file

        src_paths, tgt_paths = [], []
        with open(input_file, "r", encoding="utf-8") as f:
            reader: tp.Union[tp.Iterator[tp.List], tp.Iterator[tp.Dict]]
            if self.config.named_columns:
                reader = csv.DictReader(f, delimiter="\t")
            else:
                reader = csv.reader(f, delimiter="\t")
            for row in reader:
                src_paths.append(row[self.config.src_audio_column])  # type: ignore
                tgt_paths.append(row[self.config.tgt_audio_column])  # type: ignore

        embedder = get_embedder(
            model_name=self.config.model_type,
            model_path=self.config.model_path,
            use_cuda=self.config.use_cuda,
        )
        src_embs = embedder(src_paths)
        tgt_embs = embedder(tgt_paths)
        similarities = compute_cosine_similarity(src_embs, tgt_embs)

        with open(output_file, "w") as f:
            for score in similarities:
                print(score, file=f)
        return output_file

    def validate(
        self,
        output: tp.Any,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> bool:
        return output.exists()
