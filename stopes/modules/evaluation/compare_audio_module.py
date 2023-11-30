# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import logging
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf.omegaconf import MISSING

from stopes.core.stopes_module import Requirements, StopesModule
from stopes.eval.auto_pcp.audio_comparator import compare_audio_pairs
from stopes.utils.web import cached_file_download

logger = logging.getLogger(__name__)


@dataclass
class CompareAudiosConfig:
    # How to read the inputs
    named_columns: bool = False
    input_file: tp.Optional[Path] = None
    input_files: tp.Optional[tp.List[Path]] = None
    src_audio_column: tp.Union[int, str] = 1
    tgt_audio_column: tp.Union[int, str] = 2

    # Where to save the results
    output_file: Path = MISSING

    # How to get the comparator
    comparator_path: tp.Optional[Path] = None
    comparator_url: tp.Optional[str] = None
    comparator_save_name: tp.Optional[str] = None

    # How to compute the scores
    encoder_path: str = "facebook/wav2vec2-large-xlsr-53"
    pick_layer: int = 7
    symmetrize: bool = False
    use_cuda: bool = True
    batch_size: int = 16
    num_process: tp.Optional[int] = 1


def validate_inputs(config: CompareAudiosConfig):
    """Check consistency of the input format"""
    assert config.input_files is not None or config.input_file is not None

    if config.named_columns:
        assert isinstance(config.src_audio_column, str)
        assert isinstance(config.tgt_audio_column, str)
    else:
        assert isinstance(config.src_audio_column, int)
        assert isinstance(config.tgt_audio_column, int)

    if config.comparator_path is None and (
        config.comparator_url is None or config.comparator_save_name is None
    ):
        raise ValueError(
            "Please provide either `comparator_path`, or `comparator_url` and `comparator_save_name` to download the model"
        )


class CompareAudiosModule(StopesModule):
    def __init__(self, config: tp.Any):
        super().__init__(config, CompareAudiosConfig)
        self.config: CompareAudiosConfig
        validate_inputs(self.config)
        self.config.output_file.parent.mkdir(exist_ok=True, parents=True)

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

        comparator_path = self.config.comparator_path
        if comparator_path is None:
            comparator_path = cached_file_download(
                self.config.comparator_url,  # type: ignore
                self.config.comparator_save_name,  # type: ignore
                unzip=True,
            )

        results = compare_audio_pairs(
            src_paths,
            tgt_paths,
            comparator_path=comparator_path,
            use_cuda=self.config.use_cuda and torch.cuda.is_available(),
            encoder_path=self.config.encoder_path,
            batch_size=self.config.batch_size,
            pick_layer=self.config.pick_layer,
            symmetrize=self.config.symmetrize,
            num_process=self.config.num_process,
        )
        with open(output_file, "w") as f:
            for score in results:
                print(score, file=f)
        return output_file

    def validate(
        self,
        output: tp.Any,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> bool:
        return output.exists()
