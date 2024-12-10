# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import functools
import logging
import typing as tp
from pathlib import Path
from typing import Any, List, Optional

import omegaconf
import submitit
import torch
from torch import multiprocessing
from tqdm import tqdm

from stopes import hub
from stopes.core import Requirements, StopesModule, utils
from stopes.modules.speech.audio_load_utils import parallel_audio_read
from stopes.speech.tokenizers import SpeechTokenizer
from stopes.utils.sharding.text_shards import (
    TextShard,
    make_text_file_shards,
    resolve_output,
)

log = logging.getLogger("stopes.speech.units")
multiprocessing.set_start_method("spawn", force=True)


@dataclasses.dataclass
class SpeechUnitsConfig:
    shards: tp.Any
    output_dir: Path
    column: str

    tokenizer: tp.Any
    suffix: str = ".tsv.gz"

    # Optional: number of lines to be processed from
    # each input file. Default None (process all lines)
    nrows: tp.Optional[int] = None

    # runtime requirements:
    # nshards = no. of shards to split the inputs
    nshards: int = 1

    # accepted value: chunk | hash
    algo: str = "chunk"


def units_to_text(units: torch.Tensor, fmt="list") -> str:
    """
    Convert the unit tensor into human-readable text:

    1) For one-dimensional units:
        [tok1, tok2, tok3,..., tok_n]
    2) For multi-dimensional units (e.g encodec) of n tokens, each with m code books:
        [[code1_of_tok1, code1_of_tok2,, .., code1_of_tok_n], [code2_of_tok1, .., code2_of_tok_n], ..]

    To reconstruct to units:

    lst = ast.literal_eval(<units text form>)
    units = torch.tensor(lst)

    Args:
        fmt (str): currently only "list", meaning tensor is converted to list representation
    """
    assert units.dim() == 2 or units.dim() == 3
    units = units.squeeze(0)
    if fmt == "list":
        return " ".join(map(str, units.tolist()))
    else:
        raise NotImplementedError("Currently only support list format")


class SpeechUnitsModule(StopesModule):
    """
    Compute Wav2vec speech units for one mining result file.

    Example commands:

    1) Run the speech_units with a default speech tokenizer (xlsr1b for En, feature_layer 35)

    python -m stopes.modules +speech_units=base \
        speech_units.shards=/path/to/audio_manifest_file \
        speech_units.output_dir=/path/to/output_dir \
        speech_units.column=[column index or header]

    2) Run the speech_units with a speech tokenizer "encodec_24khz"

    python -m stopes.modules +speech_units=base \
        +speech_tokenizer=encodec_24khz \
        speech_units.shards=/path/to/audio_manifest_file \
        speech_units.output_dir=/path/to/output_dir \
        speech_units.column=[column index or header]
    """

    config: SpeechUnitsConfig

    def __init__(self, config: SpeechUnitsConfig):
        super().__init__(config)
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self._current_progress: Optional[TextShard] = None

    @functools.cached_property
    def tokenizer(self) -> SpeechTokenizer:
        if isinstance(self.config.tokenizer, (dict, omegaconf.DictConfig)):
            return hub.speech_tokenizer(**self.config.tokenizer)  # type: ignore
        else:
            assert isinstance(self.config.tokenizer, str)
            return hub.speech_tokenizer(self.config.tokenizer)

    def requirements(self) -> Requirements:
        # Rule of thumbs: Use maximum 4 cpus per task, 1 task per node
        return Requirements(
            gpus_per_node=int(self.tokenizer.config["gpu"]),
            cpus_per_task=4,
            constraint=None if self.tokenizer.config["fp16"] else "volta32gb",
        )

    def array(self) -> List[TextShard]:
        header = (
            isinstance(self.config.column, str) and not self.config.column.isdecimal()
        )
        return list(
            make_text_file_shards(
                self.config.shards,
                cache_dir=self.output_dir,
                header=header,
                sep="\t",
                nshards=self.config.nshards,
                nrows=getattr(self.config, "nrows", None),
                algo=self.config.algo,
            )
        )

    @functools.lru_cache
    def name(self) -> str:
        return str(self.tokenizer)

    def run(
        self,
        iteration_value: Optional[Any] = None,
        iteration_index: Optional[int] = None,
    ) -> Any:
        assert isinstance(iteration_value, TextShard)
        shard = iteration_value
        self._current_progress = shard
        lang = getattr(self.tokenizer.config, "lang", None)
        if not lang:
            lang = self.config.column
        output_file = resolve_output(
            shard,
            Path(self.config.output_dir),
            suffix=self.config.suffix,
        )
        assert (
            output_file is not None
        ), f"Cannot write result from {shard.input_file} to {self.config.output_dir}"
        mode = "a" if shard.has_started() else "w"
        if mode == "w" and output_file.exists():
            log.warning(f"Output file {output_file} exists and will be overriden")

        with iteration_value as progress, utils.open(output_file, mode) as o:
            column_offset = progress.resolve_column_index(self.config.column)
            lines = iter(progress)
            num_cpus = self.requirements().cpus_per_task
            for line, audio in tqdm(
                parallel_audio_read(
                    lines,
                    column_offset,
                    self.tokenizer.config["gpu"],
                    self.tokenizer.config["fp16"],
                    num_process=num_cpus,
                ),
                unit="segment",
            ):
                audio = torch.tensor(audio, dtype=torch.float)
                if audio.ndim == 1:
                    audio = audio.unsqueeze(
                        0
                    )  # audio loader returns 1D tensors on non-segmented inputs
                units = self.tokenizer.encode(audio)
                columns = line.rstrip("\n").split("\t")
                print(
                    # The first column is typically the `id` or the mining score,
                    # and it's useful to propagate it in the output.
                    columns[0],
                    columns[column_offset],
                    units_to_text(units),
                    sep="\t",
                    flush=True,
                    file=o,
                )

        return Path(output_file)

    def checkpoint(
        self,
        iteration_value: TextShard,
        iteration_index: int,
        **kwargs: Any,
    ) -> submitit.helpers.DelayedSubmission:
        progress = self._current_progress or iteration_value
        # resubmit the module with updated progress
        return submitit.helpers.DelayedSubmission(
            self, progress, iteration_index, **kwargs
        )
