# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import tempfile
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from omegaconf import MISSING

import stopes.modules.speech.utils as speech_utils
from stopes.core.stopes_module import Requirements, StopesModule
from stopes.modules.speech.audio_load_utils import load_audio
from stopes.modules.speech.speech_units import parallel_audio_read
from stopes.utils.sharding.text_shards import (
    TextShard,
    make_text_file_shards,
    parse_header,
    resolve_output,
)

logger = logging.getLogger("stopes.whisper")

WHISPER_SAMPLING_RATE = 16_000


@dataclass
class WhisperConfig:
    shards: tp.Any = MISSING

    # column index (0,1) or column name ("src_audio", "tgt_audio",..)
    column: tp.Union[int, str] = MISSING
    output_dir: Path = MISSING
    model: str = MISSING
    lang: tp.Optional[str] = None
    longest_segment: bool = MISSING
    parallel_audio_read: bool = False
    output_timestamp: bool = False
    output_info: bool = True
    tmp_dir: tp.Optional[Path] = None

    # runtime requirements:
    # nshards = no. of shards to split the inputs
    nshards: int = 1
    gpu: bool = True
    cpus_per_task: int = 4
    timeout_min: int = 180


class WhisperModule(StopesModule):

    config: WhisperConfig

    def __init__(self, config: WhisperConfig):
        super().__init__(config, WhisperConfig)
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.header = (
            isinstance(self.config.column, str) and not self.config.column.isdecimal()
        )
        self._current_progress: tp.Optional[TextShard] = None

    def array(self) -> tp.List[TextShard]:
        return list(
            make_text_file_shards(
                self.config.shards,
                nshards=self.config.nshards,
                header=self.header,
                cache_dir=self.config.tmp_dir,
                sep="\t",
                col=self.config.column,
            )
        )

    def requirements(self) -> Requirements:
        return Requirements(
            gpus_per_node=int(self.config.gpu),
            cpus_per_task=int(self.config.cpus_per_task),
            timeout_min=int(self.config.timeout_min),
        )

    def get_audio(self, infile: str, ts_start: int, ts_end: int) -> torch.Tensor:
        signal = speech_utils.read_audio(infile, WHISPER_SAMPLING_RATE)

        if ts_start is not None and ts_end is not None:
            signal = signal[ts_start:ts_end]
        return signal

    def get_lines(
        self,
        lines: tp.Iterator[str],
        column_offset: int,
    ) -> tp.Iterator[tp.Tuple[str, int, int]]:
        """
        Iterate over the audio segments based on the input file.
        If `longest_segment` is False, just parse the file line by line.
        If `longest_segment` is True, yield only the longest segment for each segment start.
        In the latter case, we assume that the file is ordered by (infile, ts_start).
        """
        current_segment = None
        current_segment_largest_ts_end = 0
        for line in lines:
            line = line.rstrip()
            line = line.split("\t")[column_offset]
            infile, ts_start, ts_end, _ = speech_utils.parse_audio_deprecated(line)
            # If no ts_start and ts_end provided,
            # return ts_start = 0 and ts_end = len(wav) to read the full audio instead
            if not (ts_start and ts_end) and not self.config.longest_segment:
                ts_start = 0
                line, wav = load_audio(
                    column_offset,
                    gpu=False,
                    fp16=True,
                    line=line,
                    sampling_factor=16,
                    collapse_channels=True,
                )
                ts_end = len(wav)
            assert (
                ts_start is not None and ts_end is not None
            ), f"Cannot parse timetamp from audio segment info: {line}"
            ts_start = int(ts_start)
            ts_end = int(ts_end)
            if not self.config.longest_segment:
                yield (infile, ts_start, ts_end)
            else:
                segment_key = (infile, ts_start)

                if current_segment is None:
                    current_segment = segment_key
                    current_segment_largest_ts_end = ts_end

                if segment_key != current_segment:
                    yield (*current_segment, current_segment_largest_ts_end)
                    current_segment = segment_key
                    current_segment_largest_ts_end = ts_end
                else:
                    current_segment_largest_ts_end = max(
                        current_segment_largest_ts_end, ts_end
                    )
        if current_segment is not None:
            yield (*current_segment, current_segment_largest_ts_end)

    def iterate_audios(
        self,
        input_lines: tp.Iterator[str],
        column_offset: int = 0,
    ) -> tp.Iterator[tp.Tuple[tp.Union[np.ndarray, torch.Tensor], str]]:
        """Read the audios from the open manifest and yield the waveforms and the descriptions"""
        if self.config.parallel_audio_read:
            for audio_info, wav in parallel_audio_read(
                input_lines, column_offset=column_offset
            ):
                wav = wav.squeeze()
                audio_info = audio_info.strip()
                yield wav, audio_info
        else:
            for infile, ts_start, ts_end in self.get_lines(
                input_lines, column_offset=column_offset
            ):
                wav = self.get_audio(infile, ts_start, ts_end)  # type: ignore
                audio_info = " ".join([infile, str(ts_start), str(ts_end)])
                yield wav, audio_info

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        import whisper

        shard = iteration_value
        # Shard is None when the run() is called directly, e.g. when running the
        # module from command line: python -m stopes.modules.speech.audio_zip ...
        # In this case we create a dummy Shard object with index = None
        if shard is None:
            assert Path(
                self.config.shards
            ).is_file(), "Direct call of run() only works with a single shard"
            cols = parse_header(self.config.shards, "\t") if self.header else None
            shard = TextShard(
                input_file=self.config.shards, columns=cols, sep="\t", filter=None
            )
        self._current_progress = shard

        assert (
            self.model is None
        ), "Environment variable DATA_GYM_CACHE_DIR was overridden assuming `run` is called only once per instance."

        out_file = resolve_output(shard, Path(self.config.output_dir), suffix=".tsv")
        assert (
            out_file
        ), f"Cannot determine the output file name for {shard.input_file} (shard #{shard.index})"
        column_offset = shard.resolve_column_index(self.config.column)
        with (
            shard as f,
            open(out_file, "a+") as o,
            tempfile.TemporaryDirectory() as data_gym_cache,
        ):
            os.environ["DATA_GYM_CACHE_DIR"] = str(data_gym_cache)
            self.model = whisper.load_model(self.config.model)

            for audio_signal, audio_info in self.iterate_audios(
                iter(f), column_offset=column_offset
            ):
                transcription_result = self.model.transcribe(  # type: ignore
                    audio_signal, verbose=False, language=self.config.lang
                )
                result_text = transcription_result["text"]
                result_lang = transcription_result["language"]

                output_txt: tp.List[str] = []
                if self.config.output_info:
                    output_txt += [result_lang, audio_info]
                output_txt.append(result_text)
                if self.config.output_timestamp:
                    # concat the segments in case there are more than one of them
                    begin = transcription_result["segments"][0]["start"]
                    end = transcription_result["segments"][-1]["end"]
                    output_txt.append(begin)
                    output_txt.append(end)

                print(*output_txt, sep="\t", file=o)
                o.flush()

        return out_file
