# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio
import xxhash
from omegaconf import MISSING
from tqdm import tqdm

import stopes.modules.speech.utils as speech_utils
from stopes.core import utils as stopes_utils
from stopes.core.stopes_module import Requirements, StopesModule
from stopes.modules.speech.vad import VAD

logger = logging.getLogger("vad_trim_audio")

SAMPLING_FACTOR = 16


@dataclass
class VADTrimAudioConfig:
    shards: tp.Any = MISSING
    audio_column: int = MISSING
    output_dir: Path = MISSING
    model: Path = MISSING
    save_trimmed_audio: tp.Optional[bool] = False
    speech_threshold: tp.Optional[float] = 0.3
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100
    window_size_samples: int = 1536
    speech_pad_ms: int = 30


class VADTrimAudioModule(StopesModule):
    """
    Run VADTrimModule to trim any silence from the start and end of an audio. 
    Expects a tab-separated input file, with a column specified for the relevant
    audio to run the trimming on. 

    It returns a manifest file with the original input lines, the trimmed audios, and 
    also a boolean value letting the user know if the audio was trimmed or not 
    (in some cases the VAD may not recognise the input speech at all and in these 
    instances we default back to the original audio without any silence trimmed).
    There is also an option to either save the trimmed audios as a new wav or simply 
    append the start and end frames to the input audio in the resulting manifest.

    Example command:

    python -m stopes.modules +speech_preproc=vad_trim \
        speech_preproc.shards=/test/input.tsv \
        speech_preproc.output_dir=/path/to/output \
        speech_preproc.model=/path/to/model \
        speech_preproc.audio_column=1 \
        speech_preproc.save_trimmed_audio=True  \
        speech_preproc.speech_threshold=0.3
    """

    def __init__(self, config: VADTrimAudioConfig = VADTrimAudioConfig()):
        super().__init__(config, VADTrimAudioConfig)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def requirements(self):
        return Requirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=1,
            cpus_per_task=1,
            timeout_min=180,
        )

    def array(self):
        if isinstance(self.config.shards, str):
            return None
        return self.config.shards

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        if iteration_value is None:
            iteration_value = self.config.shards
        assert isinstance(
            iteration_value, (str, Path)
        ), "Input value must be a path to the manifest file"

        model = self.load_model()
        digest = xxhash.xxh3_64_intdigest(str(iteration_value))
        filename = f"vad_trim_audio.{iteration_index:05d}.{digest}.gz"
        output_manifest_filename = self.config.output_dir / filename
        with stopes_utils.open(
            output_manifest_filename, mode="wt", encoding="utf-8"
        ) as out_file:
            with stopes_utils.open(iteration_value) as infile:
                for _, line in tqdm(enumerate(infile)):
                    line = line.rstrip("\n")
                    trimmed_audio, is_trimmed = self.trim_audio(model, line)
                    out_file.write(
                        "\t".join([line, trimmed_audio, str(is_trimmed)]) + "\n"
                    )
        return output_manifest_filename

    def load_model(self) -> tp.Any:
        return VAD(model_path=self.config.model)

    def load_audio(self, line: str) -> tp.Tuple[torch.Tensor, str]:
        audio = line.split("\t")[self.config.audio_column]
        audio_obj = speech_utils.parse_audio(audio, sampling_factor=SAMPLING_FACTOR)
        audio_information = str(audio_obj)
        # get string-based representation of audio segment
        wav = audio_obj.load(average_channels=True)
        if wav.ndim > 1:
            wav = wav.squeeze(0)  # remove channel dimension
        return wav, audio_information

    def trim_timestamps(
        self, timestamps: np.ndarray, wav: torch.Tensor
    ) -> tp.Tuple[int, int]:
        min_frame_start, max_frame_end = 0, wav.size(-1)
        if len(timestamps) > 0:
            start_frames, end_frames = zip(*timestamps)
            min_frame_start = min(start_frames)
            max_frame_end = max(end_frames)
        return min_frame_start, max_frame_end

    def save_audio(
        self, wav: torch.Tensor, audio_information: str, start: int, end: int
    ) -> str:
        digest = xxhash.xxh3_64_intdigest(f"{audio_information} {start} {end}")
        outfile = str(self.config.output_dir / f"trimmed_{digest}.wav")
        torchaudio.save(outfile, wav.unsqueeze(0)[:, start:end], SAMPLING_FACTOR * 1000)
        return outfile

    def trim_audio(self, model: tp.Any, line: str) -> tp.Tuple[str, bool]:
        wav, audio_information = self.load_audio(line)
        timestamps = model.get_timestamps(
            wav,
            threshold=self.config.speech_threshold,
            min_speech_duration_ms=self.config.min_speech_duration_ms,
            min_silence_duration_ms=self.config.min_silence_duration_ms,
            window_size_samples=self.config.window_size_samples,
            speech_pad_ms=self.config.speech_pad_ms,
        )
        start, end = self.trim_timestamps(timestamps, wav)
        is_trimmed = False if start == 0 and end == wav.size(-1) else True
        if self.config.save_trimmed_audio:
            trimmed_audio = self.save_audio(wav, audio_information, start, end)
        else:
            trimmed_audio = f"{audio_information} {start} {end}"
        return trimmed_audio, is_trimmed
