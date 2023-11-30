# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import logging
import typing as tp
from dataclasses import dataclass
from importlib.machinery import SourceFileLoader
from pathlib import Path

import fairseq
import torchaudio
import xxhash
from omegaconf import MISSING

from stopes.core import utils
from stopes.core.stopes_module import Requirements, StopesModule
from stopes.modules.speech.shas.shas import SHAS

logger = logging.getLogger("stopes.shas_segment_audio")


@dataclass
class SHASSegmentAudioConfig:
    # no_lid indicating that segmentation is run before LID
    lang: str = "no_lid"
    shards: tp.Any = MISSING  # absolute path to file to be segmented
    max_duration_in_seconds: tp.Optional[int] = None
    output_dir: Path = MISSING
    # path to pretrained SHAS (can be downloaded from https://github.com/mt-upc/SHAS)
    model: Path = MISSING
    # batch size (# examples) fed to the audio-frame-classifier
    inference_batch_size: int = 20
    # audio in seconds fed to the segmenter during inference
    inference_segment_length: int = 20
    # how many times to apply inference on different fixed-length segmentations
    inference_times: int = 4
    # split until all segments are below this value
    max_segment_length: int = 10
    # split only if the resulting two segments > this value
    min_segment_length: int = 2
    # after each split by the algorithm, the resulting segments are trimmed to
    # the first and last points that corresponds to a probability above this value
    dac_threshold: float = 0.5


# TODO(@hadyelsahar): create a Segmenter class to abstract over Shas/Vad
class SHASSegmentAudioModule(StopesModule):
    def __init__(self, config: SHASSegmentAudioConfig = SHASSegmentAudioConfig()):
        super().__init__(config, SHASSegmentAudioConfig)
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shas = None
        self.shards = self.config.shards
        if isinstance(self.shards, str):
            # it's a glob instead of a list of files
            self.shards = [Path(f) for f in glob.glob(self.shards)]

        assert all(f.exists() for f in self.shards)

    def requirements(self) -> Requirements:
        return Requirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=1,
            cpus_per_task=1,
            timeout_min=180,
        )

    def name(self) -> str:
        return f"shas_segment_audio.{len(self.shards)}"

    def array(self) -> tp.List[tp.List[Path]]:
        return utils.make_duration_batches(
            self.shards, self.config.max_duration_in_seconds
        )

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        if isinstance(iteration_value, list):
            input_files = iteration_value
        else:
            input_files = [iteration_value]

        self.shas = load_model(self.config)

        digest = xxhash.xxh3_64_intdigest(" ".join([str(a) for a in input_files]))
        filename = f"shas_segments.{iteration_index:05d}.{self.config.lang}.{digest}.gz"

        out_filename = self.output_dir / filename

        torchaudio.set_audio_backend("sox_io")

        with utils.open(out_filename, mode="wt", encoding="utf-8") as out_file:
            nb_lines = 0

            for input_file in input_files:
                try:
                    for start, end, batch_no in segment_file(self.shas, input_file):
                        out_file.write(f"{input_file} {start} {end} {batch_no}\n")
                        nb_lines += 1
                except Exception as e:
                    logger.error(f"Error processing {input_file} -- Error {e}")

            logger.info(f"created {out_file.name}")

        return (out_file.name, nb_lines)


def load_model(config: SHASSegmentAudioConfig) -> tp.Any:
    return SHAS(
        inference_batch_size=config.inference_batch_size,
        inference_times=config.inference_times,
        max_segment_length=config.max_segment_length,
        min_segment_length=config.min_segment_length,
        dac_threshold=config.dac_threshold,
        path_to_checkpoint=config.model,
    )


def segment_file(
    shas: tp.Any, input_file: Path
) -> tp.Iterator[tp.Tuple[int, int, int]]:
    batch_sampler, dataset = shas.build_segments_dataset(str(input_file))

    for batch_no, batch_indices in enumerate(batch_sampler):
        for ind in batch_indices:
            start, end = dataset.timestamps[ind]
            yield (start, end, batch_no)
