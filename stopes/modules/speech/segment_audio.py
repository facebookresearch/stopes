# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from dataclasses import dataclass
from importlib.machinery import SourceFileLoader
from pathlib import Path

# TODO: This code depends on fairseq branch `xlsr_laser_m2c2`
# at commit c4310cc97150d8255226e600d5fe0c57ece1e345
import fairseq
import torchaudio
import xxhash
from omegaconf import MISSING

from stopes.core import utils
from stopes.core.stopes_module import Requirements, StopesModule

logger = logging.getLogger("segment_audio")


def make_duration_batches(files: tp.List[Path], max_duration: int):
    batch_duration = 0
    batch = []
    for file in files:
        duration = utils.audio_duration(file)
        if batch_duration + duration > max_duration:
            if batch:
                yield batch
            batch = []
            batch_duration = 0
        batch.append(file)
        batch_duration += duration
    if batch:
        yield batch


@dataclass
class SegmentAudioConfig:
    shards: tp.List[Path] = MISSING
    max_duration_in_seconds: tp.Optional[int] = None
    output_dir: Path = MISSING
    vad_model: Path = MISSING
    hard_limit_min_length: float = 1.0


class SegmentAudioModule(StopesModule):
    def __init__(self, config: SegmentAudioConfig = SegmentAudioConfig()):
        super().__init__(config, SegmentAudioConfig)
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vad = None

        self._check_files_exist(self.config.shards)

    def _check_files_exist(self, files: tp.List[Path]):
        assert all(f.exists() for f in files)

    def requirements(self):
        return Requirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=1,
            cpus_per_task=1,
            timeout_min=180,
        )

    def array(self):
        if self.config.max_duration_in_seconds is None:
            return self.config.shards

        batched_shards = list(
            make_duration_batches(
                self.config.shards, self.config.max_duration_in_seconds
            )
        )

        return batched_shards

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        if isinstance(iteration_value, list):
            input_files = iteration_value
        else:
            input_files = [iteration_value]

        # it has to be done in a lazy way, because otherwise
        # submitit can't serialize the class
        if self.vad is None:
            examples = SourceFileLoader(
                "examples",
                fairseq.__path__[0] + "/../examples/__init__.py",
            ).load_module()

            from examples.laser.laser_src.vad import VAD

            self.vad = VAD(
                model_path=self.config.vad_model,
                hard_limit_min_length=self.config.hard_limit_min_length,
            )

        digest = xxhash.xxh3_64_intdigest(" ".join([str(a) for a in input_files]))
        filename = f"segment_audio.{iteration_index:05d}.{digest}.gz"
        out_filename = self.output_dir / filename

        torchaudio.set_audio_backend("sox_io")

        with utils.open(out_filename, mode="wt", encoding="utf-8") as out_file:
            nb_lines = 0

            for input_file in input_files:
                try:
                    batch_sampler, dataset = self.vad.build_segments_dataset(
                        str(input_file)
                    )

                    for batch_no, batch_indices in enumerate(batch_sampler):
                        for ind in batch_indices:
                            start, end = dataset.timestamps[ind]
                            out_file.write(f"{input_file} {start} {end} {batch_no}\n")
                            nb_lines += 1
                except:
                    logger.error(f"Error processing {input_file}")

            logger.info(f"created {out_file.name}")

        return (out_file.name, nb_lines)
