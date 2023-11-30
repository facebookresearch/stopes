# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import hashlib
import logging
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torchaudio
import xxhash
from fairseq.data.audio.raw_audio_dataset import RawAudioDataset
from omegaconf import MISSING

from stopes.core import utils
from stopes.core.stopes_module import Requirements, StopesModule
from stopes.modules.speech.vad import SAMPLING_RATE, VAD

logger = logging.getLogger("vad_segment_audio")


class FromWavAudioDataset(RawAudioDataset):
    def __init__(self, wav_list, audio_path, timestamps, sizes):
        super().__init__(
            sample_rate=SAMPLING_RATE,
            shuffle=False,
            pad=True,
            normalize=True,
            compute_mask_indices=False,
        )
        self.wav_list = wav_list
        self.audio_path = audio_path
        self.timestamps = timestamps
        self.sizes = np.array(sizes)
        self.hash_name = hashlib.sha1(audio_path.encode("utf-8")).hexdigest()[:30]

    def __getitem__(self, index):
        feats = self.postprocess(self.wav_list[index], SAMPLING_RATE)
        return {"id": index, "source": feats}

    def get_batch(self, indices):
        return self.collater([self[index] for index in indices])["net_input"]

    def get_metadata(self, index):
        return (
            self.audio_path
            + "|"
            + str(int(1000 * self.timestamps[index][0] / SAMPLING_RATE))
            + "|"
            + str(int(1000 * self.timestamps[index][1] / SAMPLING_RATE))
        )


class VADSegmenter(VAD):
    """An extension of VAD that outputs a Fairseq dataset as a result of segmentation"""

    def build_segments_dataset(self, audio_path, max_tokens=1280000):
        wav = self.read_audio(audio_path)
        segments, timestamps, sizes = self.get_segmentations(wav)
        dataset = FromWavAudioDataset(segments, audio_path, timestamps, sizes)
        indices = dataset.ordered_indices()
        batch_sampler = dataset.batch_by_size(
            indices,
            max_tokens=max_tokens,
            max_sentences=None,
            required_batch_size_multiple=1,
        )
        return batch_sampler, dataset


@dataclass
class VADSegmentAudioConfig:
    # no_lid indicating that segmentation is run before LID
    lang: str = "no_lid"
    shards: tp.Any = MISSING
    max_duration_in_seconds: tp.Optional[int] = None
    output_dir: Path = MISSING
    model: Path = MISSING
    hard_limit_min_length: float = 1.0


class VADSegmentAudioModule(StopesModule):
    def __init__(self, config: VADSegmentAudioConfig = VADSegmentAudioConfig()):
        super().__init__(config, VADSegmentAudioConfig)
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vad = None
        self.shards = self.config.shards
        if isinstance(self.shards, str):
            # it's a glob instead of a list of files
            self.shards = [Path(f) for f in glob.glob(self.shards)]
        else:
            self.shards = [Path(s) for s in self.shards]

        self._check_files_exist(self.shards)

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

    def name(self):
        return f"vad_segment_audio.{self.config.lang}.{len(self.shards)}"

    def array(self):
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

        # it has to be done in a lazy way, because otherwise
        # submitit can't serialize the class
        if self.vad is None:
            self.vad = load_model(self.config)

        digest = xxhash.xxh3_64_intdigest(" ".join([str(a) for a in input_files]))
        filename = f"vad_segments.{iteration_index:05d}.{self.config.lang}.{digest}.gz"
        out_filename = self.output_dir / filename

        torchaudio.set_audio_backend("sox_io")

        with utils.open(out_filename, mode="wt") as out_file:
            nb_lines = 0

            for input_file in input_files:
                try:
                    for start, end, batch_no in segment_file(self.vad, input_file):
                        out_file.write(f"{input_file} {start} {end} {batch_no}\n")
                        nb_lines += 1
                except Exception as e:
                    logger.error(f"Error processing {input_file} -- Error {e}")

            logger.info(f"created {out_file.name}")

        return (out_file.name, nb_lines)


def load_model(config: VADSegmentAudioConfig) -> tp.Any:
    return VADSegmenter(
        model_path=config.model,
        hard_limit_min_length=config.hard_limit_min_length,
    )


def segment_file(vad: tp.Any, input_file: Path) -> tp.Iterator[tp.Tuple[int, int, int]]:
    torchaudio.set_audio_backend("sox_io")
    batch_sampler, dataset = vad.build_segments_dataset(str(input_file))
    for batch_no, batch_indices in enumerate(batch_sampler):
        for ind in batch_indices:
            start, end = dataset.timestamps[ind]
            yield (start, end, batch_no)
