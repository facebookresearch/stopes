# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from fairseq import data as fairseq_data
from omegaconf import MISSING

from stopes.core import utils
from stopes.utils import embedding_utils

LASER_EMBEDDING_DIM = 1024


@dataclass
class LaserEmbeddingConfig:
    launcher: tp.Any
    max_tokens: int
    checkpoint_dir: Path = MISSING
    data_dir: Path = MISSING
    num_chunks: int = MISSING
    lang_dirs: str = MISSING
    out_dir: Path = MISSING


class LaserSpeechEncoder:
    def __init__(self, checkpoint_dir: Path, checkpoint_file: str, logger):
        self.logger = logger
        with utils.measure(f"Loading {checkpoint_file} speechLASER.", self.logger):
            try:
                # This code depends on fairseq branch `ust`
                from fairseq.models.wav2vec import Wav2VecLaser
            except ImportError:
                self.logger.error(
                    "Wav2VecLaser is not defined in fairseq.models.wav2vec"
                )

            self.encoder = Wav2VecLaser.from_pretrained(
                checkpoint_dir, checkpoint_file
            ).models[0]
            # our speech encoders are trained in fp16
            self.encoder = self.encoder.half()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Moved model to {self.device}")
            self.encoder.to(self.device)
            self.encoder.eval()

    def _encode_batch(self, source, padding_mask):
        with torch.no_grad():
            embeddings = self.encoder(
                padding_mask=padding_mask.to(self.device),
                source=source.to(self.device),
            )
        return embeddings

    def encode_dataset(
        self,
        batch_sampler: tp.Iterable[np.ndarray],
        dataset: fairseq_data.audio.raw_audio_dataset.RawAudioDataset,
    ) -> tp.Optional[np.ndarray]:
        ids: tp.List[int] = []
        embeddings = np.empty(
            shape=(len(dataset), LASER_EMBEDDING_DIM),
            dtype=np.float16,
        )
        start = 0
        for i, curr_ids in enumerate(batch_sampler):
            with utils.measure(f"encoding batch {i}", self.logger):
                net_input = dataset.collater([dataset[index] for index in curr_ids])[
                    "net_input"
                ]
                for k in net_input.keys():
                    net_input[k] = net_input[k].half()
                curr_embeddings = self._encode_batch(**net_input)
            ids = ids + list(curr_ids)
            end = start + len(curr_ids)
            embeddings[start:end, :] = curr_embeddings.cpu().numpy().astype(np.float16)
            start = end

        return embeddings[np.argsort(ids)]


class LaserFileAudioEncoder:
    def __init__(
        self, checkpoint_dir: Path, checkpoint_file: str, max_tokens: int, logger
    ):
        self.dataset_encoder = LaserSpeechEncoder(
            checkpoint_dir, checkpoint_file, logger
        )
        self.max_tokens = max_tokens

    def _load_dataset_from_file(
        self, file_path, max_tokens
    ) -> tp.Tuple[
        tp.List[np.ndarray], fairseq_data.audio.raw_audio_dataset.RawAudioDataset
    ]:
        dataset = fairseq_data.FileAudioDataset(
            file_path,
            sample_rate=16000,
            pad=True,
            shuffle=False,
            normalize=True,
            max_sample_size=max_tokens,
        )
        indices = dataset.ordered_indices()
        batch_sampler = dataset.batch_by_size(
            indices,
            max_tokens=max_tokens,
            max_sentences=None,
            required_batch_size_multiple=1,
        )
        return batch_sampler, dataset

    def encode_file(self, in_manifest: Path, outfile: Path, fp16: bool = False):
        batch_sampler, dataset = self._load_dataset_from_file(
            in_manifest, self.max_tokens
        )
        emb = self.dataset_encoder.encode_dataset(batch_sampler, dataset)
        with embedding_utils.EmbeddingConcatenator(outfile, fp16) as writer:
            writer.append_embedding_from_array(emb)
