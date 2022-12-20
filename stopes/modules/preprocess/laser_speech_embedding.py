# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import time
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from fairseq.data import FileAudioDataset
from omegaconf import MISSING


@dataclass
class LaserEmbeddingConfig:
    launcher: tp.Any
    max_tokens: int
    checkpoint_dir: Path = MISSING
    data_dir: Path = MISSING
    num_chunks: int = MISSING
    lang_dirs: str = MISSING
    out_dir: Path = MISSING


class LaserSpeechEmbedding:
    def __init__(self, checkpoint_dir, checkpoint_file, max_tokens, logger):
        self.max_tokens = max_tokens
        self.logger = logger
        self.logger.info(f"Loading {checkpoint_file} speechLASER.")
        start_time = time.time()
        try:
            from fairseq.models.wav2vec import Wav2VecLaser
        except ImportError:
            self.logger.error("Wav2VecLaser is not defined in fairseq.models.wav2vec")

        self.encoder = Wav2VecLaser.from_pretrained(
            checkpoint_dir, checkpoint_file
        ).models[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Moved model to {self.device}")
        self.encoder.to(self.device)
        self.encoder.eval()
        self.logger.info(f"{time.time() - start_time} seconds to load model.")

    def _encode_batch(self, source, padding_mask):
        start_time = time.time()
        with torch.no_grad():
            sample = {
                "padding_mask": padding_mask.to(self.device),
                "source": source.to(self.device),
            }
            embeddings = self.encoder(**sample)
        self.logger.info(f"{time.time() - start_time} seconds to encode batch.")
        return embeddings

    def _load_dataset_from_file(self, file_path, max_tokens):
        dataset = FileAudioDataset(
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

    def encode_file(self, infile, outfile):
        batch_sampler, dataset = self._load_dataset_from_file(infile, self.max_tokens)
        ids = []
        embeddings = None
        for i in range(len(batch_sampler)):
            self.logger.info(f"{i+1}/{len(batch_sampler)}")
            curr_ids = list(batch_sampler[i])[::-1]
            net_input = dataset.collater([dataset[index] for index in curr_ids])[
                "net_input"
            ]
            curr_embeddings = self._encode_batch(**net_input)
            ids = ids + curr_ids
            if embeddings is None:
                embeddings = curr_embeddings.cpu()
            else:
                embeddings = torch.cat((embeddings, curr_embeddings.cpu()))
        embeddings = embeddings[np.argsort(ids)].numpy()
        with open(outfile, "wb") as fp:
            embeddings.astype("float16").tofile(fp)
