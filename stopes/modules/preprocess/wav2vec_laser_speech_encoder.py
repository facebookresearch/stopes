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
from fairseq import data as fairseq_data
from omegaconf import MISSING

from stopes.core import utils
from stopes.modules.speech.wav2vec import utils as wav2vec_utils
from stopes.utils import embedding_utils

LASER_EMBEDDING_DIM = 1024
SONAR_PAD_TOKEN = 2


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
    def __init__(
        self, checkpoint_dir: Path, checkpoint_file: str, logger: logging.Logger
    ):
        self.logger = logger
        with utils.measure(f"Loading {checkpoint_file} speechLASER.", logger):
            (
                self.encoder,
                self.encoder_cfg,
                self.encoder_task,
            ) = wav2vec_utils.load_speech_encoder(checkpoint_dir, checkpoint_file)
            # our speech encoders are trained in fp16
            self.encoder = self.encoder.half()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Moved model to {self.device}")
            self.encoder.to(self.device)
            self.encoder.eval()

    @property
    def fbank_features(self) -> int:
        """Number of fbank features to feed to the encoder (0 if instead of fbank it expects a raw waveform).
        This parameter is defined based on the architecture of the underlying encoder."""
        return self.encoder_cfg.task.get("fbank_features", 0)

    def _encode_batch(self, source, padding_mask):
        with torch.no_grad():
            if (
                type(self.encoder).__name__ == "Wav2Vec2Seq2SeqModel"
            ):  # w2vbert with attention pooling
                # this is a seq2seq model, and pooling happens in the decoder,
                # so we need to run it for one step but then to ignore the decoder output
                prev_output_tokens = torch.LongTensor(
                    [[SONAR_PAD_TOKEN]] * source.shape[0]
                ).to(self.device)
                embeddings, _ = self.encoder(
                    padding_mask=padding_mask.to(self.device),
                    source=source.to(self.device),
                    prev_output_tokens=prev_output_tokens,
                )
                embeddings = embeddings.squeeze(1)
            else:
                embeddings = self.encoder(
                    padding_mask=padding_mask.to(self.device),
                    source=source.to(self.device),
                )
        return embeddings

    def encode_dataset(
        self,
        batch_sampler: tp.Iterable[tp.Sequence[int]],
        dataset: fairseq_data.audio.raw_audio_dataset.RawAudioDataset,
    ) -> embedding_utils.NpMatrix:
        ids: tp.List[int] = []
        embeddings = np.empty(
            shape=(len(dataset), LASER_EMBEDDING_DIM),
            dtype=np.float16,
        )
        start = 0
        for batch_id, curr_ids in enumerate(batch_sampler):
            with utils.measure(
                f"encoding batch {batch_id} {len(curr_ids)} samples.", self.logger
            ):
                batch = dataset.collater([dataset[i] for i in curr_ids])
                net_input = batch["net_input"]
                for k in net_input.keys():
                    net_input[k] = net_input[k].half()
                curr_embeddings = self._encode_batch(**net_input)
            ids.extend(curr_ids)
            end = start + len(curr_ids)
            embeddings[start:end, :] = curr_embeddings.cpu().numpy().astype(np.float16)
            start = end

        # Numpy type checking isn't smart enought to guess the shape of the returned value.
        return embeddings[np.argsort(ids)]  # type: ignore


class LaserFileAudioEncoder:
    def __init__(
        self, checkpoint_dir: Path, checkpoint_file: str, max_tokens: int, logger
    ):
        self.dataset_encoder = LaserSpeechEncoder(
            checkpoint_dir, checkpoint_file, logger
        )
        self.max_tokens = max_tokens

    def _load_dataset_from_file(
        self, file_path: Path, max_tokens: int
    ) -> tp.Tuple[
        tp.List[tp.Sequence[int]], fairseq_data.audio.raw_audio_dataset.RawAudioDataset
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

    def encode_file(self, in_manifest: Path, outfile: Path, fp16: bool = False) -> None:
        batch_sampler, dataset = self._load_dataset_from_file(
            in_manifest, self.max_tokens
        )
        emb = self.dataset_encoder.encode_dataset(batch_sampler, dataset)
        with embedding_utils.EmbeddingConcatenator(outfile, fp16) as writer:
            writer.append_embedding_from_array(emb)
