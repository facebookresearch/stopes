# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import typing as tp
from pathlib import Path

import hydra.utils as hydra_utils
import torch

from stopes.core import utils as core_utils
from stopes.modules.preprocess.encode_to_npy import EncodeToNPY
from stopes.modules.speech import utils as speech_utils
from stopes.modules.speech.speech_units import parallel_audio_read
from stopes.utils.embedding_utils import NpMatrix
from stopes.utils.mining_utils import extract_shard_id


class MiningSpeechEncoder(EncodeToNPY):
    def __init__(
        self,
        _name: str,
        encoder_model: str,
        # Keeping for operability speech vs text encoders
        spm_model: str,
        outfile_prefix: str,
        input_file: str,
        output_dir: Path,
        input_file_idx: int = 0,
        outfile_postfix: str = "",
        spm_vocab: tp.Optional[str] = None,
        max_tokens: int = 1_280_000,
        normalize: bool = False,
        fp16: bool = False,
        gpu: bool = False,
        num_processes: int = 1,
        input_column_offset: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            outfile_prefix=outfile_prefix,
            outfile_postfix=outfile_postfix,
            input_file=input_file,
            input_file_idx=input_file_idx,
            output_dir=output_dir,
            normalize=normalize,
            fp16=fp16,
        )
        self.logger = logging.getLogger("stopes.MiningSpeechEncoder")
        self.encoder_model = encoder_model
        self.max_tokens = max_tokens

        # Lazily construct the encoder so the lib fairseq / fairseq2
        # is only loaded right at the beginning of the inference
        self._encoder = None
        self.input_column_offset = input_column_offset
        self.gpu = gpu
        self.num_processes = num_processes
        self.kwargs = kwargs
        self._read_audio_func = speech_utils.read_audio
        if "read_audio_func" in self.kwargs:
            r_audio_func = self.kwargs.get("read_audio_func")
            if callable(r_audio_func):
                self._read_audio_func = r_audio_func
            elif isinstance(r_audio_func, str):
                self._read_audio_func = hydra_utils.get_method(r_audio_func)

    @property
    def encoder(self):
        if self._encoder is None:
            from stopes.modules.preprocess.wav2vec_laser_speech_encoder import (
                LaserSpeechEncoder,
            )

            checkpoint = Path(self.encoder_model)
            self.checkpoint = checkpoint
            self._encoder = LaserSpeechEncoder(
                checkpoint.parent, checkpoint.name, self.logger
            )  # type: ignore[assignment]
        assert (
            self._encoder is not None
        ), f"Cannot load LaserSpeechEncoder from {self.encoder_model}"
        return self._encoder

    def name_output_file(self) -> str:
        shard_idx = extract_shard_id(self.input_file, default=self.input_file_idx)

        # Avoid file name such as "XYZ."
        filename = f"{self.outfile_prefix}.{shard_idx:05d}"
        if self.outfile_postfix and len(self.outfile_postfix) > 0:
            filename = f"{filename}.{self.outfile_postfix}"

        return os.path.abspath(os.path.join(self.output_dir, filename))

    def encode_to_np(
        self, lines_with_number: tp.Iterator[tp.Tuple[int, str]]
    ) -> NpMatrix:
        """
        Reads a file where each line is a text containing information on
        the audio file and segment timestamps, and returns an embedding
        numpy array.

        Like the sentence encoder, each line of the input file is a
        "speech sentence", but instead of having actual wav samples, it
        contains information required to build these wav samples. Each line has
        the following format: <source_file> <start> <end> <batch_no>

        a.mp3 100 29432 0
        a.mp3 100 39834 0
        a.mp3 40594 58382 0
        a.mp3 60383 87443 0
        a.mp3 89430 95000 1
        a.mp3 95384 98524 1
        ...

        <batch_no> is an extra information that is already provided by the code
        that computes the segments (VAD class). Here, we ignore this information
        since we compute the batches.

        We also allow the input file to be a tsv, usually the output of the lid
        module.

        score <tab> a.mp3 40594 58382 0 <tab> eng
        score <tab> a.mp3 89430 95000 1 <tab> eng
        ...

        In this case, we read the second column to fetch the audio signal
        information.
        """
        from stopes.modules.speech.wav2vec.utils import WavesDataset

        audio_signals = []
        with core_utils.measure("parallel_audio_read", self.logger):
            for line, audio in parallel_audio_read(
                (a[1] for a in lines_with_number),
                column_offset=self.input_column_offset,
                gpu=self.gpu,
                fp16=self.fp16,
                num_process=self.num_processes,
                read_audio_func=self._read_audio_func,
            ):
                audio_signals.append(audio)

        assert len(audio_signals) > 0, "Empty audio input"

        """Given a list of audios, compute their matrix of embeddings."""
        sizes = [signal.size for signal in audio_signals]
        dataset = WavesDataset(
            audio_signals, sizes, fbank_features=self.encoder.fbank_features
        )
        batch_sampler = dataset.batch_by_size(
            dataset.ordered_indices(),
            max_tokens=self.max_tokens,
            max_sentences=None,
            required_batch_size_multiple=1,
        )
        return self.encoder.encode_dataset(batch_sampler, dataset)


class Sonar2MiningSpeechEncoder(MiningSpeechEncoder):
    """
    The speech mining encoder that uses the Sonar speech encoder model.
    Both `fairseq2` and `sonar` are required.
    """

    SONAR_PAD_TOKEN = 0
    SAMPLING_RATE = 16000

    @MiningSpeechEncoder.encoder.getter  # type: ignore[attr-defined]
    def encoder(self):
        if self._encoder is None:
            from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline

            device = torch.device(
                "cuda" if self.gpu and torch.cuda.is_available() else "cpu"
            )

            # layer_norm does not work well with fp16 in CPU mode
            fbank_dtype = torch.float16 if (self.fp16 and self.gpu) else torch.float32

            self._encoder = SpeechToEmbeddingModelPipeline(
                encoder=self.encoder_model,
                device=device,
                fbank_dtype=fbank_dtype,
            )  # type: ignore[assignment]
        return self._encoder

    def encode_to_np(
        self, lines_with_number: tp.Iterator[tp.Tuple[int, str]]
    ) -> NpMatrix:
        audio_signals = []
        with core_utils.measure("parallel_audio_read", self.logger):
            for _, audio in parallel_audio_read(
                (a[1] for a in lines_with_number),
                column_offset=self.input_column_offset,
                gpu=self.gpu,
                fp16=self.fp16,
                num_process=self.num_processes,
                read_audio_func=self._read_audio_func,
            ):
                audio_tensor = torch.from_numpy(audio)
                if audio_tensor.ndim == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                audio_signals.append(audio_tensor)

        mini_batch_size = self.kwargs.get("mini_batch_size")
        if mini_batch_size is None:
            mini_batch_size = len(audio_signals)

        return (
            self.encoder.predict(
                audio_signals,
                batch_size=mini_batch_size,
                n_parallel=self.num_processes,
                pad_idx=self.SONAR_PAD_TOKEN,
            )
            .cpu()
            .numpy()
        )  # type: ignore
