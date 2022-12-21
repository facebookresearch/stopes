# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import typing as tp
from importlib.machinery import SourceFileLoader

# TODO: This code depends on fairseq branch `xlsr_laser_m2c2`
# at commit c4310cc97150d8255226e600d5fe0c57ece1e345
import fairseq
import numpy as np
import torchaudio

from stopes.modules.preprocess.encode_to_npy import EncodeToNPY
from stopes.utils.mining_utils import extract_shard_id


class MiningSpeechEncoder(EncodeToNPY):
    def __init__(
        self,
        _name: str,
        encoder_model: str,
        # TODO: refactor speech vs text encoders, spm_model is unused here
        spm_model: str,
        outfile_prefix: str,
        input_file: str,
        input_file_idx: int,
        output_dir: str,
        outfile_postfix: str = "",
        spm_vocab: str = None,
        max_sentences: tp.Optional[int] = None,
        max_tokens: int = 12_000,
        stable_sort: bool = False,
        normalize: bool = False,
        fp16: bool = False,
        cpu: bool = False,
        fp16_model: bool = False,
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
        self.fp16_model = fp16_model
        examples = SourceFileLoader(
            "examples",
            fairseq.__path__[0] + "/../examples/__init__.py",
        ).load_module()

        from examples.laser.laser_src.laser_speech import (
            LaserSpeechEncoder as FairseqLaserEncoder,
        )

        self.encoder = FairseqLaserEncoder(encoder_model)
        if self.fp16_model:
            self.encoder.encoder.half()

    def name_output_file(self) -> str:
        shard_idx = extract_shard_id(self.input_file, default=self.input_file_idx)

        return os.path.abspath(
            os.path.join(
                self.output_dir,
                f"{self.outfile_prefix}.{shard_idx:05d}.{self.outfile_postfix}",
            )
        )

    def encode_to_np(
        self, lines_with_number: tp.Iterator[tp.Tuple[int, str]]
    ) -> np.ndarray:
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
        that computes the segments (VAD class).
        """
        last_audio_file = None
        last_batch_no = None
        wav_audio_samples = None
        batch_timestamps = []
        results = []

        examples = SourceFileLoader(
            "examples",
            fairseq.__path__[0] + "/../examples/__init__.py",
        ).load_module()
        from examples.laser.laser_src import vad

        def handle_batch_buffer():
            nonlocal batch_timestamps, results

            if len(batch_timestamps):
                wav_list = [
                    wav_audio_samples[start:end] for start, end in batch_timestamps
                ]
                sizes = [end - start for start, end in batch_timestamps]
                minidataset = vad.FromWavAudioDataset(
                    wav_list, "_dummy", batch_timestamps, sizes
                )
                batch = minidataset.get_batch([i for i in range(len(wav_list))])
                if self.fp16_model:
                    for k in batch.keys():
                        batch[k] = batch[k].half()
                embeddings = self.encoder.encode_batch(**batch)
                embeddings = embeddings.detach().cpu().numpy()
                results.append(embeddings)
                batch_timestamps = []

        torchaudio.set_audio_backend("sox_io")

        for _, line in lines_with_number:
            cur_input_file, ts_start, ts_end, cur_batch_no = line.split(" ")
            ts_start = int(ts_start)
            ts_end = int(ts_end)
            cur_batch_no = int(cur_batch_no)
            if cur_input_file != last_audio_file or last_batch_no != cur_batch_no:
                handle_batch_buffer()
            if cur_input_file != last_audio_file:
                wav_audio_samples = vad.read_audio(cur_input_file)
            batch_timestamps.append((ts_start, ts_end))
            last_audio_file = cur_input_file
            last_batch_no = cur_batch_no

        handle_batch_buffer()

        results = np.vstack(results)

        assert results.shape[0] == len(lines_with_number)
        return results
