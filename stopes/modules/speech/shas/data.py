# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This source code is adapted with minimal changes
# from https://github.com/mt-upc/SHAS

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import stopes.modules.speech.utils as speech_utils

INPUT_SAMPLE_RATE = 16_000
TARGET_SAMPLE_RATE = 49.95  # 50 (16000/320) wasnt exactly correct


@dataclass
class Segment:
    start: float
    end: float
    probs: np.array
    decimal: int = 4  # round time to the nearest decimal

    @property
    def duration(self):
        return float(round((self.end - self.start) / TARGET_SAMPLE_RATE, self.decimal))

    @property
    def offset(self):
        return float(round(self.start / TARGET_SAMPLE_RATE, self.decimal))

    @property
    def offset_plus_duration(self):
        return round(self.offset + self.duration, self.decimal)


class FixedSegmentationDatasetNoTarget(Dataset):
    def __init__(
        self,
        path_to_audio_file: str,
        segment_length: int = 20,
        inference_times: int = 1,
    ) -> None:
        """This datastructure is internal to SHAS segmenter it takes audio file
        reads it and chunks it to chunks of len `segment_length`
        The idea is that those equal chunks will be later splitted into smaller
        ones using the SHAS frame classifier and DAC algorithm.
        This process is repeated `inference times` to reduce variance
        and probabilities of frames are averaged.

        Args:
            path_to_audio_file (str): path to a wav file to be segmented
            segment_length (int, optional): fixed size segment length in secs. Defaults to 20.
            inference_times (int, optional): number of times to repeat inference. Defaults to 1.
        """
        super().__init__()
        self.input_sr = INPUT_SAMPLE_RATE
        self.target_sr = TARGET_SAMPLE_RATE
        self.in_trg_ratio = self.input_sr / self.target_sr
        self.trg_in_ratio = 1 / self.in_trg_ratio

        self.segment_length_inframes = self.secs_to_inframes(segment_length)
        self.inference_times = inference_times

        self.path_to_audio_file = path_to_audio_file
        self.wavform = speech_utils.read_audio(
            path_to_audio_file, sampling_rate=self.input_sr
        )

        self.duration_inframes = self.wavform.shape[0]
        self.duration_outframes = self._inframes_to_outframes(self.duration_inframes)

    def _inframes_to_outframes(self, x):
        # from input space to output space
        return np.round(x * self.trg_in_ratio).astype(int)

    def secs_to_inframes(self, x):
        # from seconds to input space
        return np.round(x * self.input_sr).astype(int)

    def fixed_length_segmentation(self, i: int) -> None:
        """
        Generates a fixed-length segmentation of a wav
        with "i" controlling the begining of the segmentation
        so that different values of "i" produce different segmentations

        Args:
            talk_id (str): unique wav identifier
            i (int): indicates the current inference time
                and is used to produce a different fixed-length segmentation
                minimum allowed is 0 and maximum allowed is inference_times - 1
        """

        start = round(self.segment_length_inframes / self.inference_times * i)
        if start > self.duration_inframes:
            start = 0
        segmentation = np.arange(
            start, self.duration_inframes, self.segment_length_inframes
        ).astype(int)
        if segmentation[0] != 0:
            segmentation = np.insert(segmentation, 0, 0)
        if segmentation[-1] != self.duration_inframes:
            if self.duration_inframes - segmentation[-1] < self.secs_to_inframes(2):
                segmentation[-1] = self.duration_inframes
            else:
                segmentation = np.append(segmentation, self.duration_inframes)

        self.starts = segmentation[:-1]
        self.ends = segmentation[1:]

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, int, int]:
        """
        Loads the data for this fixed-length segment

        Args:
            index (int): index of the segment in the fixed length segmentation

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor, int, int]:
                0: waveform of the segment (input space)
                1: None for consistency with datasets that have targets
                2: starting frame of the segment (output space)
                3: ending frame of the segment (output space)
        """

        start = self._inframes_to_outframes(self.starts[index] + 1e-6)
        end = self._inframes_to_outframes(self.ends[index] + 1e-6)

        return self.wavform[self.starts[index] : self.ends[index]], None, start, end


def segm_collate_fn(
    batch: list,
) -> Tuple[
    torch.FloatTensor,
    torch.FloatTensor,
    torch.LongTensor,
    torch.BoolTensor,
    List[bool],
    List[int],
    List[int],
]:
    """
    (inference) collate function for the dataloader of the SegmentationDataset

    Args:
        batch (list): list of examples from SegmentationDataset

    Returns:
        Tuple[ torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.BoolTensor, list[bool], list[int], list[int], ]:
            0: 2D tensor, padded and normalized waveforms for each random segment
            1: 2D tensor, binary padded targets for each random segment (output space)
            2: 2D tensor, binary mask for wav2vec 2.0 (input space)
            3: 2D tensor, binary mask for audio-frame-classifier (output space)
            4: a '0' indicates that the whole example is empty (torch.zeros)
            5: the start frames of the segments (output space)
            6: the end frames of the segments (output space)
    """

    included = [bool(example[0].sum()) for example in batch]
    starts = [example[2] for example in batch]
    ends = [example[3] for example in batch]

    # sequence lengths
    in_seq_len = [len(example[0]) for example in batch]
    out_seq_len = [end - start for start, end in zip(starts, ends)]
    bs = len(in_seq_len)

    # pad and concat
    audio = torch.cat(
        [
            F.pad(example[0], (0, max(in_seq_len) - len(example[0]))).unsqueeze(0)
            for example in batch
        ]
    )

    # check if the batch contains also targets
    if batch[0][1] is not None:
        target = torch.cat(
            [
                F.pad(example[1], (0, max(out_seq_len) - len(example[1]))).unsqueeze(0)
                for example in batch
            ]
        )
    else:
        target = None

    # normalize input
    # only for inputs that have non-zero elements
    included_ = torch.tensor(included).bool()
    audio[included_] = (
        audio[included_] - torch.mean(audio[included_], dim=1, keepdim=True)
    ) / torch.std(audio[included_], dim=1, keepdim=True)

    # get masks
    in_mask = torch.ones(audio.shape, dtype=torch.long)
    out_mask = torch.ones([bs, max(out_seq_len)], dtype=torch.bool)
    for i, in_sl, out_sl in zip(range(bs), in_seq_len, out_seq_len):
        in_mask[i, in_sl:] = 0
        out_mask[i, out_sl:] = 0

    return (audio, target, in_mask, out_mask, included, starts, ends)
