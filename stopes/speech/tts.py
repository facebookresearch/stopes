# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List

import torch


class SupportedTTSModel(Enum):
    MMS = "mms"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


@dataclass
class TTSOutput:
    waveform_tensors: List[torch.Tensor]
    sampling_rate: int


class TTS(ABC):
    """
    A generic TTS class that converts the Text To Speech.
    """

    def __init__(self):
        pass

    @abstractmethod
    def generate(
        self,
        texts: List[str],
    ) -> TTSOutput:
        """Receive a list of texts(text segments), return a list of Wavform tensor."""
        ...
