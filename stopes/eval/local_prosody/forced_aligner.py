# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from abc import ABC, abstractmethod

import torch

from stopes.eval.local_prosody.utterance import Utterance


class BaseSpeechForceAligner(ABC):
    """
    A generic speech-text force aligner system.
    """

    @abstractmethod
    def process_utterance(
        self,
        waveform: torch.Tensor,
        text: tp.Optional[str] = None,
        words: tp.Optional[tp.List[str]] = None,
        **kwargs,
    ) -> Utterance:
        """
        Convert a waveform to a prosodically annotated Utterance,
        potentially using a fixed text or list of words as a forced transcription.
        """
