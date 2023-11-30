# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import functools
import logging
import typing as tp
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

PathOrString = tp.Union[str, Path]
OptionalPath = tp.Optional[PathOrString]
Array_ = tp.Union[torch.Tensor, np.ndarray, tp.List]

# types of which objects can be copied to CUDA
Cudable = tp.Union[torch.nn.Module, torch.Tensor]
logger = logging.getLogger(__name__)


class ASR(ABC):
    """
    A generic ASR (Automatic Speech Recognition) system that
    wraps different ASR models including Whisper and in-house
    fairseq speech-to-text model.

    Any customized ASR should inherit fromn this class
    and implement the transcribe() method.
    """

    def __init__(self):
        pass

    @tp.overload
    def move_to_device(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @tp.overload
    def move_to_device(self, x: torch.nn.Module) -> torch.nn.Module:
        ...

    def move_to_device(self, x: Cudable) -> Cudable:
        gpu = getattr(self, "gpu", False)
        fp16 = getattr(self, "fp16", False)

        # give a warning when doing quantiztion in CPU
        if not gpu and fp16:
            logger.warning(
                "FP16 enabled in CPU. This might lead to runtime"
                "error since not architecture support HalfTensor"
            )

        if gpu:
            x = x.cuda()
        if fp16:
            x = x.half()

        return x

    def validate_model_config(self, *inputs, **kwargs):
        """
        Validate that the combination of configs are valid
        """
        pass

    @abstractmethod
    def transcribe(self, x: torch.Tensor) -> str:
        """convert wave vectors to a string of the transcribed text"""


class WhisperASR(ASR):
    """ASR model that uses Whisper to transcribe speech"""

    def __init__(
        self, checkpoint: str, lang: str = "en", gpu: bool = True, fp16: bool = False
    ):
        self.lang = lang
        self.checkpoint = checkpoint
        self.gpu = gpu
        self.fp16 = fp16

    @functools.cached_property
    def model(self):

        import whisper

        device = "cuda" if self.gpu else "cpu"
        mdl = whisper.load_model(self.checkpoint, device=device)

        if self.fp16:
            mdl = mdl.half()

        return mdl

    @torch.no_grad()
    def transcribe(self, x: torch.Tensor) -> str:
        """Convert one audio sample into a text string.
        Calls model().
        Args:
           x: tensor of audio wave form (batch x channels x frame_len or channels x frame_len)
        """

        assert len(x.shape) >= 2

        x_device = self.move_to_device(x)  # type: torch.Tensor
        if len(x_device.shape) > 2:
            num_channels, channel_dim = x_device.shape[1], 1
        else:
            num_channels, channel_dim = x_device.shape[0], 0

        if num_channels > 1:
            x_device = x_device.mean(dim=channel_dim, keepdim=True)

        whisper_model = self.model()
        result = whisper_model.transcribe(
            audio=x_device[0], language=self.lang, task="transcribe"
        )

        return result["text"]
