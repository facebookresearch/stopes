# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import dataclasses as dc
import logging
import typing as tp
from abc import ABC, abstractmethod
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from stopes.core import utils as stopes_utils

Path_ = tp.Union[str, Path]
OptionalPath_ = tp.Union[str, Path, None]
Array_ = tp.Union[torch.Tensor, np.ndarray, tp.List]

# types of which objects can be copied to CUDA
Cudable = tp.Union[torch.nn.Module, torch.Tensor]
logger = logging.getLogger(__name__)


@dc.dataclass
class HifiGANVocoderConfig:
    """The arguments of a hifiGAN vocoder"""

    checkpoint: OptionalPath_ = None
    config_path: OptionalPath_ = None


@dc.dataclass
class SpeechEncoderConfig:
    checkpoint: OptionalPath_ = None
    feature_layer: int = -1  # -1 means last layer


@dc.dataclass
class UnitsConfig:
    checkpoint: OptionalPath_ = None


@dc.dataclass
class SpeechTokenizerConfig:
    speech_encoder: SpeechEncoderConfig = SpeechEncoderConfig()
    lang: str = "en"
    feature_layer: int = 35
    units: UnitsConfig = UnitsConfig()
    vocoder: tp.Any = dc.field(default=HifiGANVocoderConfig())

    max_frames_chunk: int = 16_000_00  # 100s
    km_size: int = 1000

    # Note: These two config values are not materialized
    # in pretrained .YAML file, they are runtime-specific
    gpu: bool = torch.cuda.is_available()
    fp16: bool = False


class SpeechTokenizer(ABC):
    """
    A generic speech tokenizer that wraps the three models for
    speech encoder, kmeans clustering and vocoder.

    Any customized speech tokenizer should inherit fromn this class
    and can re-implement the speech_encode(), and
    decode(). The function encode() and batch_encode() should not
    be re-implemented, unless for advanced use case.
    """

    @staticmethod
    def build(config: tp.Any, **kwargs) -> "SpeechTokenizer":
        if kwargs:
            config = OmegaConf.merge(config, kwargs)
        assert hasattr(
            config, "_target_"
        ), "You need to specify the module to create in the yaml file with _target_"
        target = config._target_
        OmegaConf.resolve(config)
        return hydra.utils.instantiate({"_target_": target}, config, _recursive_=False)  # type: ignore[no-any-return]

    def __init__(self, config: tp.Any, config_class: tp.Optional[type] = None):
        if dc.is_dataclass(config):
            config = OmegaConf.structured(config)
        if config_class is not None:
            self.config = stopes_utils.promote_config(config, config_class)  # type: ignore
        else:
            assert isinstance(config, DictConfig), (
                "SpeechTokenizer's configs must be either a dataclass or a omega.DictConfig."
                f" Received a {type(config)}"
            )
            self.config = config
        OmegaConf.resolve(self.config)
        OmegaConf.set_readonly(self.config, True)

    def __post_init__(self, *inputs, **kwargs):
        self.validate_model_config(*inputs, **kwargs)

    @tp.overload
    def _sanitize(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @tp.overload
    def _sanitize(self, x: torch.nn.Module) -> torch.nn.Module:
        ...

    def _sanitize(self, x: Cudable) -> Cudable:
        if hasattr(self.config, "gpu") and self.config.gpu:
            x = x.cuda()
            if hasattr(self.config, "fp16") and self.config.fp16:
                x = x.half()
        return x

    def validate_model_config(self, *inputs, **kwargs):
        """
        Validate that the combination of configs are valid
        """
        pass

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """convert audio features to vectors"""
        return x

    def to_unit(self, x: torch.Tensor) -> torch.Tensor:
        """convert wave vectors to discrete units (e.g by using kmeans)"""
        return x

    @abstractmethod
    def encode(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        """Convert one audio sample into a stream of discrete units.

        Internally, this function calls extract_features() and to_unit()
        in sequence."""

    @abstractmethod
    def decode(self, units: torch.Tensor, **kwargs) -> torch.Tensor:
        """decode the one or a batch of stream of dicrete units back into wave form"""
