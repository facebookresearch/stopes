# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Collection of APIs that need access to the stopes model hub (internal or public)

import importlib
import typing as tp
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from stopes.core import utils as stopes_utils

if tp.TYPE_CHECKING:
    from stopes.eval.local_prosody.forced_aligner import BaseSpeechForceAligner
    from stopes.speech.tokenizers import SpeechTokenizer
    from stopes.speech.tts import TTS, SupportedTTSModel


# FAIR only: Load internal configurations for experiments at FAIR infrastructure
if importlib.util.find_spec("stopes.fb_config"):
    from stopes.fb_config import get_fb_config_and_hub

    FILE_CONFIG_STORE, HUB = get_fb_config_and_hub()
    from stopes.fb_hub import *
else:
    FILE_CONFIG_STORE = ""
    HUB = ""


# tokenizer configs for working models delivered as part of stopes
GIT_CONFIG_STORE = str((Path(__file__).parent / "pipelines/bitext/conf").resolve())


def hub_root():
    return str(Path(HUB).resolve())


OmegaConf.register_new_resolver("hub_root", hub_root, replace=True)


def load_config(model_name_or_path: tp.Union[str, Path], namespace: str) -> DictConfig:
    """
    read configs from different places in order:

    1. local file system
    2. git tokenizer config
    3. public config store

    Args:
        model_name_or_path (str, optional): name of the config file to be loaded
    Returns:
        the dictionary where all fields are filled and resolved

    Raises:
        ValueError if the load fails
    """
    if Path(model_name_or_path).exists():
        return tp.cast(DictConfig, OmegaConf.load(model_name_or_path))

    git_config_path: Path = (
        Path(GIT_CONFIG_STORE) / namespace / (str(model_name_or_path) + ".yaml")
    )
    if git_config_path.exists():
        return tp.cast(DictConfig, OmegaConf.load(git_config_path))

    public_config_path: Path = (
        Path(FILE_CONFIG_STORE) / namespace / (str(model_name_or_path) + ".yaml")
    )
    if public_config_path.exists():
        return tp.cast(DictConfig, OmegaConf.load(public_config_path))

    raise ValueError(f"Cannot load model from the config {model_name_or_path}")


def speech_tokenizer(
    pretrained_model_name_or_path: tp.Union[str, Path, None] = None,
    **kwargs: tp.Any,
) -> "SpeechTokenizer":
    """
    Instantiate a SpeechTokenizer to bundle a speech encoder, unit
    conversion and vocoder together. This helps handle audio encoding
    and decoding via simplified APIs
    """
    from stopes.speech.tokenizers import SpeechTokenizer

    if pretrained_model_name_or_path is None:
        config = OmegaConf.create()
    else:
        config = load_config(
            pretrained_model_name_or_path, namespace="speech_tokenizer"
        )
    return SpeechTokenizer.build(config, **kwargs)


def forced_aligner(model_name_or_path: tp.Union[str, Path]) -> "BaseSpeechForceAligner":
    """
    Instantiate a BaseSpeechForceAligner based on one of the following:
    - an ASR model (currently only Wav2Vec2forCTC are supported)
    - a speech synthesis model (i.e. UnitY2 forced aligner)
    with a common API
    """
    config = load_config(
        model_name_or_path=model_name_or_path, namespace="forced_alignment/"
    )
    if config.aligner_type == "Wav2Vec2":
        from stopes.eval.local_prosody.ctc_forced_aligner import Wav2Vec2ForcedAligner

        kwargs = dict(OmegaConf.to_container(config.config, resolve=True))  # type: ignore
        return Wav2Vec2ForcedAligner(**kwargs)
    elif config.aligner_type == "UnitY2":
        from stopes.eval.local_prosody.unity2_forced_aligner_f1 import (
            UnitY2ForcedAligner,
            UnitY2ForcedAlignerConfig,
        )

        typed_config_u2f1: UnitY2ForcedAlignerConfig = stopes_utils.promote_config(
            config.config, UnitY2ForcedAlignerConfig
        )
        return UnitY2ForcedAligner(typed_config_u2f1)
    elif config.aligner_type == "UnitY2F2":
        from stopes.eval.local_prosody.unity2_forced_aligner_f2 import (
            UnitY2F2ForcedAligner,
            UnitY2F2ForcedAlignerConfig,
        )

        typed_config_u2f2: UnitY2F2ForcedAlignerConfig = stopes_utils.promote_config(
            config.config, UnitY2F2ForcedAlignerConfig
        )
        return UnitY2F2ForcedAligner(typed_config_u2f2)
    else:
        raise ValueError(
            f"aligner_type {config.aligner_type} for forced alignment is not supported, expected `Wav2Vec2`, `UnitY2`, or `UnitY2F2`"
        )
