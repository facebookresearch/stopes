# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This package contains module for running the fairseq wav2vec decoders
# for ASR. We must port this from
# fairseq/examples/speech_recognition/new/decoders and fix a few things:
# 1. Get rid of all relative imports in "examples"
# 2. Original fairseq "examples" will be located differently ("examples" is
# oustide "fairseq" package if fairseq is installed
# in editable mode, but in public PyPI, it is put under "fairseq")
# 3. The package name "examples" gets conflict with other library (e.g. SONAR)
# when those libraries are also installed from source in editable mode


from fairseq.data.dictionary import Dictionary  # type: ignore

from .base_decoder import BaseDecoder
from .decoder_config import DecoderConfig, FlashlightDecoderConfig


def decoder(
    cfg: DecoderConfig,
    tgt_dict: Dictionary,
    gpu: bool = True,
) -> BaseDecoder:

    if cfg.type == "viterbi":
        from .viterbi_decoder import ViterbiDecoder

        return ViterbiDecoder(tgt_dict)
    if cfg.type == "kenlm":
        assert isinstance(
            cfg, FlashlightDecoderConfig
        ), "kenlm needs extra configs (see FlashlightDecoderConfig)"
        from .flashlight_decoder import KenLMDecoder

        return KenLMDecoder(cfg, tgt_dict)
    if cfg.type == "fairseqlm":
        assert isinstance(
            cfg, FlashlightDecoderConfig
        ), "fairseqLM needs extra configs (see FlashlightDecoderConfig)"
        from .flashlight_decoder import FairseqLMDecoder

        return FairseqLMDecoder(cfg, tgt_dict, gpu=gpu)
    raise NotImplementedError(f"Invalid decoder name: {cfg.type}")
