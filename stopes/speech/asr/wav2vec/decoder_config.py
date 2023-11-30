# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING


@dataclass
class DecoderConfig:
    type: str = field(
        default="viterbi",
        metadata={
            "help": "The type of decoder to use (Permitted values: 'viterbi', 'kenlm', 'fairseqlm')"
        },
    )


@dataclass
class FlashlightDecoderConfig(DecoderConfig):
    nbest: int = field(
        default=1,
        metadata={"help": "Number of decodings to return"},
    )
    unitlm: bool = field(
        default=False,
        metadata={"help": "If set, use unit language model"},
    )
    lmpath: str = field(
        default=MISSING,
        metadata={"help": "Language model for KenLM decoder"},
    )
    lexicon: Optional[str] = field(
        default=None,
        metadata={"help": "Lexicon for Flashlight decoder"},
    )
    beam: int = field(
        default=50,
        metadata={"help": "Number of beams to use for decoding"},
    )
    beamthreshold: float = field(
        default=50.0,
        metadata={"help": "Threshold for beam search decoding"},
    )
    beamsizetoken: Optional[int] = field(
        default=None, metadata={"help": "Beam size to use"}
    )
    wordscore: float = field(
        default=-1,
        metadata={"help": "Word score for KenLM decoder"},
    )
    unkweight: float = field(
        default=-math.inf,
        metadata={"help": "Unknown weight for KenLM decoder"},
    )
    silweight: float = field(
        default=0,
        metadata={"help": "Silence weight for KenLM decoder"},
    )
    lmweight: float = field(
        default=2,
        metadata={"help": "Weight for LM while interpolating score"},
    )
