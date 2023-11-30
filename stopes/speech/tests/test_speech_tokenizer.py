# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import dataclasses
from pathlib import Path

import pytest
import torch

from stopes import hub
from stopes.speech.tokenizers import SpeechTokenizer


@dataclasses.dataclass
class MockConfig:
    mock_attr: str
    mock_checkpoint: str
    max_frames_chunk: int


class MockSpeechTokenizer(SpeechTokenizer):

    config: MockConfig

    def __init__(self, config):
        super().__init__(config, MockConfig)

    def __repr__(self):
        return "My mock tokenizer"

    def encode(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.zeros(2, 3)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def to_unit(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, units: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.ones(2, 3)


@pytest.mark.parametrize("config_file", ["config.yaml", "partial_config.yaml"])
def test_load_local_config(config_file):
    """test that the config file can be parsed and config object is properly built"""

    cfg = str((Path(__file__).parent / config_file).resolve())
    if config_file == "config.yaml":
        tokenizer = hub.speech_tokenizer(cfg)
    else:
        tokenizer = hub.speech_tokenizer(
            cfg, mock_attr="fake_lang", mock_checkpoint="my_ckpt"
        )

    assert str(tokenizer) == "My mock tokenizer"
    assert tokenizer.config.mock_checkpoint == "my_ckpt"
    units = data = torch.rand(2, 3)

    assert torch.equal(tokenizer.encode(data), torch.zeros(2, 3))
    assert torch.equal(tokenizer.decode(units), torch.ones(2, 3))


def test_load_empty_config() -> None:
    with pytest.raises(AssertionError, match=r"\w+ _target_"):
        _ = hub.speech_tokenizer()
