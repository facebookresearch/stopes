# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from unittest.mock import mock_open, patch

import pytest
import torch
from omegaconf import DictConfig

from stopes.speech.mms_tts import (
    MMSTTS,
    MMSTTSDataConfig,
    MMSTTSInferenceInput,
    MMSTTSTextProcessor,
)


@pytest.fixture
@patch("builtins.open", new_callable=mock_open, read_data="a\nb\nc")
def mock_mms_tts_text_processor(mock_file):
    mock_mms_tts_text_proceesor = MMSTTSTextProcessor(
        lang="eng",
        vocab_file="mock",
        data_config=MMSTTSDataConfig(),
    )
    return mock_mms_tts_text_proceesor


@pytest.fixture
@patch("builtins.open", new_callable=mock_open, read_data="a\nb\nc")
def mock_mms_tts_text_processor_when_uroman_is_needed(mock_file):
    mock_mms_tts_text_proceesor = MMSTTSTextProcessor(
        lang="amh",
        vocab_file="mock",
        data_config=MMSTTSDataConfig(training_files="train.uroman"),
    )
    return mock_mms_tts_text_proceesor


@pytest.fixture
@patch("builtins.open", new_callable=mock_open, read_data="a\nb\nc")
def mock_mms_tts_text_processor_when_lang_is_ron(mock_file):
    mock_mms_tts_text_proceesor = MMSTTSTextProcessor(
        lang="ron",
        vocab_file="mock",
        data_config=MMSTTSDataConfig(training_files="train.uroman"),
    )
    return mock_mms_tts_text_proceesor


def test_MMSTTSTextProcessor__init__(
    mock_mms_tts_text_processor, mock_mms_tts_text_processor_when_uroman_is_needed
):
    assert mock_mms_tts_text_processor.nb_vocab == 3
    assert mock_mms_tts_text_processor.symbol_to_id == {"a": 0, "b": 1, "c": 2}
    assert mock_mms_tts_text_processor.is_uroman == False
    assert mock_mms_tts_text_processor.add_blank == True

    assert mock_mms_tts_text_processor_when_uroman_is_needed.is_uroman == True


@pytest.mark.parametrize(
    "text,expected_output",
    [
        # single string
        (
            "bc",
            MMSTTSInferenceInput(
                text_ids=torch.LongTensor([[0, 1, 0, 2, 0]]),
                text_lengths=torch.LongTensor([5]),
            ),
        ),
        # list of strings
        (
            ["bc", "cb"],
            MMSTTSInferenceInput(
                text_ids=torch.LongTensor([[0, 1, 0, 2, 0], [0, 2, 0, 1, 0]]),
                text_lengths=torch.LongTensor([5, 5]),
            ),
        ),
        # with expected padding
        (
            ["bc", "c"],
            MMSTTSInferenceInput(
                text_ids=torch.LongTensor([[0, 1, 0, 2, 0], [0, 2, 0, 0, 0]]),
                text_lengths=torch.LongTensor([5, 3]),
            ),
        ),
    ],
)
def test_MMSTTSTextProcessor_text2tensor(
    mock_mms_tts_text_processor, text, expected_output
):
    inference_input: MMSTTSInferenceInput = mock_mms_tts_text_processor.text2tensor(
        text
    )
    assert torch.equal(inference_input.text_ids, expected_output.text_ids)
    assert torch.equal(inference_input.text_lengths, expected_output.text_lengths)


@pytest.mark.parametrize(
    "text",
    ["", [None], [None, "ab"], ["", "ab"]],
)
def test_MMSTTSTextProcessor_text2tensor_with_invalid_input(
    mock_mms_tts_text_processor, text
):
    with pytest.raises(ValueError):
        mock_mms_tts_text_processor.text2tensor(text)


@pytest.mark.parametrize(
    "sentence,expected_output",
    [
        ("ab", [0, 0, 0, 1, 0]),
        ("aB", [0, 0, 0, 1, 0]),
    ],
)
def test_MMSTTSTextProcessor_text2tensor_text2id(
    mock_mms_tts_text_processor, sentence, expected_output
):
    assert mock_mms_tts_text_processor._text2id(sentence) == expected_output


@pytest.mark.parametrize(
    "lst,item,expected_output",
    [
        ([1, 2], 0, [0, 1, 0, 2, 0]),
        ([1, 2], 3, [3, 1, 3, 2, 3]),
    ],
)
def test_MMSTTSTextProcessor_intersperse(
    mock_mms_tts_text_processor, lst, item, expected_output
):
    mock_mms_tts_text_processor._intersperse(lst, item) == expected_output


def test_MMSTTSTextProcessor_is_handling_special_romanian_character(
    mock_mms_tts_text_processor,
    mock_mms_tts_text_processor_when_lang_is_ron,
):
    assert (
        mock_mms_tts_text_processor_when_lang_is_ron._is_handling_special_romanian_character(
            "ț"
        )
        is True
    )
    assert (
        mock_mms_tts_text_processor._is_handling_special_romanian_character("a")
        is False
    )
    assert (
        mock_mms_tts_text_processor._is_handling_special_romanian_character("ț")
        is False
    )


@pytest.fixture
@patch("builtins.open", new_callable=mock_open, read_data="a\nb\nc")
def mock(mock_file):
    mock_mms_tts_text_proceesor = MMSTTSTextProcessor(
        lang="eng",
        vocab_file="mock",
        data_config=MMSTTSDataConfig(),
    )
    return mock_mms_tts_text_proceesor


MOCK_MMS_TTS_CONFIG = DictConfig(
    {
        "data": {
            "add_blank": True,
            "cleaned_text": True,
            "filter_length": 1024,
            "hop_length": 256,
            "max_wav_value": 32768.0,
            "mel_fmax": None,
            "mel_fmin": 0.0,
            "n_mel_channels": 80,
            "n_speakers": 0,
            "sampling_rate": 16000,
            "text_cleaners": ["transliteration_cleaners"],
            "training_files": "train.ltr",
            "validation_files": "dev.ltr",
            "win_length": 1024,
        },
        "model": {
            "filter_channels": 768,
            "hidden_channels": 192,
            "inter_channels": 192,
            "kernel_size": 3,
            "n_heads": 2,
            "n_layers": 6,
            "n_layers_q": 3,
            "p_dropout": 0.1,
            "resblock": "1",
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "resblock_kernel_sizes": [3, 7, 11],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [16, 16, 4, 4],
            "upsample_rates": [8, 8, 2, 2],
            "use_spectral_norm": False,
        },
        "train": {
            "batch_size": 64,
            "betas": [0.8, 0.99],
            "c_kl": 1.0,
            "c_mel": 45,
            "epochs": 20000,
            "eps": 1e-09,
            "eval_interval": 1000,
            "fp16_run": True,
            "init_lr_ratio": 1,
            "learning_rate": 0.0002,
            "log_interval": 200,
            "lr_decay": 0.999875,
            "seed": 1234,
            "segment_size": 8192,
            "warmup_epochs": 0,
        },
    }
)


@pytest.fixture
def mock_mms_tts():
    with patch("builtins.open", new_callable=mock_open, read_data="a\nb\nc"):
        with patch(
            "stopes.speech.mms_tts.load_config", return_value=MOCK_MMS_TTS_CONFIG
        ):
            mms_tts = MMSTTS.build_from_config(
                config=DictConfig(
                    {
                        "_target_": "stopes.speech.mms_tts.MMSTTS",
                        "lang": "eng",
                        "model_config_path": "/mock/mms/tts/eng/config.yaml",
                        "checkpoint_path": "/mock/mms/tts/eng/G_100000.pth",
                        "vocab_path": "/mock/mms/tts/eng//vocab.txt",
                    }
                ),
            )
            return mms_tts


@pytest.fixture
def mock_mms_tts_with_overriden_device():
    with patch("builtins.open", new_callable=mock_open, read_data="a\nb\nc"):
        with patch(
            "stopes.speech.mms_tts.load_config", return_value=MOCK_MMS_TTS_CONFIG
        ):
            mms_tts = MMSTTS.build_from_config(
                config=DictConfig(
                    {
                        "_target_": "stopes.speech.mms_tts.MMSTTS",
                        "lang": "eng",
                        "model_config_path": "/mock/mms/tts/eng/config.yaml",
                        "checkpoint_path": "/mock/mms/tts/eng/G_100000.pth",
                        "vocab_path": "/mock/mms/tts/eng//vocab.txt",
                    }
                ),
                device="cpu",
            )
            return mms_tts


def test_MMSTTS_init(mock_mms_tts):
    assert mock_mms_tts.lang == "eng"
    assert mock_mms_tts.config == MOCK_MMS_TTS_CONFIG
    assert mock_mms_tts.checkpoint_path == "/mock/mms/tts/eng/G_100000.pth"
    assert mock_mms_tts.device == "cuda"
    assert type(mock_mms_tts.text_processor) == MMSTTSTextProcessor


def test_MMSTTS_init_with_overriden_device(mock_mms_tts_with_overriden_device):
    assert mock_mms_tts_with_overriden_device.device == "cpu"


def test_MMSTTS_get_valid_lengths_based_on_masks(mock_mms_tts):
    # fmt: off
    # turn off the format checking for readability
    assert mock_mms_tts._get_valid_lengths_based_on_masks(
        audio_tensor=torch.LongTensor(
            [
                [
                    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95,
                ],
                [
                    0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91, 0.951,
                ],
            ]
        ),  # audio_tensor has shape (2, 10)
        mask=torch.LongTensor(
            [
                [
                    1, 1, 1, 1, 0,
                ],
                [
                    1, 1, 0, 0, 0,
                ]
            ]
        ),
    ) == [8, 4]
    # fmt: on
