# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import functools
import logging
import typing as tp
from dataclasses import dataclass, field
from textwrap import dedent
from typing import List, Optional, Union

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from stopes.core import utils as stopes_utils
from stopes.hub import load_config
from stopes.modules.preprocess.uromanize_cli_module import uromanize
from stopes.speech.tts import TTS, TTSOutput

logger = logging.getLogger(__name__)


@dataclass
class MMSTTSDataConfig:
    add_blank: bool = True
    cleaned_text: bool = True
    filter_length: int = 1024
    hop_length: int = 256
    max_wav_value: float = 32768.0
    mel_fmax: Optional[float] = None
    mel_fmin: float = 0.0
    n_mel_channels: int = 80
    n_speakers: int = 0
    sampling_rate: int = 16000
    text_cleaners: List[str] = field(
        default_factory=lambda: ["transliteration_cleaners"]
    )
    training_files: str = "train.ltr"
    validation_files: str = "dev.ltr"
    win_length: int = 1024


@dataclass
class MMSTTSModelConfig:
    filter_channels: int = 768
    hidden_channels: int = 192
    inter_channels: int = 192
    kernel_size: int = 3
    n_heads: int = 2
    n_layers: int = 6
    n_layers_q: int = 3
    p_dropout: float = 0.1
    resblock: str = "1"
    resblock_dilation_sizes: List[List[int]] = field(
        default_factory=lambda: [[1, 2, 3], [1, 3, 5], [1, 3, 5]]
    )
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [16, 16, 4, 4])
    upsample_rates: List[int] = field(default_factory=lambda: [8, 8, 2, 2])
    use_spectral_norm: bool = False


@dataclass
class MMSTTSTrainConfig:
    batch_size: int = 64
    betas: List[float] = field(default_factory=lambda: [0.8, 0.99])
    c_kl: float = 1.0
    c_mel: int = 45
    epochs: int = 20000
    eps: float = 1.0e-09
    eval_interval: int = 1000
    fp16_run: bool = True
    init_lr_ratio: int = 1
    learning_rate: float = 0.0002
    log_interval: int = 200
    lr_decay: float = 0.999875
    seed: int = 1234
    segment_size: int = 8192
    warmup_epochs: int = 0


@dataclass
class MMSTTSConfig:
    data: MMSTTSDataConfig
    model: MMSTTSModelConfig
    train: MMSTTSTrainConfig


@dataclass
class MMSTTSRootConfig:
    lang: str
    model_config_path: str
    checkpoint_path: str
    vocab_path: str


@dataclass
class MMSTTSInferenceInput:
    text_ids: torch.Tensor
    text_lengths: torch.Tensor

    def to(self, device: str) -> None:
        self.text_ids = self.text_ids.to(device)
        self.text_lengths = self.text_lengths.to(device)


class MMSTTSTextProcessor:
    def __init__(self, lang: str, vocab_file: str, data_config: MMSTTSDataConfig):
        self.lang = lang
        symbols = [
            x.rstrip("\n") for x in open(vocab_file, encoding="utf-8").readlines()
        ]
        self.nb_vocab = len(symbols)
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.is_uroman = data_config.training_files.split(".")[-1] == "uroman"
        self.add_blank = data_config.add_blank

    def text2tensor(self, text: Union[List[str], str]) -> MMSTTSInferenceInput:
        if isinstance(text, str):
            text = [text]
        if any(x is None or len(x) == 0 for x in text):
            raise ValueError(f"text expects to not None or empty, got {text}")
        if self.is_uroman:
            text = uromanize(text)
        return self._text2tensor(text)

    def _text2tensor(self, texts: List[str]) -> MMSTTSInferenceInput:
        processed_texts: List[List[int]] = []
        for text in texts:
            processed_texts.append(self._text2id(text))
        text_lengths = [len(seq) for seq in processed_texts]
        max_text_len = max(text_lengths)
        padded_texts = [
            seq + [0] * (max_text_len - len(seq)) for seq in processed_texts
        ]
        return MMSTTSInferenceInput(
            text_ids=torch.LongTensor(padded_texts),
            text_lengths=torch.LongTensor(text_lengths),
        )

    @staticmethod
    def _intersperse(lst: List[int], item: int) -> List[int]:
        # copied from https://github.com/jaywalnut310/vits/blob/main/commons.py#LL24C1-L28C1
        result = [item] * (len(lst) * 2 + 1)
        result[1::2] = lst
        return result

    def _is_handling_special_romanian_character(self, char: str):
        return self.lang == "ron" and char == "ț"

    def _text2id(self, sentence: str) -> List[int]:
        char_ids = []
        for char in sentence.strip():
            char = char.lower()
            if self._is_handling_special_romanian_character(char):
                char = "ţ"
            if char in self.symbol_to_id:
                char_ids.append(self.symbol_to_id[char])
        if self.add_blank:
            char_ids = self._intersperse(char_ids, 0)
        return char_ids


class MMSTTS(TTS):
    NOISE_SCALE = 0.667
    NOISE_SCALE_W = 0.8
    LENGTH_SCALE = 1.0
    NB_THREADS_FOR_WAV_GENERATION = 32
    DEFAULT_DEVICE = "cuda"
    """
    The Text-to-Speech module wrapper for MMS model,
        proposed in Pratap et al. Scaling Speech Technology to 1,000+ Languages. 2023.

    Usage example:
    >>> from stopes.hub import tts
    >>> mms_tts_eng = tts(model="mms", lang="eng")
    >>> mms_tts_eng.generate(["this is a test", "this is an another test"])
    """

    def __init__(self, config: MMSTTSRootConfig):
        self.lang = config.lang
        self.config: MMSTTSConfig = self._init_model_config(
            config.model_config_path, config.lang
        )
        self.checkpoint_path = config.checkpoint_path
        self.text_processor = MMSTTSTextProcessor(
            config.lang, config.vocab_path, self.config.data
        )
        self.device = getattr(config, "device", MMSTTS.DEFAULT_DEVICE)

    def __str__(self):
        return f"MMS Text-to-Speech (TTS) for the {self.lang} langauge"

    @classmethod
    def build_from_config(cls, config: DictConfig, **kwargs) -> "MMSTTS":
        if kwargs:
            config = tp.cast(DictConfig, OmegaConf.unsafe_merge(config, kwargs))
        assert hasattr(
            config, "_target_"
        ), "You need to specify the module to create in the yaml file with _target_"
        target = config._target_
        OmegaConf.resolve(config)
        return hydra.utils.instantiate({"_target_": target}, config, _recursive_=False)

    @classmethod
    def _init_model_config(cls, model_config_path: str, lang: str) -> MMSTTSConfig:
        config = load_config(model_config_path, namespace=f"mms/tts/{lang}")
        model_config = tp.cast(
            MMSTTSConfig, stopes_utils.promote_config(config, MMSTTSConfig)
        )
        return model_config

    @functools.cached_property
    def model(self):
        # models module from vits
        # https://github.com/jaywalnut310/vits/blob/main/models.py
        try:
            from models import SynthesizerTrn
        except ImportError as err:
            logger.error(
                dedent(
                    """
            Fail to import SynthesizerTrn!

            In order to use the MMSTTS's model, you have to install the VITS library.
            The instructions are as follows:
                cd /path/to/where/you/put/VIST/repo
                git clone git@github.com:jaywalnut310/vits.git
                cd monotonic_align
                mkdir monotonic_align
                python setup.py build_ext --inplace
            After doing that you can include the VITS repository path to the PYTHONPATH.
            
            PYTHONPATH=$PYTHONPATH:/path/to/vits python /path/to/where/you/use/MMSTTS/xxxx.py
            """
                )
            )
            raise err

        model = SynthesizerTrn(
            n_vocab=self.text_processor.nb_vocab,
            spec_channels=self.config.data.filter_length // 2 + 1,
            segment_size=self.config.train.segment_size // self.config.data.hop_length,
            **self.config.model,
        )
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return model.to(self.device)

    @staticmethod
    def _get_valid_lengths_based_on_masks(
        audio_tensor: torch.Tensor, mask: torch.Tensor
    ) -> List[int]:
        return (
            torch.ceil(mask.sum(dim=1) / mask.size(1) * audio_tensor.size(1))
            .to(torch.int32)
            .tolist()
        )

    @property
    def sampling_rate(self):
        return self.config.data.sampling_rate

    def generate(
        self,
        texts: List[str],
    ) -> TTSOutput:
        if texts is None or len(texts) == 0:
            logger.warn("Empty input!")
            return TTSOutput(waveform_tensors=[], sampling_rate=self.sampling_rate)
        text_input: MMSTTSInferenceInput = self.text_processor.text2tensor(texts)
        text_input.to(self.device)
        with torch.no_grad():
            # We need the mask because some inputs might have been padded,
            audio_tensor, _, mask, _ = self.model.infer(
                text_input.text_ids,
                text_input.text_lengths,
                noise_scale=MMSTTS.NOISE_SCALE,
                noise_scale_w=MMSTTS.NOISE_SCALE_W,
                length_scale=MMSTTS.LENGTH_SCALE,
            )
        audio_tensor = audio_tensor.squeeze(1)
        mask = mask.squeeze(1)
        valid_lengths = self._get_valid_lengths_based_on_masks(audio_tensor, mask)
        return TTSOutput(
            waveform_tensors=[
                audio_tensor[i][: valid_lengths[i]] for i in range(len(texts))
            ],
            sampling_rate=self.sampling_rate,
        )
