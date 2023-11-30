# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
This module contains a tool that can extract rich prosodic annotations of spoken utterances:
- Timestamps of words and pauses between them (produced with forced aligmment and corrected with VAD)
- Speech rate in words, syllables, characters, all phonemes or wovels, per net speech duration (without pauses)
- Probabilities of emphasis for each word

As an input, this tool requires a dataframe with transcriptions and paths to the audios.
Please take a look at the local readme.md file for additional details.
"""

import csv
import logging
import typing as tp
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import hydra
import pandas as pd
import syllables
import torch
from ipapy.ipastring import IPAString
from omegaconf import DictConfig
from tqdm.auto import tqdm

import stopes.hub
from stopes.core import utils as core_utils
from stopes.eval.local_prosody.emphasis_detection import (
    load_frame_tagger,
    predict_emphasis,
)
from stopes.eval.local_prosody.forced_aligner import BaseSpeechForceAligner
from stopes.eval.local_prosody.phonemization import cached_phonemize
from stopes.eval.local_prosody.utterance import PRECISION_DECIMALS, Utterance
from stopes.modules.speech.utils import parse_audio
from stopes.modules.speech.vad import VAD

logger = logging.getLogger(__name__)


SUPPORTED_SPEECH_UNIT_NAMES = [
    "sentence",
    "word",
    "syllable",
    "char",
    "phoneme",
    "vowel",
]


def count_speech_units(utterance: Utterance, unit="word", lang="eng"):
    """Compute speech rate as number of units-of-speech per second.
    `unit` can be one of {"sentence", "word", "syllable", "char", "phoneme", "vowel"}.
    """
    if unit == "sentence":
        units = 1
    elif unit == "word":
        units = len(utterance.words)
    elif unit == "char":
        units = sum(len(w) for w in utterance.words)
    elif unit == "syllable":
        units = sum(syllables.estimate(w) for w in utterance.words)
    elif unit == "phoneme":
        phonemized = cached_phonemize(utterance.words, language=lang)
        units = sum(len(w) for w in phonemized)
    elif unit == "vowel":
        phonemized = cached_phonemize(utterance.words, language=lang)
        units = sum(len(IPAString(unicode_string=w).vowels) for w in phonemized)
    else:
        raise ValueError(f"speech unit unknown: {unit}")
    return units


@dataclass
class AnnotateUtterancesConfig:
    data_path: Path
    audio_column: str
    audio_sampling_factor: int = 16
    result_path: tp.Optional[Path] = None
    text_column: tp.Optional[str] = None
    # if the `lang_column`` is indicated, it has higher priority than the `lang` argument
    lang_column: tp.Optional[str] = None
    # forced_aligner refers to a config in stopes/pipelines/bitext/conf/forced_alignment loaded via stopes.hub
    forced_aligner: str = "fairseq2_nar_t2u_aligner"
    lang: str = "eng"
    speech_units: tp.List[str] = field(
        default_factory=lambda: ["word", "syllable", "char"]
    )
    net: bool = True
    vad: bool = False
    # if we set vad=True, but keep vad_model empty, a version of the Silero VAD model is automatically downloaded
    vad_model: tp.Optional[Path] = None
    vad_window_size: int = 512
    emphasis_model: tp.Optional[Path] = None


class UtteranceAnnotator:
    def __init__(self, config: AnnotateUtterancesConfig):
        self.config = config
        self.aligner: BaseSpeechForceAligner = stopes.hub.forced_aligner(
            config.forced_aligner
        )
        if config.vad_model or config.vad:
            self.vad = VAD(model_path=config.vad_model or "auto")
        else:
            self.vad = None  # type: ignore
        if config.emphasis_model:
            self.emphasis_model = load_frame_tagger(config.emphasis_model)
        else:
            self.emphasis_model = None

    def process_data(
        self, data: pd.DataFrame, config: AnnotateUtterancesConfig
    ) -> pd.DataFrame:
        # TODO: split the config into parts responsible for models and for data
        tqdm.pandas()
        return data.progress_apply(
            partial(self.row_transform, config=config),
            axis=1,
            result_type="expand",
        )

    def row_transform(
        self, row: tp.Dict[str, tp.Any], config: AnnotateUtterancesConfig
    ) -> tp.Dict[str, tp.Any]:
        units = config.speech_units
        if isinstance(units, str):
            units = [units]
        utterance = self.annotate_utterance(row, config=config)

        # Computing the speech rate
        if config.net:
            if utterance.vad_duration is not None:
                duration = utterance.vad_duration
            else:
                duration = utterance.net_duration
        else:
            duration = utterance.total_duration or utterance.trimmed_duration
        rates = [
            count_speech_units(
                utterance,
                unit=unit,
                lang=utterance.lang,
            )
            / duration
            for unit in units
        ]
        results: tp.Dict[str, tp.Any] = dict(
            utterance=utterance.serialize(),
            text_with_markup=utterance.get_text_with_markup(),
            duration=duration,
        )
        for unit_name, unit_rate in zip(units, rates):
            results[f"speech_rate_{unit_name}"] = unit_rate
        if "id" in row:
            results["id"] = row["id"]
        return results

    def annotate_utterance(
        self, row: tp.Dict[str, tp.Any], config: AnnotateUtterancesConfig
    ) -> Utterance:
        if config.text_column:
            raw_text = str(row[config.text_column])
        else:
            raw_text = None

        if config.lang_column and config.lang_column in row:
            lang = row[config.lang_column]
        else:
            lang = config.lang

        audio_path = row[config.audio_column]
        waveform = parse_audio(
            audio_path, sampling_factor=config.audio_sampling_factor
        ).load()

        # Audio processing 1: word alignment
        utterance = self.aligner.process_utterance(
            waveform,
            text=raw_text,
            lang=lang,
        )
        utterance.lang = lang
        if "id" in row:
            utterance.id = row["id"]
        utterance.total_duration = waveform.size(-1) / (
            config.audio_sampling_factor * 1000
        )

        # Audio processing 2: emphasis detection
        if self.emphasis_model:
            frame_preds, word_mean_preds, word_max_preds = predict_emphasis(
                wav=waveform, model=self.emphasis_model, utterance=utterance
            )
            utterance.emphasis_scores = [
                round(x, PRECISION_DECIMALS) for x in word_mean_preds
            ]

        # Audio processing 3: VAD, to refine total speech duration and duration of pauses
        if self.vad:
            (
                vad_duration,
                speech_probs,
                audio_length_samples,
            ) = self.vad.get_net_speech_duration(
                waveform, window=config.vad_window_size
            )
            sample_level_speech_probs = torch.tensor(speech_probs).repeat_interleave(
                config.vad_window_size
            )
            sample_level_speech_probs = pad_or_trim(
                sample_level_speech_probs, audio_length_samples
            )
            utterance.compute_speech_prob_between_words(sample_level_speech_probs)
            utterance.vad_duration = vad_duration

        # TODO: maybe, incorporate the speech rate estimations as a part of the Utterance
        return utterance


def pad_or_trim(x: torch.Tensor, size: int, pad_value=0) -> torch.Tensor:
    """Pad or trim a tensor along the first dimension to the desired length"""
    if x.size(0) > size:
        return x[:size]
    elif x.size(0) < size:
        return torch.nn.functional.pad(
            x,
            pad=(0, size - x.size(0)),
            value=pad_value,
        )
    else:
        return x


def transcribe_and_compute_speech_rate(
    config: AnnotateUtterancesConfig, data: tp.Optional[pd.DataFrame] = None
):
    annotator = UtteranceAnnotator(config=config)
    if data is None:
        data = pd.read_csv(config.data_path, sep="\t", quoting=3)

    results_df = annotator.process_data(data, config=config)

    if config.result_path:
        results_df.to_csv(
            config.result_path, sep="\t", index=None, quoting=csv.QUOTE_MINIMAL
        )
        logger.info(f"Transcripts and speech rates saved to {config.result_path}")
    return results_df


@hydra.main(version_base="1.1")
def main(config: DictConfig):
    typed_config: AnnotateUtterancesConfig = core_utils.promote_config(
        config, AnnotateUtterancesConfig
    )
    if not typed_config.result_path:
        raise ValueError(
            "When running speech rate calculation from CLI, please provide a result_path."
        )
    transcribe_and_compute_speech_rate(config=typed_config)


if __name__ == "__main__":
    # TODO: separate the speech rate tools from the main transcription command
    main()
