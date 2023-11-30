# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import typing as tp
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

try:
    from fairseq2.typing import Device
    from seamless_communication.models.aligner.alignment_extractor import (
        AlignmentExtractor,
    )
except ImportError:
    AlignmentExtractor = None
    Device = None  # type: ignore

from stopes.eval.local_prosody.forced_aligner import BaseSpeechForceAligner
from stopes.eval.local_prosody.forced_aligner_utils import (
    find_words_in_text,
    get_words_except_punct,
    prepare_cmn_seq,
    split_duration_long_spm,
)
from stopes.eval.local_prosody.utterance import Utterance
from stopes.pipelines.monolingual.utils.word_tokenization import get_word_tokenizer

logger = logging.getLogger(__name__)


@dataclass
class UnitY2F2ForcedAlignerConfig:
    """A config for initializing speech tokenizer and M4T aligner"""

    aligner_name: str = "nar_t2u_aligner"
    unit_extractor_name: str = "xlsr2_1b_v2"
    unit_extractor_output_layer_n: int = 35
    unit_extractor_kmeans_uri: str = "https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy"
    device: str = "cpu"

    fps: float = (
        50.0  # the default speech tokenizer produces 20ms frames, i.e. 50 per second
    )
    allow_empty_characters: bool = True
    # this option is recommended for detecting pauses in the end, but by default it is false, for backward compatibility
    append_sow: bool = False


class UnitY2F2ForcedAligner(BaseSpeechForceAligner):
    """
    A wrapper of M4T speech-text aligner that projects character timestamps into words.
    """

    def __init__(self, config: UnitY2F2ForcedAlignerConfig):
        self.config = config
        if AlignmentExtractor is None or Device is None:
            raise ImportError(
                "Please install a recent version of `fairseq2` and `seamless_communication` packages"
            )
        self.extractor = AlignmentExtractor(
            self.config.aligner_name,
            self.config.unit_extractor_name,
            self.config.unit_extractor_output_layer_n,
            self.config.unit_extractor_kmeans_uri,
            device=Device(self.config.device),
        )

    def process_utterance(
        self,
        waveform: torch.Tensor,
        text: tp.Optional[str] = None,
        words: tp.Optional[tp.List[str]] = None,
        lang: str = "eng",
        **kwargs,
    ) -> Utterance:
        """
        Convert a waveform to a prosodically annotated Utterance,
        using a fixed text as a forced transcription.
        This class doesn't do ASR and is purely character based using chat SPM model, text or words arg is required.
        It works exclusively with SPM processed data to avoid inconsistencies e.g. decode(char_spm(signiﬁcado)) returns significado (fi != ﬁ)
        Therefore, the the `words` property of the resulting utterance may differ from `words` provided as the argument.
        """
        if text is None:
            if words is None:
                raise ValueError(
                    "for UnitY2ForcedAligner, text or words argument is required"
                )
            text = " ".join(words)
        tokenizer = self.get_word_tokenizer(lang=lang)

        if lang == "cmn":
            # remove punctuation and introduce spaces for better alignment
            text_word_list = tokenizer.tokenize(text)
            text = prepare_cmn_seq(text_word_list)

        assert (
            text is not None
        ), f"text has to be not None after preprocessing"  # make mypy happy
        with torch.inference_mode():
            if len(waveform.shape) == 2:
                waveform = waveform.mean(0)
            durations, _, tokenized_text_cstr = self.extractor.extract_alignment(
                waveform, text, plot=False, add_trailing_silence=self.config.append_sow
            )
        char_spm: tp.List[str] = [str(token) for token in tokenized_text_cstr]

        # if some spm tokens have length >1 (happens for CMN), then split it up and split corresponding durations
        char_spm_lengths = [len(s) for s in char_spm]
        if max(char_spm_lengths) > 1:
            durations, char_spm = split_duration_long_spm(
                durations=durations,
                char_spm=char_spm,
                allow_empty_characters=self.config.allow_empty_characters,
            )

        durations_cumul = np.concatenate(
            [np.array([0]), np.cumsum(durations.cpu().numpy())]
        )
        alignment_secs = durations_cumul / self.config.fps

        # locating words in the post-processed text
        # intro silence (space) is required to align with alignment_secs
        text_recovered = " " + str(
            self.extractor.alignment_model.alignment_frontend.decode_text.decode_from_tokens(
                char_spm
            )
        )

        # make sure words will only have audible words in it excluding punctuation and parentheses
        words = get_words_except_punct(tokenizer.tokenize(text_recovered))
        word_segs = find_words_in_text(text_recovered, words)

        utterance = Utterance(
            text=text,
            words=words,
            starts=[alignment_secs[i] for i, j in word_segs],
            ends=[alignment_secs[j] for i, j in word_segs],
        )
        utterance.compute_aligner_based_pause_lengths()
        return utterance

    @lru_cache(maxsize=1000)
    def get_word_tokenizer(self, lang: str = "eng"):
        """Initialize a word tokenizer for the given language and cache it"""
        return get_word_tokenizer(lang=lang, default="nltk")
