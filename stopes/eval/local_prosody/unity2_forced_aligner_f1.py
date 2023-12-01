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

from stopes.eval.local_prosody.forced_aligner import BaseSpeechForceAligner
from stopes.eval.local_prosody.forced_aligner_utils import (
    find_words_in_text,
    get_words_except_punct,
    prepare_cmn_seq,
    split_duration_long_spm,
)
from stopes.eval.local_prosody.utterance import Utterance
from stopes.hub import speech_tokenizer
from stopes.pipelines.monolingual.utils.word_tokenization import get_word_tokenizer
from stopes.utils.aligner_utils import Aligner

logger = logging.getLogger(__name__)


@dataclass
class UnitY2ForcedAlignerConfig:
    """A config for initializing speech tokenizer and M4T aligner"""

    nar_model_ckpt: str
    m4t_char_dict_path: str
    m4t_unit_dict_path: str
    m4t_char_spm_path: str
    m4t_lang_token_dict: str
    speech_tokenizer: str = "lang41_10k_xlsr_lyr35"
    fps: float = (
        50.0  # the default speech tokenizer produces 20ms frames, i.e. 50 per second
    )
    allow_empty_characters: bool = True
    # this option is recommended for detecting pauses in the end, but by default it is false, for backward compatibility
    append_sow: bool = False


class UnitY2ForcedAligner(BaseSpeechForceAligner):
    """
    A wrapper of M4T speech-text aligner that projects character timestamps into words.
    """

    def __init__(self, config: UnitY2ForcedAlignerConfig):
        self.config = config
        self.xlsr_tokenizer = speech_tokenizer(self.config.speech_tokenizer)
        self.m4t_aligner_model = Aligner(
            config.nar_model_ckpt,
            config.m4t_char_dict_path,
            config.m4t_unit_dict_path,
            config.m4t_char_spm_path,
            config.m4t_lang_token_dict,
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
            units = self.xlsr_tokenizer.encode(waveform)
            units_str = " ".join([str(u) for u in units[0].tolist()])
            (
                durations,
                attn_lprob,
                char_spm,
                char_ids,
                unit_ids,
            ) = self.m4t_aligner_model.compute_alignment(
                text, units_str, append_sow=self.config.append_sow
            )
        # if some spm tokens have length >1 (happens for CMN), then split it up and split corresponding durations
        char_spm_lengths = [len(s) for s in char_spm]
        if max(char_spm_lengths) > 1:
            durations, char_spm = split_duration_long_spm(
                durations=durations,
                char_spm=char_spm,
                allow_empty_characters=self.config.allow_empty_characters,
            )

        durations_cumul = np.concatenate([np.array([0]), np.cumsum(durations)])
        alignment_secs = durations_cumul / self.config.fps

        # locating words in the post-processed text

        text_recovered = " " + self.m4t_aligner_model.char_spm.decode(
            char_spm
        )  # intro silence (space) is required to align with alignment_secs
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
