# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import string
import typing as tp

import torch

from stopes.pipelines.monolingual.utils.word_tokenization import get_word_tokenizer

logger = logging.getLogger(__name__)

PUNCTUATIONS_EXCLUDE_APOSTROPHE = (
    string.punctuation.replace("'", "") + "¡¨«°³º»¿‘“”…♪♫ˆᵉ™，ʾ˚"
)

FULL_WIDTH_PUNCT = [
    "！",
    "＂",
    "＃",
    "＄",
    "％",
    "＆",
    "＇",
    "（",
    "）",
    "＊",
    "＋",
    "，",
    "－",
    "．",
    "／",
    "：",
    "；",
    "＜",
    "＝",
    "＞",
    "？",
    "＠",
    "［",
    "＼",
    "］",
    "＾",
    "＿",
    "｀",
    "｛",
    "｜",
    "｝",
    "～",
    "、",
    "。",
    "「",
    "」",
    "〔",
    "〕",
    "〝",
    "〞",
    "–",
    "—",
    "《",
    "》",
    "！",
    "＂",
    "＃",
    "＄",
    "％",
    "＆",
    "＇",
    "（",
    "）",
    "＊",
    "＋",
    "，",
    "－",
    "．",
    "／",
    "：",
    "；",
    "＜",
    "＝",
    "＞",
    "？",
    "＠",
    "［",
    "＼",
    "］",
    "＾",
    "＿",
    "｀",
    "｛",
    "｜",
    "｝",
    "～",
    "、",
    "。",
    "「",
    "」",
    "〔",
    "〕",
    "〝",
    "〞",
    "–",
    "—",
    "《",
    "》",
]


def find_words_in_text(
    text: str, words: tp.List[str], lowercase=False
) -> tp.List[tp.Tuple[int, int]]:
    """
    For each word, find its start (inclusive) and end (exclusive) indices in the text after the previous words.
    The option `lowercase` is NOT recommended for some scripts, because it may occasionally change text length
    (for example, "İ" has length 1, but "İ".lower() has length 2)
    """
    if lowercase:
        text = text.lower()
    start = 0
    word_segs = []
    for word in words:
        if lowercase:
            word = word.lower()
        i_found = text.index(word, start)
        assert (
            i_found >= 0
        ), f"Word '{word}' not found in text '{text}' starting at {start}"
        i_next = i_found + len(word)
        word_segs.append((i_found, i_next))
        start = i_next
    return word_segs


def test_find_words_in_text():
    """test on a tricky example that should never get lowercased"""
    text = "Et qui 를 İ designs"
    tkn = get_word_tokenizer("fra", default="nltk")
    words = tkn.tokenize(text)
    indices = find_words_in_text(text, words)
    assert indices == [(0, 2), (3, 6), (7, 8), (9, 10), (11, 18)]


def prepare_cmn_seq(word_list):
    text = ""
    for word in word_list:
        if word not in FULL_WIDTH_PUNCT and word not in PUNCTUATIONS_EXCLUDE_APOSTROPHE:
            text += f" {word}"
        else:
            text += f"{word}"
    return text


def split_duration_long_spm(durations, char_spm, allow_empty_characters=False):
    """For some sequences spm returns char tokens of length >1 which breaks the pipeline.
    Here we split it and distribute corresponding durations
    """
    spm_lengths = [len(s) for s in char_spm]
    assert (
        max(spm_lengths) > 1
    ), f"this should only be called when some spm length > 1, {spm_lengths}"
    new_durations = []
    new_char_spm = []
    for char, dur in zip(char_spm, durations[0]):
        new_char_spm.extend(list(char))  # split multi char spm tokens
        if allow_empty_characters:
            if dur < len(char):
                text = " ".join(char_spm)
                logger.warning(
                    f"Could not asign nonzero duration to each character of a long SPM token, for the text {text}"
                )
        else:
            assert dur >= len(char), "Impossible to distribute durations in this case"
        dur_per_char = dur // len(char)
        all_but_last_durations = [dur_per_char] * (len(char) - 1)
        all_durations = all_but_last_durations + [dur - sum(all_but_last_durations)]
        new_durations.extend(all_durations)
    return torch.tensor([new_durations]), new_char_spm


def get_words_except_punct(words_wpunct: tp.List[str]) -> tp.List[str]:
    words = []
    for w in words_wpunct:
        if (
            w not in PUNCTUATIONS_EXCLUDE_APOSTROPHE
            and w not in ["``", "''"]
            and w not in FULL_WIDTH_PUNCT
        ):
            words.append(w)
    return words
