# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Based on https://github.com/keithito/tacotron


import logging
import re
import string
from typing import Callable, List

import pandas as pd
from unidecode import unidecode

from stopes.utils.tts_preprocessing.cmn import cmn_norm
from stopes.utils.tts_preprocessing.numbers import (
    SUPPORTED_LANGS as NUMEXP_SUPPORTED_LANGS,
)
from stopes.utils.tts_preprocessing.numbers import expand_numbers

logger = logging.getLogger(__name__)

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = {
    "eng_Latn": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("mrs", "misess"),
            ("mr", "mister"),
            ("dr", "doctor"),
            ("st", "saint"),
            ("co", "company"),
            ("jr", "junior"),
            ("maj", "major"),
            ("gen", "general"),
            ("drs", "doctors"),
            ("rev", "reverend"),
            ("lt", "lieutenant"),
            ("hon", "honorable"),
            ("sgt", "sergeant"),
            ("capt", "captain"),
            ("esq", "esquire"),
            ("ltd", "limited"),
            ("col", "colonel"),
            ("ft", "fort"),
        ]
    ],
}


def expand_abbreviations(text, lang="eng_Latn"):
    if lang not in _abbreviations:
        return text

    for regex, replacement in _abbreviations[lang]:
        text = re.sub(regex, replacement, text)
    return text


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


PUNCTUATION_EXCLUDE_APOSTROPHE = (
    string.punctuation.replace("'", "") + "¡¨«°³º»¿‘“”…♪♫ˆᵉ™，ʾ˚"
)
PUNCTUATIONS_TO_SPACE = "-/–·—•"


def remove_punctuations(text, punctuations=string.punctuation):
    text = text.translate(
        str.maketrans(PUNCTUATIONS_TO_SPACE, " " * len(PUNCTUATIONS_TO_SPACE))
    )
    return text.translate(str.maketrans("", "", punctuations))


def remove_parentheses(text: str) -> str:
    # remove all substring within () or []
    out = ""
    num_p = 0
    start_i = 0
    for i, c in enumerate(text):
        if c == "(" or c == "[" or c == "（":
            if num_p == 0 and i > start_i:
                out += text[start_i:i]
            num_p += 1
        elif c == ")" or c == "]" or c == "）":
            num_p -= 1
            if num_p == 0:
                start_i = i + 1

    if len(text) > start_i:
        out += text[start_i:]

    return out.strip()


REMAP_CHARS = {
    "`": "'",
    "’ ": " ",
    "’": "'",
}


def remap_chars(text, remap_chars=REMAP_CHARS):
    for k, v in remap_chars.items():
        text = text.replace(k, v)
    return text


def expand_capitals(text):
    words = text.split()
    for i, w in enumerate(words):
        if w.isupper():
            words[i] = " ".join(w)

    return " ".join(words)


def clean_eng_Latn(text, remove_punctuation="all"):
    """Pipeline for English text, including number and abbreviation expansion."""
    if remove_punctuation == "all":
        punctuation = string.punctuation
    elif remove_punctuation == "keep_apostrophe":
        punctuation = PUNCTUATION_EXCLUDE_APOSTROPHE
    text = convert_to_ascii(text)
    text = remap_chars(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = remove_parentheses(text)
    text = remove_punctuations(text, punctuation=punctuation)
    text = collapse_whitespace(text)
    text = text.strip()
    return text


def text_cleaners(text, lang="eng_Latn"):
    # No op for Hokkien
    if lang == "oan_Hant":
        return text

    text = remap_chars(text)
    text = expand_capitals(text)
    text = lowercase(text)
    text = remove_parentheses(text)

    # Convert Chinese to simplified script
    if lang in {"cmn_Hans", "cmn_Hant"}:
        from hanziconv import HanziConv

        text = HanziConv.toSimplified(text)

    # Expand numbers
    if lang in NUMEXP_SUPPORTED_LANGS:
        try:
            text = expand_numbers(text, lang)
        except Exception as e:
            logger.error(f"{type(e)} when processing the following line: {text}")
            raise

    text = expand_abbreviations(text, lang)

    if lang in {"cmn_Hans", "cmn_Hant"}:
        text = cmn_norm(text)
    else:
        text = remove_punctuations(text, punctuations=PUNCTUATION_EXCLUDE_APOSTROPHE)
        text = collapse_whitespace(text)

    if lang in {"arb_Arab", "ary_Arab", "arz_Arab"}:
        import tnkeeh as tn

        text = tn._clean_text(
            text,
            remove_special_chars=True,
            normalize=True,
            remove_diacritics=True,
            remove_tatweel=True,
            segment=False,
            remove_repeated_chars=False,
            remove_html_elements=False,
            remove_english=False,
            excluded_chars=[],
            remove_links=False,
            remove_twitter_meta=False,
            remove_long_words=False,
        )

    text = text.strip()
    return text


def clean_column(df: pd.DataFrame, cleaner_func: Callable, column_name: str):
    assert (
        column_name in df.columns
    ), f"Column {column_name} does not exist in the manifest"

    logging.info(f"Applying {cleaner_func} to {column_name}")
    df[column_name] = df[column_name].map(lambda x: cleaner_func(str(x)))


def clean_df(df: pd.DataFrame, cleaner_funcs: List[Callable], column_name: str):
    for cleaner_func in cleaner_funcs:
        clean_column(df, cleaner_func, column_name)


def apply_text_functions(text_funcs: List[Callable], text: str) -> str:
    for func in text_funcs:
        text = func(text)
    return text
