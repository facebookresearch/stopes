# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import nltk
from sacremoses import MosesTokenizer

LANGS_MOSES = {
    "cat": "ca",
    "ces": "cs",
    "dan": "da",
    "nld": "nl",
    "eng": "en",
    "fin": "fi",
    "fra": "fr",
    "deu": "de",
    "ell": "el",
    "hun": "hu",
    "isl": "is",
    "ita": "it",
    "lav": "lv",
    "lit": "lt",
    "nob": "no",
    "pol": "pl",
    "por": "pt",
    "ron": "ro",
    "rus": "ru",
    "slk": "sk",
    "slv": "sl",
    "spa": "es",
    "swe": "sv",
    "tur": "tr",
}
# this list is copypasted, in order to aviod extra dependencies


class BaseWordTokenizer:
    def tokenize(self, text: str) -> tp.List[str]:
        raise NotImplementedError()


TWordTokenizer = tp.Union[MosesTokenizer, BaseWordTokenizer]


def get_word_tokenizer(lang: str, default="moses") -> TWordTokenizer:
    """
    Load a language-specific word tokenizer (currently supported for Mandarin and Thai), or a default one.
    By default, the default tokenizer is Moses; for some applications it is recommended to use NLTK instead,
    because Moses sometimes incorrectly processes contractions.
    """
    if lang in {"thai", "tha"}:
        return ThaiTokenizer()
    if lang in {"cmn", "zho", "zho_Hans", "zh"}:
        return ChineseTokenizer()
    if default == "nltk":
        return NLTKTokenizer()
    moses_lang = LANGS_MOSES.get(lang[:3], lang[:2])
    tokenizer = MosesTokenizer(moses_lang)
    return tokenizer


class NLTKTokenizer(BaseWordTokenizer):
    def tokenize(self, text: str) -> tp.List[str]:
        return nltk.tokenize.word_tokenize(text)


class ThaiTokenizer(BaseWordTokenizer):
    def __init__(self):
        from pythainlp import word_tokenize

        self.f = word_tokenize

    def tokenize(self, text: str) -> tp.List[str]:
        return self.f(text, keep_whitespace=False, join_broken_num=False)


class ChineseTokenizer(BaseWordTokenizer):
    def __init__(self):
        import pkuseg

        self.seg = pkuseg.pkuseg()

    def tokenize(self, text: str) -> tp.List[str]:
        return self.seg.cut(text)
