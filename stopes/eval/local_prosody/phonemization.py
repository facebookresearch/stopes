# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
This is a wrapper around the `phonemizer` package (essentially, around Espeak),
intended to speed it up by extensive caching.
"""

import logging
import typing as tp
from functools import lru_cache

from phonemizer.backend import BACKENDS, EspeakBackend

logger = logging.getLogger(__name__)

_espeak_backends: tp.Dict[str, EspeakBackend] = {}

# This mapping is for P0 expressive languages.
# For other languages, please consult BACKENDS["espeak"].supported_languages()
ESPEAK_LANG_REMAP = {
    "eng": "en-us",
    "en": "en-us",
    "spa": "es",
    "zho": "cmn",
    "zh": "cmn",
    "fra": "fr-fr",
    "fr": "fr-fr",
    "deu": "de",
    "ita": "it",
}

_PHONEMIZER_WORD_LANG_CACHE: tp.Dict[tp.Tuple[str, str], str] = {}


@lru_cache()
def _get_espeak_langs() -> tp.Dict[str, str]:
    """
    Get languages supported by ESPEAK.
    We cache it as function, because it is slow.
    We don't cache it as a global variable, because we don't want to call Espeak backend on import.
    """
    return BACKENDS["espeak"].supported_languages()


def cached_phonemize(texts, language):
    """
    Phonemize a list of words using the phonemizer package; pre-load a backend.
    The Mandarin phonemizer is going to complain about word number mismatch; please ignore it.
    For Mandarin, it counts each character as a word, which is not really the case.
    """
    # TODO: support more principled language id conversion
    # https://bootphon.github.io/phonemizer/cli.html#supported-languages
    ebk = BACKENDS["espeak"]
    if language in ESPEAK_LANG_REMAP:
        language = ESPEAK_LANG_REMAP[language]
    elif language in _get_espeak_langs():
        pass
    elif language[:2] in _get_espeak_langs():
        language = language[:2]
    elif language[:2] + "-" + language[:2] in _get_espeak_langs():
        language = language[:2] + "-" + language[:2]

    if language in _espeak_backends:
        backend = _espeak_backends[language]
    else:
        backend = ebk(language=language)
        _espeak_backends[language] = backend

    # Run the phonemization only on the uncached texts.
    # Because we normally phonemize individual words, the cache hit rate is going to be high.
    uncached_texts = list(
        {text for text in texts if (text, language) not in _PHONEMIZER_WORD_LANG_CACHE}
    )
    if uncached_texts:
        phonemized = [t.strip() for t in backend.phonemize(uncached_texts)]
        for input_text, result_phonemes in zip(uncached_texts, phonemized):
            _PHONEMIZER_WORD_LANG_CACHE[(input_text, language)] = result_phonemes
    result = [
        _PHONEMIZER_WORD_LANG_CACHE[(input_text, language)] for input_text in texts
    ]
    return result
