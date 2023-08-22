#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import typing as tp

_MIN_NUM_WORDS = {
    "arb_Arab": 1,
    "cmn_Hans": 25,
    "cmn_Hant": 25,
    "eng_Latn": 10,
    "fra_Latn": 12,
    "rus_Cyrl": 5,
    "spa_Latn": 5,
    "tur_Latn": 1,
    "vie_Latn": 1,
}

_MAX_NUM_WORDS = {
    "arb_Arab": 10,
    "cmn_Hans": 50,
    "cmn_Hant": 50,
    "eng_Latn": 20,
    "fra_Latn": 25,
    "rus_Cyrl": 35,
    "spa_Latn": 50,
    "tur_Latn": 10,
    "vie_Latn": 10,
}


def get_num_words(text, lang):
    if lang in {"cmn_Hans", "cmn_Hant"}:
        # hack fix for code mix with Chinese
        return sum(
            [
                1 if re.fullmatch("[a-zA-Z0-9]+", tok) else len(tok)
                for tok in text.split()
            ]
        )
    return len(text.split())


def split_sentence(
    text,
    boundaries="!?.,:;，。",
    lang="eng_Latn",
    min_num_words=None,
    max_num_words=None,
) -> tp.List[str]:
    """Split sentences that are longer than a set limit at the specified boundary characters
    Note that this is a lot more aggressive than the the common sentence splitting that
    is done in NLP -- the fragments here will not necessarily be full sentences."""
    if boundaries == " ":
        sentences = text.split()
    else:
        # split by boundaries
        for p in boundaries:
            text = text.replace(f"{p}", f"{p}\t")
        sentences = [s.strip() for s in text.split("\t") if s.strip()]

    # merge short phrases
    i = 0
    while i < len(sentences):
        if len(sentences) == 1:
            return sentences
        num_words = get_num_words(sentences[i], lang)
        if num_words > (max_num_words or _MAX_NUM_WORDS[lang]):
            i += 1
        else:
            if i == 0:
                j = 1
            elif i == len(sentences) - 1:
                j = i - 1
            else:
                prev_num_words = get_num_words(sentences[i - 1], lang)
                next_num_words = get_num_words(sentences[i + 1], lang)
                j = i - 1 if prev_num_words < next_num_words else i + 1
            neighbor_num_words = get_num_words(sentences[j], lang)

            if num_words < (
                min_num_words or _MIN_NUM_WORDS[lang]
            ) or num_words + neighbor_num_words < (
                min_num_words or _MIN_NUM_WORDS[lang]
            ):  # merge with neighbor
                if j < i:
                    sentences[j] = sentences[j] + " " + sentences[i]
                else:  # i < j
                    sentences[j] = sentences[i] + " " + sentences[j]

                del sentences[i]
            else:
                i += 1

    return sentences
