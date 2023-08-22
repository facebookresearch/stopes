# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import regex

from stopes.pipelines.monolingual.utils.text_normalizer import replace_unicode_punct


class ToxicityList:
    """A toxicity list is a list of toxic token sequences."""

    def __init__(self, word_list_paths: tp.List[str]):
        self._split = regex.compile(r"(\p{P}|\p{S}|\p{Han})")
        self.toxicity: tp.Set[str] = set()
        for path in word_list_paths:
            with open(path, "rt") as fin:
                for line in fin:
                    line = line.strip()
                    tokenized = self._tokenize(line)
                    self.toxicity.add(tokenized)
                    self.toxicity.add(tokenized.lower())

    def _tokenize(self, s: str):
        "A very simple way of tokenizing for toxicity detection"
        s = replace_unicode_punct(s.strip())
        # split on punctuation, symbols and Han characters
        tok = self._split.sub(r" \1 ", s)
        # collapse multiple spaces
        tok = " ".join(tok.split())
        # add spaces before and after – this allows us to use fast substring matching
        # (via Python's `in` operator) without running the risk of getting false
        # positives due to sub-token matches.
        # example: `" yz " in " abc xyz "` is False
        return " " + tok + " "

    def toxicity_count(self, s: str):
        tokenized = self._tokenize(replace_unicode_punct(s))
        regular = sum(1 for t in self.toxicity if t in tokenized)
        lowercased = sum(1 for t in self.toxicity if t in tokenized.lower())
        return max(regular, lowercased)
