# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import regex
import sentencepiece

from stopes.pipelines.monolingual.utils.text_normalizer import replace_unicode_punct


class ToxicityList:
    """A toxicity list is a list of toxic token sequences."""

    def __init__(self, word_list_paths: tp.List[str], spm_path=None, add_space=True):
        self._split = regex.compile(r"(\p{P}|\p{S}|\p{Han})")
        self.toxicity: tp.Set[str] = set()
        if spm_path:
            self.sp = sentencepiece.SentencePieceProcessor(spm_path)
        else:
            self.sp = None
        self.add_space = add_space
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
        if self.sp:
            tok = self.sp.EncodeAsPieces(s)
        else:
            tok = self._split.sub(r" \1 ", s)
        # collapse multiple spaces
        tok = " ".join(tok.split())
        # add spaces before and after – this allows us to use fast substring matching
        # (via Python's `in` operator) without running the risk of getting false
        # positives due to sub-token matches.
        # example: `" yz " in " abc xyz "` is False
        if self.add_space:
            tok = " " + tok + " "
        return tok

    def toxicity_count(self, text: str) -> int:
        """Count the number of unique toxic words in the text"""
        return len(self.get_toxic_words(text))

    def get_toxic_words(self, text: str) -> tp.List[str]:
        """Extract all the occurrences of toxic words in the given text"""
        # maybe at some moment speedup it with pyahocorasick, if the toxicity lists become too long
        tokenized = self._tokenize(replace_unicode_punct(text))
        lower = tokenized.lower()
        words = set()
        for t in self.toxicity:
            if t in tokenized or t in lower:
                words.add(t.strip())
        return list(words)
