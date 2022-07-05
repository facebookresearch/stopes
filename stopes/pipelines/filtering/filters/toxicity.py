#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import List, Optional, Set, Tuple

import regex

from stopes.pipelines.filtering.dataset import DatasetLine
from stopes.pipelines.filtering.filters.base import Filter, FilteringCounts
from stopes.pipelines.monolingual.utils.text_normalizer import replace_unicode_punct


class ToxicityList:
    """A toxicity list is a list of toxic token sequences."""

    def __init__(self, word_list_paths: List[str]):
        self._split = regex.compile(r"(\p{P}|\p{S}|\p{Han})")
        self.toxicity: Set[str] = set()
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


class ToxicityFilter(Filter):
    def __init__(
        self,
        twl_path_template: str,
        eng_porn_twl_path: Optional[str],
        max_toxicity: Optional[int],
        max_toxicity_difference: Optional[int],
        src_lang: str,
        tgt_lang: Optional[str],
    ):
        self.max_toxicity = max_toxicity
        self.max_toxicity_difference = max_toxicity_difference
        self.tgt_toxicity_list: Optional[ToxicityList] = None
        self.src_toxicity_list: Optional[ToxicityList] = None

        # load src toxicity list
        src_paths = []
        src_twl_path = twl_path_template.format(lang=src_lang)
        if os.path.isfile(src_twl_path):
            src_paths.append(src_twl_path)
        # always concatenate the English pornography list
        if eng_porn_twl_path is not None:
            src_paths.append(eng_porn_twl_path)
        if src_paths:
            self.src_toxicity_list = ToxicityList(src_paths)

        # load tgt toxicity list
        tgt_paths = []
        if tgt_lang is not None:
            tgt_twl_path = twl_path_template.format(lang=tgt_lang)
            if os.path.isfile(tgt_twl_path):
                tgt_paths.append(tgt_twl_path)
        # always concatenate the English pornography list
        if eng_porn_twl_path is not None:
            tgt_paths.append(eng_porn_twl_path)
        if tgt_paths:
            self.tgt_toxicity_list = ToxicityList(tgt_paths)
        else:
            self.tgt_toxicity_list = None

    def filter_line(
        self, line: DatasetLine, counts: FilteringCounts
    ) -> Optional[DatasetLine]:
        if self.src_toxicity_list is not None:
            src_toxicity = self.src_toxicity_list.toxicity_count(line.src)
            if self.max_toxicity is not None and src_toxicity > self.max_toxicity:
                counts.max_toxicity += 1
                return None

        if line.tgt is not None and self.tgt_toxicity_list is not None:
            tgt_toxicity = self.tgt_toxicity_list.toxicity_count(line.tgt)
            if self.max_toxicity is not None and tgt_toxicity > self.max_toxicity:
                counts.max_toxicity += 1
                return None
            if (
                self.src_toxicity_list is not None
                and self.max_toxicity_difference is not None
            ):
                difference = abs(src_toxicity - tgt_toxicity)
                if difference > self.max_toxicity_difference:
                    counts.max_toxicity_difference += 1
                    return None
        return line
