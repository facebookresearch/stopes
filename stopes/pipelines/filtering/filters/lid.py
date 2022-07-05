#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fasttext

from stopes.pipelines.filtering.dataset import DatasetLine
from stopes.pipelines.filtering.filters.base import Filter, FilteringCounts


class LidFilter(Filter):
    def __init__(
        self,
        model_path: Path,
        default_threshold: float,
        thresholds: Dict[str, float],
        excluded_corpora: Optional[List[str]],
        excluded_languages: Optional[List[str]],
        src_lang: str,
        tgt_lang: Optional[str],
    ):
        self.src_threshold = thresholds.get(src_lang, default_threshold)
        self.tgt_threshold = thresholds.get(tgt_lang, default_threshold)
        self.excluded_corpora = excluded_corpora
        self.excluded_languages = excluded_languages or []
        self.lid = fasttext.load_model(model_path)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def filter_line(
        self, line: DatasetLine, counts: FilteringCounts
    ) -> Optional[DatasetLine]:
        # certain corpora may be excluded from LID
        if self.excluded_corpora and line.corpus in self.excluded_corpora:
            return line

        # store LID probs in DatasetLine
        lid_l, lid_p = self.lid.predict(line.src, k=-1)
        lid_probs = {lang[9:]: prob for lang, prob in zip(lid_l, lid_p)}
        line.src_lid_prob = lid_probs.get(self.src_lang, 0.0)
        if self.tgt_lang is not None:
            lid_l, lid_p = self.lid.predict(line.tgt, k=-1)
            lid_probs = {lang[9:]: prob for lang, prob in zip(lid_l, lid_p)}
            line.tgt_lid_prob = lid_probs.get(self.tgt_lang, 0.0)

        if self.src_threshold and self.src_lang not in self.excluded_languages:
            if line.src_lid_prob < self.src_threshold:
                counts.lid_threshold += 1
                return None
        if (
            self.tgt_lang is not None
            and self.tgt_threshold
            and self.tgt_lang not in self.excluded_languages
        ):
            if line.tgt_lid_prob < self.tgt_threshold:
                counts.lid_threshold += 1
                return None
        return line
