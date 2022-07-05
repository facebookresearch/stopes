#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path
from typing import Dict, Optional

from stopes.pipelines.filtering.dataset import DatasetLine
from stopes.pipelines.filtering.filters.base import Filter, FilteringCounts
from stopes.pipelines.filtering.utils import ngrams

logger = logging.getLogger(__name__)


class LengthFilter(Filter):
    def __init__(
        self,
        min_len: Optional[int],
        max_len: Optional[int],
        max_len_ratio: Optional[float],
        min_src_unique_ratio: Optional[float],
        length_factors: Dict[str, float],
        src_lang: str,
        tgt_lang: str,
    ):
        self.min_len = min_len
        self.max_len = max_len
        self.max_len_ratio = max_len_ratio
        self.min_src_unique_ratio = min_src_unique_ratio

        try:
            self.src_factor = length_factors[src_lang]
        except KeyError:
            logging.warning(f"Missing length factor for {src_lang}")
            self.src_factor = 1.0

        try:
            self.tgt_factor = length_factors[tgt_lang]
        except KeyError:
            logging.warning(f"Missing length factor for {tgt_lang}")
            self.tgt_factor = 1.0

    def filter_line(
        self, line: DatasetLine, counts: FilteringCounts
    ) -> Optional[DatasetLine]:
        # filter empty lines
        if not line.src or (line.tgt is not None and not line.tgt):
            counts.empty += 1
            return None

        if line.tgt is not None:
            if not line.tgt:
                counts.empty += 1
                return None

        if (
            self.min_len
            or self.max_len
            or self.max_len_ratio
            or self.min_src_unique_ratio
        ):
            # min len, max len
            src_len = max(1, len(line.src) * self.src_factor)
            if self.min_len is not None and src_len < self.min_len:
                counts.min_len += 1
                return None
            if self.max_len is not None and src_len > self.max_len:
                counts.max_len += 1
                return None
            # same as above, but for tgt if set
            if line.tgt is not None:
                tgt_len = max(1, len(line.tgt) * self.tgt_factor)
                if self.min_len is not None and tgt_len < self.min_len:
                    counts.min_len += 1
                    return None
                if self.max_len is not None and tgt_len > self.max_len:
                    counts.max_len += 1
                    return None
                # len ratio
                if self.max_len_ratio is not None:
                    ratio = (
                        src_len / tgt_len if src_len > tgt_len else tgt_len / src_len
                    )
                    if ratio > self.max_len_ratio:
                        counts.max_len_ratio += 1
                        return None

            # source minimum unique tokens
            # rationale: in neural BT the decoder will occasionally get stuck in a loop
            # and repeatedly produce the same sequence; this filter catches that
            if self.min_src_unique_ratio is not None:
                # the order is six for English, appropriately scaled for other langs
                order = min(1, 6 * self.src_factor)
                ngrms = ngrams(line.src, order=order)
                if len(set(ngrms)) / len(ngrms) < self.min_src_unique_ratio:
                    counts.min_src_unique_ratio += 1
                    return None

        return line
