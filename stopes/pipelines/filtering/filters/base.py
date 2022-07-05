#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import copy
from typing import Optional, Tuple

from stopes.pipelines.filtering.dataset import DatasetLine


class FilteringCounts:
    def __init__(
        self,
        total_before: int = 0,  # total examples before filtering
        total_after: int = 0,  # total examples after filtering
        empty: int = 0,  # num of examples filtered due to being empty
        min_len: int = 0,
        max_len: int = 0,
        max_len_ratio: int = 0,
        min_src_unique_ratio: int = 0,
        max_toxicity: int = 0,
        max_toxicity_difference: int = 0,
        lid_threshold: int = 0,
        laser_threshold: int = 0,  # num of examples filtered due to LASER score
        source_dedup: int = 0,
        target_dedup: int = 0,
        pair_dedup: int = 0,
    ):
        self.total_before = total_before
        self.total_after = total_after

        # LengthFilter
        self.empty = empty
        self.min_len = min_len
        self.max_len = max_len
        self.max_len_ratio = max_len_ratio
        self.min_src_unique_ratio = min_src_unique_ratio

        # ToxicityFilter
        self.max_toxicity = max_toxicity
        self.max_toxicity_difference = max_toxicity_difference

        # LidFilter
        self.lid_threshold = lid_threshold

        # LaserFilter
        self.laser_threshold = laser_threshold

        # DedupFilter
        self.pair_dedup = pair_dedup
        self.target_dedup = target_dedup
        self.source_dedup = source_dedup

    def __add__(self, other):
        return FilteringCounts(
            **{key: val + getattr(other, key) for key, val in self.__dict__.items()}
        )

    def __radd__(self, other):
        if other == 0:
            return self
        return other.__add__(self)


class Filter:
    def filter_line(
        self, line: DatasetLine, counts: FilteringCounts
    ) -> Optional[DatasetLine]:
        raise NotImplementedError
