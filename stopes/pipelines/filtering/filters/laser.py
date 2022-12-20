#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter
from typing import Optional

from stopes.pipelines.filtering.dataset import DatasetLine
from stopes.pipelines.filtering.filters.base import Filter, FilteringCounts


class LaserFilter(Filter):
    """Uses pre-computed LASER scores to filter bitext."""

    def __init__(self, threshold: float):
        self.threshold = threshold

    def filter_line(
        self, line: DatasetLine, counts: FilteringCounts
    ) -> Optional[DatasetLine]:
        if not line.score:
            return line
        if line.score < self.threshold:
            counts.laser_threshold += 1
            return None
        return line
