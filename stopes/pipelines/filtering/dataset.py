#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Optional

from stopes.core.utils import open


@dataclass
class Dataset:
    """A dataset, represented as a set of paths.

    This can be:
    - a monolingual corpus, in which case only the `src` path is set;
    - a standard bitext corpus, in which case both `src` and `tgt` are set;
    - a mined TSV corpus, in which case only the `tsv` path is set."""

    src: Optional[str] = None
    tgt: Optional[str] = None
    tsv: Optional[str] = None

    def __post_init__(self):
        # mined
        if not self.src and not self.tgt and self.tsv:
            pass
        # bilingual
        elif self.src and self.tgt and not self.tsv:
            pass
        # monolingual
        elif self.src and not self.tgt and not self.tsv:
            pass
        else:
            raise ValueError(f"Invalid combination of paths {self}")


@dataclass
class DatasetLine:
    corpus: str  # we keep track of the corpus name as certain filters might need it
    src: str
    tgt: Optional[str] = None  # unset for monolingual corpora
    score: Optional[float] = None  # optional score for mined corpora and similar

    # Store LID probabilities for each sentence. Needed only for debug purposes.
    src_lid_prob: Optional[float] = None
    tgt_lid_prob: Optional[float] = None


class DatasetReader(ExitStack):
    """A reader for MT datasets.

    It can be used as a context manager and will abstract over reading from differet
    types of Dataset."""

    def __init__(self, dataset: Dataset, corpus: str):
        super().__init__()
        self.dataset = dataset
        self.corpus = corpus
        self.src_path = dataset.src
        self.tgt_path = dataset.tgt
        self.tsv_path = dataset.tsv
        if self.tsv_path:
            self.f_tsv = self.enter_context(open(self.tsv_path, "rt"))
        if self.src_path:
            self.f_src = self.enter_context(open(self.src_path, "rt"))
        if self.tgt_path:
            self.f_tgt = self.enter_context(open(self.tgt_path, "rt"))

    def __iter__(self):
        if self.tsv_path:
            for line in self.f_tsv:
                score, src, tgt = line.strip().split("\t")
                yield DatasetLine(
                    corpus=self.corpus,
                    score=float(score),
                    src=src.strip(),
                    tgt=tgt.strip(),
                )
        elif self.src_path and self.tgt_path:
            for src, tgt in zip(self.f_src, self.f_tgt):
                yield DatasetLine(corpus=self.corpus, src=src.strip(), tgt=tgt.strip())
        elif self.src_path:
            for src in self.f_src:
                yield DatasetLine(corpus=self.corpus, src=src.strip())
        else:
            raise ValueError("Invalid combination of paths: {self.dataset}")
