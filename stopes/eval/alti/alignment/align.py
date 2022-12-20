# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code was adapted from the repository https://github.com/mt-upc/transformer-contributions-nmt by Javier Ferrando.

""" Various utilities for computing word attributions and word alignment quality metrics."""

import itertools
import typing as tp
from string import punctuation

import numpy as np


def contrib_tok2words_partial(
    contributions: np.ndarray,
    tokens: tp.List[str],
    axis: int,
    reduction: str,
    keep_punctuation: bool = False,
) -> tp.Tuple[np.ndarray, tp.List[str]]:
    """Aggregate toking contributions along one of the axes by merging subtokens into words."""
    reduction_f = np.mean if reduction == "avg" else np.sum
    bpe_space = "▁"

    words = []
    w_contributions = []  # for each word, there is a list of vectors
    for counter, (tok, contrib) in enumerate(zip(tokens, contributions.T)):
        if (
            tok.startswith(bpe_space)
            or tok.startswith("__")
            or tok.startswith("<")
            or counter == 0
        ) or (tok in punctuation and keep_punctuation):
            if tok.startswith(bpe_space):
                tok = tok[1:]
            words.append(tok)
            w_contributions.append([contrib])
        else:
            words[-1] += tok
            w_contributions[-1].append(contrib)

    # concatenate the token vectors for each word
    word_matrices = [np.stack(contrib, axis=axis) for contrib in w_contributions]

    # reduce the words to single vectors and join them together into a matrix
    word_contrib = np.stack(
        [reduction_f(matrix, axis=axis) for matrix in word_matrices],
        axis=axis,
    )

    return word_contrib, words


def contrib_tok2words(
    contributions: np.ndarray,
    tokens_in: tp.List[str],
    tokens_out: tp.List[str],
    keep_punctuation: bool = False,
) -> tp.Tuple[np.ndarray, tp.List[str], tp.List[str]]:
    """Aggregate the token contributions matrix from subwords into words
    by adding them up for inputs and averaging them for outputs"""
    word_contrib, words_in = contrib_tok2words_partial(
        contributions,
        tokens_in,
        axis=0,
        reduction="sum",
        keep_punctuation=keep_punctuation,
    )
    word_contrib, words_out = contrib_tok2words_partial(
        word_contrib,
        tokens_out,
        axis=1,
        reduction="avg",
        keep_punctuation=keep_punctuation,
    )
    return word_contrib.T, words_in, words_out


def parse_single_alignment(
    text: str, reverse: bool = False, one_add: bool = False, one_indexed: bool = False
) -> tp.Tuple[int, int]:
    """
    Given an alignment (as a string such as "3-2" or "5p4"), return the index pair.
    """
    assert "-" in text or "p" in text

    a, b = text.replace("p", "-").split("-")
    a, b = int(a), int(b)

    if one_indexed:
        a = a - 1
        b = b - 1

    if one_add:
        a = a + 1
        b = b + 1

    if reverse:
        a, b = b, a

    return a, b


def compute_alignment_metrics(
    sure: tp.List[tp.Set], possible: tp.List[tp.Set], hypothesis: tp.List[tp.Set]
) -> tp.Tuple[float, float, float]:
    """Compute average alignment rate, precision and recall for alignment.
    Inputs are lists of alignments. All alignments are presented as sets of (tgt, src) pairs."""
    sum_a_intersect_p, sum_a_intersect_s, sum_s, sum_a = 0, 0, 0, 0

    for s, p, a in itertools.zip_longest(sure, possible, hypothesis):
        sum_a += len(a)
        sum_s += len(s)
        sum_a_intersect_p += len(a.intersection(p))
        sum_a_intersect_s += len(a.intersection(s))

    precision = sum_a_intersect_p / max(1, sum_a)
    recall = sum_a_intersect_s / max(1, sum_s)
    aer = 1.0 - ((sum_a_intersect_p + sum_a_intersect_s) / max(1, sum_a + sum_s))

    return aer, precision, recall
