# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Different helper functions to interface with fairseq
#
import json
import os
import typing as tp
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import scipy.sparse as sp
import torch
from numba import njit


@dataclass
class VideoShard:
    sibling_id: str
    lang: str
    audio_path: str


@dataclass
class VideoShardPair:
    """
    Example :
        ShardPair(sibling_id='LdMx2U5tby0', lang1='fra', ts1='2023-07-19 03:14:12', lang2='eng', ts2='2023-07-19 03:06:44')
    """

    sibling_id: str
    lang1: str
    ts1: str
    lang2: str
    ts2: str


def pyarrow_fixed_size_array_to_numpy(cc: pa.ChunkedArray) -> np.ndarray:
    cc = cc.combine_chunks()
    assert cc.null_count == 0
    return np.reshape(cc.values, (-1, cc.type.list_size))


@njit
def _build_optimal_monotone_choice(sims: np.ndarray) -> np.ndarray:
    """
    First part of back tracing algo accelerating with nopython numba
    """
    cumulative_rewards = np.zeros_like(sims)
    choices = np.zeros_like(sims, dtype=np.int32)

    for i in range(sims.shape[0]):
        for j in range(0, sims.shape[1]):
            # option 1: align i to j
            best = sims[i, j]
            if i > 0 and j > 0:
                best += cumulative_rewards[i - 1, j - 1]
                choices[i, j] = 1
            # option 2: skip row i
            if i > 0 and cumulative_rewards[i - 1, j] > best:
                best = cumulative_rewards[i - 1, j]
                choices[i, j] = 2
            # option 3: skip column j
            if j > 0 and cumulative_rewards[i, j - 1] > best:
                best = cumulative_rewards[i, j - 1]
                choices[i, j] = 3
            cumulative_rewards[i, j] = best
    return choices


def get_strict_monotonic_alignment(sims: np.ndarray) -> np.ndarray:
    """Given a matrix of pairwise item similaritirs,
    return a list of pairs of strictly increasing indices that maximize the sum of similarities.
    Note that any number of indices can be skipped if they do not contribute positively.
    See :
    https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm
    """
    choices = _build_optimal_monotone_choice(sims)
    alignment = []
    i = sims.shape[0] - 1
    j = sims.shape[1] - 1
    while i > 0 and j > 0:
        if choices[i, j] == 1:
            alignment.append((i, j))
            i -= 1
            j -= 1
        elif choices[i, j] == 2:
            i -= 1
        else:
            j -= 1
    return np.array(alignment[::-1], dtype=np.int32)


@njit
def _build_intersection_matrix(
    start: np.ndarray, end: np.ndarray
) -> tp.List[tp.List[int]]:
    # assuming that both are sorted by "start"
    def has_intersection(ss1, ee1, ss2, ee2):
        return max(ss1, ss2) <= min(ee1, ee2)

    res = []
    for i, (ss, ee) in enumerate(zip(start, end)):
        adj = []
        for j in range(i, start.shape[0]):
            if start[j] >= ee:
                break
            if has_intersection(ss, ee, start[j], end[j]):
                adj.append(j)
        res.append(adj)
    return res


def build_segment_intersection_connection(
    start: np.ndarray, end: np.ndarray
) -> sp.csr_matrix:
    """
    Given arrays of segments starts and ends (assumed sorted by start),
    returns a sparse symmetric connectivity matrix :
        CC[i,j] = segment `i` intersects segment `j`
    """
    res = _build_intersection_matrix(start, end)
    indices = np.hstack(res)
    indptr = np.cumsum([0] + list(map(len, res)))
    data = np.ones_like(indices, dtype=np.bool_)
    adj_matrix = sp.csr_matrix((data, indices, indptr), shape=(len(res), len(res)))
    return (adj_matrix + adj_matrix.transpose()).tocsr(copy=False)


@njit
def _build_contained_matrix(
    start: np.ndarray, end: np.ndarray
) -> tp.List[tp.List[int]]:
    def contains_(ss1: int, ee1: int, ss2: int, ee2: int) -> bool:
        # seg1 contains seg2
        return ss1 <= ss2 and ee1 >= ee2

    # assuming that both are sorted by "start"
    res = []
    for i, (ss, ee) in enumerate(zip(start, end)):
        adj = []
        for j in range(i - 1, -1, -1):
            if end[j] <= ss:
                break
            if contains_(ss, ee, start[j], end[j]):
                adj.append(j)
        adj = adj[::-1]
        for j in range(i, start.shape[0]):
            if start[j] >= ee:
                break
            if contains_(ss, ee, start[j], end[j]):
                adj.append(j)

        res.append(adj)
    return res


def build_contains_matrix(start: np.ndarray, end: np.ndarray) -> sp.csr_matrix:
    """
    Given arrays of segments starts and ends (assumed sorted by start),
    returns a sparse symmetric connectivity matrix :
        CC[i,j] = segment `i` contains segment `j`
    """
    res = _build_contained_matrix(start, end)
    indices = np.hstack(res)
    indptr = np.cumsum([0] + list(map(len, res)))
    data = np.ones_like(indices, dtype=np.bool_)
    mat = sp.csr_matrix((data, indices, indptr), shape=(len(res), len(res)))
    return mat.sorted_indices()


def folder_size(folder_path: str) -> int:
    size = 0
    for path, _, files in os.walk(folder_path):
        for f in files:
            fp = os.path.join(path, f)
            size += os.stat(fp).st_size
    return size


def add_metadata_to_table(table: pa.Table, meta: dict) -> pa.Table:
    existing_metadata = table.schema.metadata or {}
    encoded_meta = {key: json.dumps(val) for key, val in meta.items()}
    combined_meta = {**existing_metadata, **encoded_meta}
    return table.replace_schema_metadata(combined_meta)


def generate_ngrams(words, max_depth, min_depth=1):
    return [
        (words[i], words[min(i + depth, len(words) - 1)])
        for depth in range(min_depth, max_depth)
        for i in range(len(words) - depth + 1)
    ]


def numpy_to_fixed_size_pyarrow_array(array: np.ndarray) -> pa.Array:
    return pa.FixedSizeListArray.from_arrays(array.ravel(order="C"), array.shape[1])


def wav_to_arrow_array(
    wav_segments: tp.List[torch.Tensor], bs: int = 1000
) -> pa.ChunkedArray:
    """
    chunk wav segments manually to avoid overflow
    """
    type_ = pa.large_list(pa.float32())
    chunks = [
        wav_segments[bs * i : bs * (i + 1)]
        for i in range(len(wav_segments) // bs + int(len(wav_segments) % bs != 0))
    ]
    ch_arr = pa.chunked_array(
        [[yy.numpy().astype("float32") for yy in xx] for xx in chunks], type=type_
    )
    return ch_arr
