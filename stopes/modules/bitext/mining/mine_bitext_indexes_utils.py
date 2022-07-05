# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from collections import namedtuple
from pathlib import Path

import numpy as np

# TODO: maybe import that from calculate_distances_utils once merged or
# ideally, rely on the filenames returned by the module within global_mining_pipeline.py
DISTANCES_FILE_SUFFIX = ".dist"
INDICES_FILE_SUFFIX = ".idx"

# when requesting more neighbors than elements in the index, FAISS returns -1
INVALID_INDEX_VALUE = np.uint32(-1)
INVALID_INDEX_REPLACEMENT = 0
INVALID_DISTANCES_REPLACEMENT = 2.0


# TODO: we should merge Alignments with that of the mine_sentences module,
# within a single util file once all ports have been merged
# most of the functions below could be integrated into an actual class
Alignments = namedtuple("Alignments", ["scores", "src_idx", "tgt_idx"])
ArrayMeta = namedtuple("ArrayMeta", ["nb_elements", "dtype"])


# TODO: this should probably be unified with calculate_distances too
class Neighbors:
    def __init__(self, dists: np.ndarray, indices: np.ndarray):
        self.dists = dists
        self.indices = indices


def mine(
    dists_x2y_files: tp.List[Path],
    dists_y2x_files: tp.List[Path],
    indices_x2y_files: tp.List[Path],
    indices_y2x_files: tp.List[Path],
    k_src: int,
    k_trg: int,
    k_extract: int,
    threshold: float,
    mean_is_last: bool,
    logger: logging.Logger,
) -> Alignments:
    # forward direction
    logger.info("Loading forward neighbors")
    neighbors_x2y = load_neighbors(dists_x2y_files, indices_x2y_files, k_trg)
    avg_x2y = compute_avg(neighbors_x2y, k_trg, k_extract, mean_is_last)

    # backward direction
    logger.info("Loading backward neighbors")
    neighbors_y2x = load_neighbors(dists_y2x_files, indices_y2x_files, k_src)
    avg_y2x = compute_avg(neighbors_y2x, k_src, k_extract, mean_is_last)

    # computing margin scores
    logger.info("Calculating forward margin scores")
    compute_margin_scores(neighbors_x2y, avg_x2y, avg_y2x)
    logger.info("Calculating backward margin scores")
    compute_margin_scores(neighbors_y2x, avg_y2x, avg_x2y)

    logger.info("Starting fastmax retrieval with threshold {:f}".format(threshold))
    fastmax_neighbors = fastmax_retrieval(
        neighbors_x2y, neighbors_y2x, k_extract, threshold
    )

    # extracting bitexts
    # TODO: add dedup logic if we need it
    nb_total_bitexts = fastmax_neighbors.dists.shape[0]
    pos = 0
    dists = np.empty(nb_total_bitexts, dtype=np.float32)
    src_idx = np.empty(nb_total_bitexts, dtype=np.uint32)
    tgt_idx = np.empty(nb_total_bitexts, dtype=np.uint32)
    for score_index in np.argsort(-fastmax_neighbors.dists):
        source_index, target_index = fastmax_neighbors.indices[score_index]
        dists[pos] = fastmax_neighbors.dists[score_index]
        src_idx[pos] = source_index
        tgt_idx[pos] = target_index
        pos += 1

    # TODO: make these logs optional with a verbose option, as the min and max
    # calculations could be taxing
    logger.info("Margin stats: min={:.3f} max={:.3f}".format(dists.min(), dists.max()))
    for th in np.arange(threshold - 0.02, threshold + 0.02, 0.01):
        nb_bitexts = src_idx[dists >= th].shape[0]
        logger.info("Mined {:d} bitexts with threshold {:.2f}".format(nb_bitexts, th))

    return Alignments(scores=dists, src_idx=src_idx, tgt_idx=tgt_idx)


# TODO: could be put at a more global level e.g. to read in fp16 setting
def read_array_metadata(file: Path) -> ArrayMeta:
    with open(file, "rb") as shard:
        version = np.lib.format.read_magic(shard)
        shape, _fortran, dtype = np.lib.format._read_array_header(shard, version)

    return ArrayMeta(nb_elements=shape[0], dtype=dtype)


def load_neighbors(
    dists_files: tp.List[Path],
    indices_files: tp.List[Path],
    k: int,
) -> Neighbors:
    # getting final shape size by reading in header from npy files
    nb_elements = 0
    for file in dists_files:
        dists_metadata = read_array_metadata(file)
        nb_elements += dists_metadata.nb_elements

    # loading all distances
    dists = np.empty((nb_elements, k), dtype=np.float32)
    pos = 0
    for file in dists_files:
        tmp = np.load(file, mmap_mode="r")
        if tmp.dtype != np.float32:
            np.copyto(dists[pos : pos + tmp.shape[0], :], tmp[:, :k].astype(np.float32))
        else:
            np.copyto(dists[pos : pos + tmp.shape[0], :], tmp[:, :k])
        pos += tmp.shape[0]

    # loading all indices
    # TODO: simplify the logic and ensure that dist/idx are in sync
    indices = None
    pos = 0
    for file in indices_files:
        tmp = np.load(file, mmap_mode="r")
        if indices is None:
            k_on_file = tmp.shape[1]
            indices = np.empty((nb_elements, k_on_file), dtype=np.uint32)

        if tmp.dtype != np.uint32:
            np.copyto(
                indices[pos : pos + tmp.shape[0], :],
                tmp[:, :k_on_file].astype(np.uint32),
            )
        else:
            np.copyto(indices[pos : pos + tmp.shape[0], :], tmp[:, :k_on_file])
        pos += tmp.shape[0]

    # when we request more neighbors than there are entries in the index,
    # FAISS returns -1. This can happen with very low resource languages
    # TODO: trim k directly when this happens
    indices[indices == INVALID_INDEX_VALUE] = INVALID_INDEX_REPLACEMENT
    dists[np.isinf(dists)] = INVALID_DISTANCES_REPLACEMENT

    # Normalizing distances
    # dists = 2.0 - dists may cause OOM error for temp allocations
    # performing in-place operation dists = 2 - dists instead
    np.negative(dists, out=dists)
    np.add(dists, 2, out=dists)

    return Neighbors(dists=dists, indices=indices)


def compute_avg(
    neighbors: Neighbors, k: int, k_extract: int, mean_is_last: bool
) -> np.ndarray:
    # this function modifies neighbors

    # computing the average
    if mean_is_last:
        avg = neighbors.dists[:, k - 1]
    else:
        avg = neighbors.dists.mean(axis=1)

    # now we don't need the full arrays anymore, saving some memory
    # TODO: maybe use resize instead of this to remove the last columns inplace
    neighbors.dists = neighbors.dists[:, :k_extract].copy()
    neighbors.indices = neighbors.indices[:, :k_extract].copy()

    return avg


def compute_margin_scores(
    neighbors: Neighbors,
    avg_x2y: np.ndarray,
    avg_y2x: np.ndarray,
):
    # the dists array is modified in place
    scores = avg_x2y.reshape(-1, 1) + avg_y2x[neighbors.indices]
    np.divide(scores, 2.0, out=scores)
    np.divide(neighbors.dists, scores, out=neighbors.dists)

    # TODO: for efficiency, move the view filter earlier
    # i.e. (dist / (scores / 2))[:, :1] -> dist[:, :1] / (scores / 2)
    neighbors.dists = neighbors.dists[:, :1]


def fastmax_retrieval(
    neighbors_x2y: Neighbors,
    neighbors_y2x: Neighbors,
    k_extract: int,
    threshold: float,
) -> Neighbors:
    # TODO: rework this whole part to make it easier to read
    # TODO: harmonize naming with the mining pipeline (src2tgt and tgt2src)
    valid_candidates_fwd = neighbors_x2y.dists[:, :k_extract] >= threshold
    valid_candidates_bwd = neighbors_y2x.dists[:, :k_extract] >= threshold
    nb_candidates_fwd = np.count_nonzero(valid_candidates_fwd)
    nb_candidates_bwd = np.count_nonzero(valid_candidates_bwd)

    nb_candidates = nb_candidates_bwd + nb_candidates_fwd
    indices = np.empty((nb_candidates, 2), dtype=np.uint32)

    # src_idx forward
    indices[:nb_candidates_fwd, 0] = np.where(valid_candidates_fwd)[0]
    # src_idx backward
    indices[nb_candidates_fwd:nb_candidates, 0] = neighbors_y2x.indices[:, :k_extract][
        valid_candidates_bwd
    ].flatten()
    # trg_idx forward
    indices[:nb_candidates_fwd, 1] = neighbors_x2y.indices[:, :k_extract][
        valid_candidates_fwd
    ].flatten()
    # trg_idx backward
    indices[nb_candidates_fwd:nb_candidates, 1] = np.where(valid_candidates_bwd)[0]

    scores = np.empty(nb_candidates, dtype=np.float32)
    scores[:nb_candidates_fwd] = neighbors_x2y.dists[:, :k_extract][
        valid_candidates_fwd
    ].flatten()
    scores[nb_candidates_fwd:nb_candidates] = neighbors_y2x.dists[:, :k_extract][
        valid_candidates_bwd
    ].flatten()

    return Neighbors(dists=scores, indices=indices)
