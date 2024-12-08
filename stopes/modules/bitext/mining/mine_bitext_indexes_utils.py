# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib
import logging
import typing as tp
from collections import Counter, namedtuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity

from stopes.eval.auto_pcp.audio_comparator import Comparator, get_model_pred
from stopes.utils.embedding_utils import Embedding

# TODO: maybe import that from calculate_distances_utils once merged or
# ideally, rely on the filenames returned by the module within global_mining_pipeline.py
DISTANCES_FILE_SUFFIX = ".dist"
INDICES_FILE_SUFFIX = ".idx"

# when requesting more neighbors than elements in the index, FAISS returns -1
INVALID_INDEX_VALUE = np.int32(-1)
INVALID_INDEX_REPLACEMENT = 0
INVALID_DISTANCES_REPLACEMENT = 2.0


# TODO: we should merge Alignments with that of the mine_sentences module,
# within a single util file once all ports have been merged
# most of the functions below could be integrated into an actual class
Alignments = namedtuple("Alignments", ["scores", "src_idx", "tgt_idx", "bwd_pos"])
ArrayMeta = namedtuple("ArrayMeta", ["nb_elements", "dtype"])


class Margin(Enum):
    RATIO = "ratio"
    DISTANCE = "distance"
    ABSOLUTE = "absolute"

    @classmethod
    def margin_fct(cls, margin_type):
        """
        given the distance between points (a) and the avg. distance between
        their neighbors in both directions (b), return the appropriate margin fct
        for more details see here: http://aclanthology.lst.uni-saarland.de/P19-1309.pdf
        """
        if margin_type == cls.RATIO.value:
            return lambda a, b: a / b
        elif margin_type == cls.DISTANCE.value:
            return lambda a, b: a - b
        elif margin_type == cls.ABSOLUTE.value:
            return lambda a, b: a
        raise NotImplementedError(f"Margin: {margin_type} is not yet supported")


# TODO: implement "intersection" mining type
class MineType(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    UNION = "union"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

    @classmethod
    def requires_x2y(cls, value):
        return True if value != MineType.BACKWARD.value else False

    @classmethod
    def requires_y2x(cls, value):
        return True if value != MineType.FORWARD.value else False


# TODO: this should probably be unified with calculate_distances too
@dataclass
class Neighbors:
    dists: np.ndarray
    indices: np.ndarray
    # arrays are organized with fwd scores/index in front, then backwards scores/index,
    # this is the index in the ndarray from which backwards data starts
    backward_starts_at: tp.Optional[int] = None
    blended_scores: tp.Optional[np.ndarray] = None

    @staticmethod
    def concat_neighbors(
        list_of_neighbors: tp.List[Neighbors],
    ) -> tp.Optional[Neighbors]:
        if len(list_of_neighbors) < 1:
            return None

        nb_elements = 0
        for n_tmp in list_of_neighbors:
            nb_elements += n_tmp.indices.shape[0]
        k = list_of_neighbors[0].indices.shape[1]

        out_n = Neighbors(
            np.empty((nb_elements, k), dtype=np.float32),
            np.empty((nb_elements, k), dtype=np.uint32),
        )

        pos = 0
        for n in list_of_neighbors:
            np.copyto(out_n.dists[pos : pos + n.dists.shape[0], :], n.dists[:, :k])
            np.copyto(
                out_n.indices[pos : pos + n.indices.shape[0], :], n.indices[:, :k]
            )
            pos += n.indices.shape[0]
        return out_n


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
    margin_type: str,
    mine_type: str,
    sort_neighbors: bool,
    logger: logging.Logger,
    src_aux_embeddings: tp.Optional[tp.List[str]] = None,
    tgt_aux_embeddings: tp.Optional[tp.List[str]] = None,
    alpha: float = 0.5,
    comparator_path: tp.Optional[Path] = None,
    symmetrize_comparator: bool = False,
) -> Alignments:
    logger.info(
        f"Processing distance files: load k_extract={k_extract} dists and compute global avg"
    )
    list_of_dists_x2y, avg_x2y = load_distances_and_compute_average(
        dists_x2y_files, k_trg, k_extract, mean_is_last
    )
    list_of_dists_y2x, avg_y2x = load_distances_and_compute_average(
        dists_y2x_files, k_src, k_extract, mean_is_last
    )

    logger.info(f"Processing index files: build k_extract={k_extract} neighbors")

    neighbors_x2y = (
        build_neighbors(
            indices_x2y_files,
            list_of_dists_x2y,
            avg_x2y,
            avg_y2x,
            k_extract,
            margin_type,
            sort_neighbors,
        )
        if MineType.requires_x2y(mine_type)
        else None
    )
    del list_of_dists_x2y

    neighbors_y2x = (
        build_neighbors(
            indices_y2x_files,
            list_of_dists_y2x,
            avg_y2x,
            avg_x2y,
            k_extract,
            margin_type,
            sort_neighbors,
        )
        if MineType.requires_y2x(mine_type)
        else None
    )
    del list_of_dists_y2x
    if src_aux_embeddings and tgt_aux_embeddings:
        comparator_model = Comparator.load(comparator_path) if comparator_path else None
        logger.info(f"Starting blended retrieval with alpha: {alpha}")
        neighbors_x2y, neighbors_y2x = blended_retrieval(
            neighbors_x2y,
            neighbors_y2x,
            src_aux_embeddings,
            tgt_aux_embeddings,
            alpha,
            comparator_model,
            symmetrize_comparator,
        )
    logger.info("Starting fastmax retrieval with threshold {:f}".format(threshold))
    fastmax_neighbors = fastmax_retrieval(neighbors_x2y, neighbors_y2x, threshold)

    # extracting bitexts
    # TODO: add dedup logic if we need it
    nb_total_bitexts = fastmax_neighbors.dists.shape[0]
    pos = 0
    dists = np.empty(nb_total_bitexts, dtype=np.float32)
    src_idx = np.empty(nb_total_bitexts, dtype=np.uint32)
    tgt_idx = np.empty(nb_total_bitexts, dtype=np.uint32)
    bwd_pos = []
    threshold_counter_for_log: Counter = Counter()

    # logging vars
    th_cuts = np.arange(threshold - 0.02, threshold + 0.02, 0.01)
    min = None
    max = None
    for score_index in np.argsort(-fastmax_neighbors.dists):
        source_index, target_index = fastmax_neighbors.indices[score_index]
        dists[pos] = fastmax_neighbors.dists[score_index]
        src_idx[pos] = source_index
        tgt_idx[pos] = target_index
        if score_index >= fastmax_neighbors.backward_starts_at:
            bwd_pos.append(pos)

        # tracking for logging
        if max is None or dists[pos] > max:
            max = dists[pos]
        if min is None or dists[pos] < min:
            min = dists[pos]
        for th in th_cuts:
            if dists[pos] >= th:
                threshold_counter_for_log[th] += 1

        pos += 1

    logger.info(f"Margin stats: min={min:.3f} max={max:.3f}")
    for th in th_cuts:
        logger.info(
            f"counted {threshold_counter_for_log[th]:d} with threshold {th:.2f}"
        )

    return Alignments(scores=dists, src_idx=src_idx, tgt_idx=tgt_idx, bwd_pos=bwd_pos)


def load_distances_and_compute_average(
    dists_files: tp.List[Path],
    k: int,
    k_extract: int,
    mean_is_last: bool,
) -> (tp.List[np.ndarray], np.ndarray):
    # compute avg from all k columns as a single array for all files
    # copy and return k_extract distances from each file as list of batches
    nb_elements = 0
    for file in dists_files:
        dists_metadata = read_array_metadata(file)
        nb_elements += dists_metadata.nb_elements
    all_avgs = np.empty(nb_elements, dtype=np.float32)

    pos = 0
    list_of_dists = []
    for dfile in dists_files:
        dists, avg = load_distances_and_compute_average_single_file(
            dfile, k, k_extract, mean_is_last
        )
        list_of_dists.append(dists)
        np.copyto(all_avgs[pos : pos + avg.shape[0]], avg)
        pos += avg.shape[0]
    return list_of_dists, all_avgs


def load_distances_and_compute_average_single_file(
    dists_path: str,
    k: int,
    k_extract: int,
    mean_is_last: bool,
) -> (np.ndarray, np.ndarray):
    # load mmap as read-only
    tmp = np.load(dists_path, mmap_mode="r")

    if tmp.dtype != np.float32:
        # note: float16->float32 cast requires copying all data to memory.
        #       Better/faster to avoid if possible.
        dists = tmp.astype(np.float32)
        if dists[np.isinf(dists)].size > 0:
            # data clean-up
            # assuming it occurs rarely, otherwise it may be more efficient to cache np.isinf mask
            dists[np.isinf(dists)] = INVALID_DISTANCES_REPLACEMENT
    else:
        dists = tmp
        if dists[np.isinf(dists)].size > 0:
            # rare data clean-up
            dists = tmp.astype(np.float32)
            dists[np.isinf(dists)] = INVALID_DISTANCES_REPLACEMENT

    # average distance
    if mean_is_last:
        avg = dists[:, k - 1]
    else:
        avg = dists.mean(axis=1)

    # raw distances
    extracted_dists = dists.copy()
    del dists

    # Normalization
    # dists = 2.0 - dists may cause OOM error for temp allocations
    # performing in-place operation dists = 2 - dists instead

    # `extracted_dists` contains L2^2 distances, and all vectors are normalized.
    # Under these assumptions, the correct formula for cosine is
    # cosine = (2 - L2^2) / 2, however the implementation below is
    # cosine = 2 - L2^2, (without the divisor). This doesn't affect the ranking
    # but absolute values don't correspond to cosine similarity, in particular
    # they are not in the range [0, 1].
    np.negative(extracted_dists, out=extracted_dists)
    np.add(extracted_dists, 2, out=extracted_dists)
    # And the same normalization for avg, as needed for neighbors score formula downstream
    np.negative(avg, out=avg)
    np.add(avg, 2, out=avg)

    return extracted_dists, avg


def build_neighbors(
    indices_files: tp.List[Path],
    list_of_dists: tp.List[np.ndarray],
    avg1: np.ndarray,
    avg2: np.ndarray,
    k_extract: int,
    margin_type: str,
    sort_neighbors: bool,
) -> tp.Optional[Neighbors]:
    n_files = len(indices_files)
    assert n_files == len(list_of_dists)
    pos = 0
    list_of_neighbors = []
    for i in range(n_files):
        n = build_neighbors_single_file(
            indices_files[i],
            list_of_dists[i],
            avg1,
            avg2,
            k_extract,
            pos,
            margin_type,
            sort_neighbors,
        )
        list_of_neighbors.append(n)
        pos += n.dists.shape[0]
    return Neighbors.concat_neighbors(list_of_neighbors)


def build_neighbors_single_file(
    path_ind1: str,
    dist1: np.ndarray,
    avg1: np.ndarray,
    avg2: np.ndarray,
    k_extract: int,
    pos: int,
    margin_type: str,
    sort_neighbors: bool = False,
):
    # load indices of k_extract columns
    ind1 = np.load(path_ind1, mmap_mode="r")

    # when we request more neighbors than there are entries in the index,
    # FAISS returns -1. This can happen with very low resource languages
    # TODO: trim k directly when this happens
    if len(ind1[ind1 == INVALID_INDEX_VALUE]) > 0:
        ind1 = ind1.copy()
        ind1[ind1 == INVALID_INDEX_VALUE] = INVALID_INDEX_REPLACEMENT

    avg_dist = (
        avg1[pos : pos + len(ind1)].reshape(-1, 1) + avg2[ind1]
    ) / 2.0  # average distances of neighbors in both directions

    # calculate margin-based distances
    dists = Margin.margin_fct(margin_type)(dist1, avg_dist)

    if sort_neighbors:
        # get indices of sorted dists in descending order
        sorted_indices = (-dists).argsort()
        # permute dists and indices accordingly accordingly to score
        dists = np.take_along_axis(dists, sorted_indices, axis=1)
        ind1 = np.take_along_axis(ind1, sorted_indices, axis=1)

    # return the "k" chosen neighbors
    return Neighbors(dists=dists[:, :k_extract], indices=ind1[:, :k_extract])


def get_cosine_similarity(src_emb: np.ndarray, neighbor_embs: np.ndarray) -> np.ndarray:
    assert (
        src_emb.shape[1] == neighbor_embs.shape[1]
    ), f"{src_emb.shape[1]} != {neighbor_embs.shape[1]}"
    src_embs = np.repeat(
        src_emb, neighbor_embs.shape[0], axis=0
    )  # repeat src embedding
    cosine_scores = cosine_similarity(src_embs, neighbor_embs).diagonal()
    return cosine_scores


def get_comparator_scores(
    src_emb: np.ndarray,
    neighbor_embs: np.ndarray,
    comparator_model: tp.Any,
    symmetrize_comparator: bool,
) -> np.ndarray:
    src_embs = np.repeat(src_emb, neighbor_embs.shape[0], axis=0)
    a = torch.from_numpy(src_embs).unsqueeze(1)  # restore depth dim
    b = torch.from_numpy(neighbor_embs).unsqueeze(1)
    res = get_comparator_preds(a, b, comparator_model, symmetrize_comparator)
    scores_softmax = softmax(res)
    return np.array(scores_softmax)


def get_comparator_preds(
    src_emb: torch.Tensor, tgt_emb: torch.Tensor, model: tp.Any, symmetrize: bool
):
    preds = (
        get_model_pred(
            model,
            src=src_emb[:, 0],
            mt=tgt_emb[:, 0],
            use_gpu=model.use_gpu,
            batch_size=1,
        )[:, 0]
        .cpu()
        .numpy()
    )
    if symmetrize:
        preds2 = (
            get_model_pred(
                model,
                src=tgt_emb[:, 0],
                mt=src_emb[:, 0],
                use_gpu=model.use_gpu,
                batch_size=1,
            )[:, 0]
            .cpu()
            .numpy()
        )
        preds = (preds2 + preds) / 2
    return preds


def get_sharded_embeddings(
    indices: list[int], embedding_list: tp.List[np.ndarray], emb_idx_map: dict
) -> np.ndarray:
    embeddings = []
    for index in indices:
        x, y = emb_idx_map[index]
        embedding = embedding_list[x][y]
        embeddings.append(embedding)
    return np.vstack(embeddings)


def get_aux_scores(
    src_emb: np.ndarray,
    neighbor_embs: np.ndarray,
    comparator_model: tp.Any,
    symmetrize_comparator: bool,
) -> np.ndarray:
    return (
        get_comparator_scores(
            src_emb, neighbor_embs, comparator_model, symmetrize_comparator
        )
        if comparator_model
        else get_cosine_similarity(src_emb, neighbor_embs)
    )


def get_blended_best_neighbor(
    neighbors: tp.Optional[Neighbors],
    src_embeddings: tp.List[np.ndarray],
    tgt_embeddings: tp.List[np.ndarray],
    src_emb_idx_map: dict,
    tgt_emb_idx_map: dict,
    alpha: float,
    comparator_model: tp.Optional[tp.Any] = None,
    symmetrize_comparator: bool = False,
) -> Neighbors:
    if not neighbors:
        return None
    predictions = []
    nbex = neighbors.dists.shape[0]
    for src_index in range(nbex):
        margin_scores = neighbors.dists[src_index]
        neighbor_inds = neighbors.indices[src_index].tolist()
        neighbor_embs = get_sharded_embeddings(
            neighbor_inds, tgt_embeddings, tgt_emb_idx_map
        )
        src_emb = get_sharded_embeddings([src_index], src_embeddings, src_emb_idx_map)
        aux_scores = get_aux_scores(
            src_emb, neighbor_embs, comparator_model, symmetrize_comparator
        )
        assert margin_scores.shape == aux_scores.shape
        blended_scores = alpha * margin_scores + (1 - alpha) * aux_scores
        blended_neighbor_idx = blended_scores.argmax()
        predictions.append(np.arange(blended_scores.shape[0]) == blended_neighbor_idx)
    preds = np.vstack(predictions)
    return Neighbors(
        dists=np.expand_dims(neighbors.dists[preds], axis=1),
        indices=np.expand_dims(neighbors.indices[preds], axis=1),
        blended_scores=np.expand_dims(np.asarray(blended_scores), axis=1),
    )


def get_emb_idx_map(embedding_list: tp.List[np.ndarray]) -> dict:
    cumsum = np.cumsum([len(emb) for emb in embedding_list])
    emb_idx_map = {
        j + (cumsum[i - 1] if i > 0 else 0): [i, j]
        for i, emb in enumerate(embedding_list)
        for j in range(len(emb))
    }
    return emb_idx_map


def blended_retrieval(
    neighbors_x2y: tp.Optional[Neighbors],
    neighbors_y2x: tp.Optional[Neighbors],
    src_aux_embeddings: tp.List[str],
    tgt_aux_embeddings: tp.List[str],
    alpha: float,
    comparator_model: tp.Any,
    symmetrize_comparator: bool = False,
) -> tp.Tuple[Neighbors, Neighbors]:
    """
    return neighbor with highest blend of both margin and auxiliary scores
    """
    with contextlib.ExitStack() as stack:
        # load memory-mapped embeddings
        src_embeddings = [
            stack.enter_context(Embedding(fname).open_for_read())
            for fname in src_aux_embeddings
        ]
        tgt_embeddings = [
            stack.enter_context(Embedding(fname).open_for_read())
            for fname in tgt_aux_embeddings
        ]
        # map global (combined) indices from FAISS index to sharded files
        src_emb_idx_map = get_emb_idx_map(src_embeddings)
        tgt_emb_idx_map = get_emb_idx_map(tgt_embeddings)
        best_blended_neighbor_x2y = get_blended_best_neighbor(
            neighbors_x2y,
            src_embeddings,
            tgt_embeddings,
            src_emb_idx_map,
            tgt_emb_idx_map,
            alpha,
            comparator_model,
            symmetrize_comparator,
        )
        best_blended_neighbor_y2x = get_blended_best_neighbor(
            neighbors_y2x,
            tgt_embeddings,
            src_embeddings,
            tgt_emb_idx_map,
            src_emb_idx_map,
            alpha,
            comparator_model,
            symmetrize_comparator,
        )
    return best_blended_neighbor_x2y, best_blended_neighbor_y2x


def fastmax_retrieval(
    neighbors_x2y: Neighbors,
    neighbors_y2x: Neighbors,
    threshold: float,
) -> Neighbors:
    # TODO: rework this whole part to make it easier to read
    # TODO: harmonize naming with the mining pipeline (src2tgt and tgt2src)

    valid_candidates_fwd = (neighbors_x2y.dists >= threshold) if neighbors_x2y else 0
    valid_candidates_bwd = (neighbors_y2x.dists >= threshold) if neighbors_y2x else 0

    nb_candidates_fwd = np.count_nonzero(valid_candidates_fwd)
    nb_candidates_bwd = np.count_nonzero(valid_candidates_bwd)
    nb_candidates = nb_candidates_bwd + nb_candidates_fwd
    scores = np.empty(nb_candidates, dtype=np.float32)
    indices = np.empty((nb_candidates, 2), dtype=np.uint32)

    if nb_candidates_fwd > 0:
        # src_idx forward
        indices[:nb_candidates_fwd, 0] = np.where(valid_candidates_fwd)[0]
        # trg_idx forward
        indices[:nb_candidates_fwd, 1] = neighbors_x2y.indices[
            valid_candidates_fwd
        ].flatten()
        scores[:nb_candidates_fwd] = neighbors_x2y.dists[valid_candidates_fwd].flatten()

    if nb_candidates_bwd > 0:
        # src_idx backward
        indices[nb_candidates_fwd:nb_candidates, 0] = neighbors_y2x.indices[
            valid_candidates_bwd
        ].flatten()
        # trg_idx backward
        indices[nb_candidates_fwd:nb_candidates, 1] = np.where(valid_candidates_bwd)[0]
        scores[nb_candidates_fwd:nb_candidates] = neighbors_y2x.dists[
            valid_candidates_bwd
        ].flatten()

    return Neighbors(
        dists=scores, indices=indices, backward_starts_at=nb_candidates_fwd
    )


# TODO: could be put at a more global level e.g. to read in fp16 setting
def read_array_metadata(file: Path) -> ArrayMeta:
    with open(file, "rb") as shard:
        version = np.lib.format.read_magic(shard)
        shape, _fortran, dtype = np.lib.format._read_array_header(shard, version)

    return ArrayMeta(nb_elements=shape[0], dtype=dtype)
