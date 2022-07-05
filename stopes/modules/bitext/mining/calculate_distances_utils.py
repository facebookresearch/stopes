# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import faiss
import numpy as np

from stopes.core.utils import path_append_suffix
from stopes.utils.embedding_utils import Embedding

DISTANCES_FILE_SUFFIX = ".dist"
INDICES_FILE_SUFFIX = ".idx"


# TODO: Unsure if we want to keep that separate or reuse train_index (train_index.py)
# The handling of the gpu is not 100% the same here (nprobe & shard args)
def index_to_gpu(idx: faiss.Index, gpu_type: str, nprobe: int) -> faiss.Index:
    """
    Applies special Faiss settings to run on GPU
    """
    # TODO: harmonize with train_index
    # TODO: split into 2 options (shard_index=True, index_use_fp16=True)
    # instead of inferring from the string
    co = faiss.GpuMultipleClonerOptions()
    if "fp16" in gpu_type:
        co.useFloat16 = True
    if "shard" in gpu_type:
        co.shard = True
    idx = faiss.index_cpu_to_all_gpus(idx, co=co)
    if nprobe > 0:
        faiss.GpuParameterSpace().set_index_parameter(idx, "nprobe", nprobe)
    return idx


def load_index(
    idx_name: Path, nprobe: int = 128, gpu_type=tp.Optional[str]
) -> faiss.Index:
    # TODO: have our own faiss utils that does the right things for our usecases
    """
    Loads a Faiss index in memory
    """
    idx = faiss.read_index(idx_name)

    if gpu_type:
        idx = index_to_gpu(idx, gpu_type, nprobe)

    return idx


def compute_distances(
    query_embeddings_file: Path,
    idx: faiss.Index,
    embedding_dimensions: int,
    embedding_dtype: tp.Union[tp.Type[np.float32], tp.Type[np.float16]],
    knn: int,
    normalize_query_embeddings: bool,
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Performs kNN search for embeddings embs on faiss index idx
    and returns neighbors/distances
    """

    # Limiting kNN to size of the index
    k = min(knn, idx.ntotal)

    # Loading the query embeddings from disk
    q_embs = Embedding(query_embeddings_file, embedding_dimensions, embedding_dtype)

    with q_embs.open_for_read(mode="memory") as data:
        if normalize_query_embeddings:
            faiss.normalize_L2(data)
        # Run knn search and save distances, indices
        # FAISS expects specific dtypes
        distances = np.empty((data.shape[0], k), dtype=np.float32)
        indices = np.empty((data.shape[0], k), dtype=np.int64)
        idx.search(data, k, distances, indices)

    return distances, indices


def save_to_disk(
    distances: np.ndarray,
    indices: np.ndarray,
    out_basename: Path,
    as_fp16: bool,
) -> tp.Tuple[Path, Path]:
    """
    Save results from knn search to disk in specified format
    """

    dists_out_filename = path_append_suffix(
        out_basename, f"{DISTANCES_FILE_SUFFIX}.npy"
    )
    with dists_out_filename.open(mode="wb") as file:
        if as_fp16:
            np.save(file, distances.astype(np.float16))
        else:
            np.save(file, distances.astype(np.float32))

    indices_out_filename = path_append_suffix(
        out_basename, f"{INDICES_FILE_SUFFIX}.npy"
    )
    with indices_out_filename.open(mode="wb") as file:
        # downcasting the int64 returned by FAISS to save on disk space
        # this prevents using indexes bigger than 4G sentences
        # TODO: add a config param to control this limit
        np.save(file, indices.astype(np.uint32))

    return (dists_out_filename, indices_out_filename)
