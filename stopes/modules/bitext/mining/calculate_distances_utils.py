# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from pathlib import Path

import faiss
import numpy as np

from stopes.core.utils import measure, path_append_suffix
from stopes.utils.embedding_utils import Embedding

DISTANCES_FILE_SUFFIX = ".dist"
INDICES_FILE_SUFFIX = ".idx"


# TODO: Unsure if we want to keep that separate or reuse train_index (train_index.py)
# The handling of the gpu is not 100% the same here (nprobe & shard args)
def index_to_gpu(
    idx: faiss.Index,
    gpu_type: str,
    nprobe: int,
    index_type: str,
) -> faiss.Index:
    """
    Applies special Faiss settings to run on GPU
    """
    logger = logging.getLogger("stopes.calculate_distances")
    # TODO: harmonize with train_index
    # TODO: split into 2 options (shard_index=True, index_use_fp16=True)
    # instead of inferring from the string
    logger.info(f" - transfering index to {faiss.get_num_gpus()} GPUs")
    with measure(" - done in ", logger):
        co = faiss.GpuMultipleClonerOptions()
        if "fp16" in gpu_type:
            co.useFloat16 = True
        if "shard" in gpu_type:
            co.shard = True
        idx = faiss.index_cpu_to_all_gpus(idx, co=co)
        if nprobe > 0 and index_type != "Flat":
            faiss.GpuParameterSpace().set_index_parameter(idx, "nprobe", nprobe)
    return idx


def load_index(
    idx_name: Path,
    index_type: str,
    nprobe: int = 128,
    gpu_type: tp.Optional[str] = None,
) -> faiss.Index:
    logger = logging.getLogger("stopes.calculate_distances")
    # TODO: have our own faiss utils that does the right things for our usecases
    """
    Loads a Faiss index in memory
    """
    logger.info(f"Reading FAISS index (version {faiss.__version__})")
    logger.info(f" - index: {idx_name}")

    idx = faiss.read_index(idx_name)

    logger.info(f" - found {idx.ntotal} sentences of dim {idx.d}")
    logger.info(f" - setting nbprobe to {nprobe}")

    if gpu_type:
        idx = index_to_gpu(
            idx=idx,
            gpu_type=gpu_type,
            nprobe=nprobe,
            index_type=index_type,
        )
    elif nprobe > 0 and index_type != "Flat":
        faiss.ParameterSpace().set_index_parameter(idx, "nprobe", nprobe)

    return idx


def compute_distances(
    query_embeddings_file: Path,
    idx: faiss.Index,
    embedding_dimensions: int,
    embedding_dtype: tp.Union[tp.Type[np.float32], tp.Type[np.float16]],
    knn: int,
    normalize_query_embeddings: bool,
    batch_size: int,
    save_as_fp16: bool,
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Performs kNN search for embeddings embs on faiss index idx
    and returns neighbors/distances
    """
    logger = logging.getLogger("stopes.calculate_distances")

    # Limiting kNN to size of the index
    k = min(knn, idx.ntotal)

    # Loading the query embeddings from disk
    q_embs = Embedding(query_embeddings_file)
    total_lines = len(q_embs)

    # final arrays for distances and indices
    distances = np.empty(
        (total_lines, k), dtype=np.float16 if save_as_fp16 else np.float32
    )
    # downcasting the int64 returned by FAISS to save on disk space
    # this prevents using indexes bigger than 4G sentences
    # TODO config?
    indices = np.empty((total_lines, k), dtype=np.uint32)

    logger.info(
        f"loading embedding {query_embeddings_file} fp{'fp16' if embedding_dtype == np.float16 else 'fp32'} to mmap"
    )
    logger.info(f"numpy version{np.__version__}")
    with measure("searched everyting in ", logger):
        with q_embs.open_for_read(mode="mmap") as data:
            # we open embedding as mmap and load a "batch" of data in memory
            data_buffer = np.empty((batch_size, embedding_dimensions), dtype=np.float32)
            lines_processed = 0
            logger.info(
                f" - {k:d}-nn search {total_lines:d} in {idx.ntotal:d} with shards {batch_size:d}"
            )

            while lines_processed < total_lines:
                with measure(
                    f" - searched {lines_processed}/{total_lines} in ", logger
                ):
                    if lines_processed + batch_size > total_lines:
                        # batch_size is bigger than available lines
                        # let's resize the buffer
                        batch_size = total_lines - lines_processed
                        data_buffer.resize((batch_size, embedding_dimensions))

                    start = lines_processed
                    end = lines_processed + batch_size

                    # load a slice of mmap into the buffer
                    # (faiss need the real data in memory)
                    if embedding_dtype == np.float16:
                        np.copyto(
                            data_buffer,
                            data[start:end, :].astype(np.float32),
                        )
                    else:
                        np.copyto(
                            data_buffer,
                            data[start:end, :],
                        )

                    if normalize_query_embeddings:
                        logger.info("normalizing")
                        faiss.normalize_L2(data_buffer)
                    # Run knn search and save distances, indices
                    logger.info("searching")
                    distances_batch, indices_batch = idx.search(data_buffer, k)

                    # fill up the final array with what we got back
                    # FAISS returns specific types, but we don't need to store that much
                    # in memory as we only write smaller types, so we downcast
                    distances[start:end, :] = (
                        distances_batch.astype(np.float16)
                        if save_as_fp16
                        else distances_batch
                    )  # already in fp32

                    indices[start:end, :] = indices_batch.astype(np.uint32)

                    lines_processed += batch_size

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
    logger = logging.getLogger("stopes.calculate_distances")

    dists_out_filename = path_append_suffix(
        out_basename, f"{DISTANCES_FILE_SUFFIX}.npy"
    )
    logger.info(
        f" - saving distances {dists_out_filename} {distances.shape[0]:d}x{distances.shape[1]:d} ({'fp16' if as_fp16 else 'fp32'})"
    )

    with dists_out_filename.open(mode="wb") as file:
        if as_fp16:
            np.save(file, distances.astype(np.float16))
        else:
            np.save(file, distances.astype(np.float32))

    indices_out_filename = path_append_suffix(
        out_basename, f"{INDICES_FILE_SUFFIX}.npy"
    )
    logger.info(
        f" - saving indexes {indices_out_filename} {indices.shape[0]:d}x{indices.shape[1]:d}"
    )

    with indices_out_filename.open(mode="wb") as file:
        # downcasting the int64 returned by FAISS to save on disk space
        # this prevents using indexes bigger than 4G sentences
        # TODO: add a config param to control this limit
        np.save(file, indices.astype(np.uint32))

    return (dists_out_filename, indices_out_filename)
