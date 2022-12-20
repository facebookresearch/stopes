# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import faiss
import numpy as np

from stopes.utils.embedding_utils import Embedding


def index_to_gpu(idx):
    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = True
    idx = faiss.index_cpu_to_all_gpus(idx, co=co)
    return idx


def train_index(
    train_embeddings: Path,
    idx_type: str,
    dim: int,
    gpu: bool = False,
    dtype=np.float32,
) -> faiss.Index:
    emb = Embedding(train_embeddings)
    idx = faiss.index_factory(dim, idx_type)
    if gpu:
        idx = index_to_gpu(idx)

    with emb.open_for_read(mode="memory") as data:

        # fp16 currently not supported by FAISS
        if dtype == np.float16:
            data = data.astype(np.float32)

        faiss.normalize_L2(data)
        idx.train(data)

    return idx
