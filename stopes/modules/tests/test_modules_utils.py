# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import numpy as np

from stopes.utils.embedding_utils import create_header

test_length = 10_000
test_dim = 16
test_dtype = np.float16
test_lang = "bn"
test_idx_type = "OPQ8,IVF16,PQ8"


def generate_embedding(
    emb_length: int = test_length,
    dim: int = test_dim,
    dtype=test_dtype,
) -> np.ndarray:
    data = np.random.randn(emb_length, dim).astype(dtype)
    return data


def generate_saved_embedding(
    file: tp.Optional[Path] = None,
    emb_length: int = test_length,
    dim: int = test_dim,
    dtype=test_dtype,
    legacy_mode: bool = False,
) -> np.ndarray:
    """Create ndarray and saved file"""
    data = generate_embedding(emb_length, dim, dtype)
    if file is not None:
        if legacy_mode:
            np.ascontiguousarray(data).tofile(file)
            create_header(file, data.shape, dtype)
        else:
            np.save(file, data)
    return data


def generate_n_saved_embeddings(
    dir: Path,
    dim: int = test_dim,
    split_length: int = test_length,
    splits: int = 3,
    dtype=test_dtype,
) -> tp.Tuple[np.ndarray, tp.List[Path]]:
    out_files = []
    filename_template = "shard_{idx}.npy"
    shards = []
    for i in range(splits):
        filename = dir / filename_template.format(idx=i)
        shard = generate_saved_embedding(filename, split_length, dim, dtype)
        out_files.append(filename)
        shards.append(shard)

    return np.vstack(shards), out_files
