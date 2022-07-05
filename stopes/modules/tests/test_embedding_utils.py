# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
from pathlib import Path

import numpy as np

from stopes.utils.embedding_utils import Embedding

from .test_modules_utils import generate_embedding, test_dim, test_dtype


def test_len(tmp_path: Path):
    outfile = tmp_path / "embedding.bin"
    test_data = generate_embedding(file=outfile)
    emb = Embedding(outfile, test_dim, dtype=test_dtype)
    assert len(emb) == test_data.shape[0]


def test_read_mmap(tmp_path: Path):
    outfile = tmp_path / "embedding.bin"
    test_data = generate_embedding(file=outfile)
    emb = Embedding(outfile, test_dim, dtype=test_dtype)

    with emb.open_for_read() as np_array:
        assert np_array.shape == test_data.shape
        assert np.array_equal(np_array, test_data)


def test_read_memory(tmp_path: Path):
    outfile = tmp_path / "embedding.bin"
    test_data = generate_embedding(file=outfile)
    emb = Embedding(outfile, test_dim, dtype=test_dtype)

    with emb.open_for_read(mode="memory") as np_array:
        assert np_array.shape == test_data.shape
        assert np.array_equal(np_array, test_data)
