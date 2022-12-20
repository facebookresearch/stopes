# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import numpy as np
import pytest

from stopes.utils.embedding_utils import (
    Embedding,
    EmbeddingConcatenator,
    MissingHeaderError,
)

from .test_modules_utils import generate_n_saved_embeddings, generate_saved_embedding


def test_legacy_redirection(tmp_path: Path):
    outfile = tmp_path / "embedding.npy"
    test_data = generate_saved_embedding(file=outfile, legacy_mode=False)
    # overwrite data without header
    test_data.tofile(outfile)
    with pytest.raises(MissingHeaderError):
        Embedding(outfile)


def test_legacy_len(tmp_path: Path):
    outfile = tmp_path / "embedding.bin"
    test_data = generate_saved_embedding(file=outfile, legacy_mode=True)
    emb = Embedding(outfile)
    assert len(emb) == test_data.shape[0]


def test_legacy_read_mmap(tmp_path: Path):
    outfile = tmp_path / "embedding.bin"
    test_data = generate_saved_embedding(file=outfile, legacy_mode=True)
    emb = Embedding(outfile)

    with emb.open_for_read() as np_array:
        assert np_array.shape == test_data.shape
        assert np.array_equal(np_array, test_data)


def test_legacy_read_memory(tmp_path: Path):
    outfile = tmp_path / "embedding.bin"
    test_data = generate_saved_embedding(file=outfile, legacy_mode=True)
    emb = Embedding(outfile)
    with emb.open_for_read(mode="memory") as np_array:
        assert np_array.shape == test_data.shape
        assert np.array_equal(np_array, test_data)


def test_len(tmp_path: Path):
    outfile = tmp_path / "embedding.npy"
    test_data = generate_saved_embedding(file=outfile, legacy_mode=False)
    emb = Embedding(outfile)
    assert len(emb) == test_data.shape[0]


def test_read_mmap(tmp_path: Path):
    outfile = tmp_path / "embedding.npy"
    test_data = generate_saved_embedding(file=outfile, legacy_mode=False)
    emb = Embedding(outfile)

    with emb.open_for_read() as np_array:
        assert np_array.shape == test_data.shape
        assert np.array_equal(np_array, test_data)


def test_read_memory(tmp_path: Path):
    outfile = tmp_path / "embedding.npy"
    test_data = generate_saved_embedding(file=outfile, legacy_mode=False)
    emb = Embedding(outfile)
    with emb.open_for_read(mode="memory") as np_array:
        assert np_array.shape == test_data.shape
        assert np.array_equal(np_array, test_data)


@pytest.mark.parametrize("sample_fraction", [0, 0.25, 0.5])
@pytest.mark.parametrize("fp16", [True, False])
@pytest.mark.parametrize("read_mode", ["mmap", "memory"])
def test_save_embeddings(
    tmp_path: Path, sample_fraction: float, fp16: bool, read_mode: str
):
    outfile = tmp_path / "embedding.npy"
    test_data = generate_saved_embedding(file=outfile, legacy_mode=False)
    dtype = np.float16 if fp16 else np.float32
    emb = Embedding(outfile)
    copy_path = tmp_path / "emb_copy.npy"

    if sample_fraction == 0:
        sample = None
        final_result = test_data.astype(dtype)
    else:
        sample = np.random.choice(
            len(test_data), round(len(test_data) * sample_fraction), replace=False
        )
        final_result = test_data[
            sample,
        ]
    emb.save(copy_path, sample, fp16, mode=read_mode)
    assert np.array_equal(np.load(copy_path), final_result)


@pytest.mark.parametrize("fp16", [True, False])
def test_embed_concat(tmp_path: Path, fp16: bool):
    concat_path = tmp_path / "concat_path.npy"
    if fp16:
        dtype = np.float16
    else:
        dtype = np.float32
    array, paths = generate_n_saved_embeddings(
        tmp_path, dim=3, split_length=5, dtype=dtype
    )
    with EmbeddingConcatenator(concat_path, fp16) as concat:
        concat.append_files(paths)
    with Embedding(concat_path).open_for_read(mode="memory") as data:
        assert np.array_equal(data, array)


def test_encode_to_npy(tmp_path: Path):
    # TODO: add tests for EncodeToNPY
    pass


def test_create_header(tmp_path: Path):
    # TODO: add tests to create the header separately
    pass
