# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import tempfile
import unittest
from pathlib import Path

import faiss
import pytest

from stopes.modules.bitext.indexing.train_index import train_index

from .test_modules_utils import (
    generate_saved_embedding,
    test_dim,
    test_dtype,
    test_idx_type,
    test_lang,
)


def generate_train_index(
    dir_name: Path,
    use_gpu: bool,
    embedding_outfile: Path,
) -> Path:
    index_out_file = dir_name / f"{test_idx_type}.{test_lang}.train.idx"
    returned_index = train_index(
        embedding_outfile, test_idx_type, test_dim, use_gpu, test_dtype
    )

    if use_gpu:
        returned_index = faiss.index_gpu_to_cpu(returned_index)
    faiss.write_index(returned_index, str(index_out_file))

    return index_out_file


@pytest.mark.parametrize("gpu", [True, False])
def test_generate_index(tmp_path: Path, gpu: bool):
    if gpu:
        # get_num_gpus will crash if you installed faiss-gpu on a server without GPU.
        if os.environ.get("GITHUB_ACTIONS") or faiss.get_num_gpus() == 0:
            pytest.skip("no GPU")
    embedding_outfile = tmp_path / "embedding.npy"
    generate_saved_embedding(file=embedding_outfile)
    index_out_file = generate_train_index(tmp_path, gpu, embedding_outfile)
    assert index_out_file.exists(), f"index file {index_out_file} missing"
