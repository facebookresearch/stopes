# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

import numpy as np
import pytest

from stopes.modules.bitext.indexing import sample_embedding_module as sem
from stopes.utils.data_utils import DataConfig


@pytest.mark.parametrize("sample_size", [100, 1_000])
@pytest.mark.parametrize("splits", [1, 5])
@pytest.mark.parametrize("dataset_size", [5_000, 20_000])
def test_sample_embedding(
    tmpdir: Path, sample_size: int, splits: int, dataset_size: int
):
    dataset_paths = []
    embedding_size = 1024
    for i in range(splits):
        dataset_path = tmpdir / f"dataset_{i}.npy"
        dataset = i * np.ones(shape=(dataset_size, embedding_size))
        np.save(str(dataset_path), dataset)
        dataset_paths.append(str(dataset_path))
    dataconf = DataConfig(
        data_version="0",
        iteration=1,
        data_shard_dir=str(dataset_path),
        shard_type="",
        bname="dataset",
        shard_list=None,
        shard_glob=None,
        meta_glob=None,
        nl_file_template=None,
    )
    out_dir = tmpdir / "result"
    os.mkdir(out_dir)
    sampler = sem.SampleEmbeddingModule(
        sem.SampleEmbeddingModuleConfig(
            "fr",
            dataset_paths,
            data=dataconf,
            output_dir=str(out_dir),
            sample_size=sample_size,
        )
    )
    final_file = sampler.run()
    # check file is loadable
    loaded = np.load(final_file)
    # assert correct shape
    assert loaded.shape[1] == embedding_size
    assert sample_size - splits < loaded.shape[0] < sample_size + splits
