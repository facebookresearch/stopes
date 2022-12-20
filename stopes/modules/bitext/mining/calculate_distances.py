# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import math
import os
import typing as tp
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from omegaconf.omegaconf import MISSING

from stopes.core.stopes_module import Requirements, StopesModule
from stopes.core.utils import ensure_dir
from stopes.modules.bitext.mining.mine_bitext_indexes_utils import read_array_metadata
from stopes.utils.embedding_utils import Embedding


class DistanceType(Enum):
    src2tgt = "x2y"
    tgt2src = "y2x"


@dataclass
class CalculateDistancesConfig:
    lang: str = MISSING
    other_lang: str = MISSING  # mostly for logging
    lang_embeddings: tp.List[str] = MISSING  # list of embedding files
    distance_type: DistanceType = MISSING  # mostly for logging
    index_other_lang: str = MISSING  # "path/to/index"
    index_other_lang_type: str = MISSING  # type of the index

    output_dir: tp.Optional[
        str
    ] = None  # If None, will be set to dist.src-tgt.knn.numprobe.gpu_type

    num_probe: int = 128
    knn: int = 16
    gpu_type: str = "fp16-shard"
    gpu_memory_gb: int = 32
    save_dists_as_fp16: bool = True
    embedding_dimensions: int = 1024
    normalize_query_embeddings: bool = True
    fp16: bool = False
    batch_size: int = 8192


class CalculateDistancesModule(StopesModule):
    def __init__(
        self,
        config: CalculateDistancesConfig = CalculateDistancesConfig(),
    ):
        super().__init__(config, CalculateDistancesConfig)
        self.bi = "-".join(sorted([self.config.lang, self.config.other_lang]))
        self.output_dir = os.path.abspath(
            f"dist-{self.bi}.k{self.config.knn}.np{self.config.num_probe}"
            f".{self.config.gpu_type}"
            if self.config.output_dir is None
            else self.config.output_dir
        )
        ensure_dir(self.output_dir)

        fp16 = getattr(self.config, "fp16", False)
        self.embedding_dtype = np.float16 if fp16 else np.float32

        assert os.path.exists(
            self.config.index_other_lang
        ), f"index file missing: {self.config.index_other_lang}"
        index_size = os.path.getsize(self.config.index_other_lang) >> 30  # size in GB
        # TODO: add gpu_memory_gb to either a preset or this module's config
        num_gpu = math.ceil(index_size / self.config.gpu_memory_gb)
        # max to 8, min to 1
        self.num_gpu = 8 if num_gpu > 8 else 1 if num_gpu < 1 else num_gpu

    def requirements(self):
        return Requirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=self.num_gpu,
            cpus_per_task=10,
            timeout_min=500,
            constraint="volta32gb",
        )

    def array(self):
        return self.config.lang_embeddings

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,  # the embedding shard
        iteration_index: int = 0,
    ) -> tp.Tuple[Path, Path]:
        from stopes.modules.bitext.mining.calculate_distances_utils import (
            compute_distances,
            load_index,
            save_to_disk,
        )

        current_index = load_index(
            idx_name=self.config.index_other_lang,
            nprobe=self.config.num_probe,
            gpu_type=self.config.gpu_type,
            index_type=self.config.index_other_lang_type,
        )

        distances, indices = compute_distances(
            query_embeddings_file=iteration_value,
            idx=current_index,
            embedding_dimensions=self.config.embedding_dimensions,
            embedding_dtype=self.embedding_dtype,
            knn=self.config.knn,
            normalize_query_embeddings=self.config.normalize_query_embeddings,
            batch_size=self.config.batch_size,
            save_as_fp16=self.config.save_dists_as_fp16,
        )

        # persisting to disk
        out_base_name = (
            Path(self.output_dir)
            / f"{self.bi}.{self.config.distance_type.value}.{iteration_index:03d}"
        )

        return save_to_disk(
            distances,
            indices,
            out_base_name,
            self.config.save_dists_as_fp16,
        )

    def name(self):
        return (
            f"knn.{self.bi}.{self.config.distance_type.value}"
            f".x{len(self.config.lang_embeddings)}.k{self.config.knn}"
            f".np{self.config.num_probe}"
        )

    def comment(self):
        return "Calculating distances between embeddings and FAISS index"

    def version(self):
        return "0.7"

    def validate(
        self,
        output: tp.Any,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> bool:
        # output should be a tuple of Path
        dist_path, index_path = output

        # files should still exist
        assert (
            isinstance(dist_path, Path) and dist_path.exists()
        ), f"{dist_path} is missing"

        assert (
            isinstance(index_path, Path) and index_path.exists()
        ), f"{index_path} is missing"

        # sizes should match
        dist_meta = read_array_metadata(dist_path)
        index_meta = read_array_metadata(index_path)

        assert (
            dist_meta.nb_elements == index_meta.nb_elements
        ), f"{index_path} and {dist_path} sizes do not match"

        # should have the same size as the query embedding
        q_embs = Embedding(iteration_value)
        assert (
            len(q_embs) == dist_meta.nb_elements
        ), "dist/indices have different sizes from query embedding"

        return True
