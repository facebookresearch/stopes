# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools
import math
import os
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from omegaconf.omegaconf import MISSING

from stopes.core.stopes_module import DistributedRequirements, StopesModule
from stopes.utils.data_utils import DataConfig
from stopes.utils.embedding_utils import Embedding


@dataclass
class SampleEmbeddingModuleConfig:
    lang: str = MISSING
    embedded_files: tp.List[str] = MISSING
    data: DataConfig = MISSING
    output_dir: str = MISSING
    embedding_dimensions: int = 1024
    fp16: bool = True
    sample_size: int = 40_000_000
    tmp_dir: str = "/tmp"
    max_num_workers: int = 40


def _sample_single(
    embedded_file: str,
    out_file: Path,
    embed_dim: int = 1024,
    fp16: bool = True,
    sample_size: int = 40_000_000,
) -> Path:
    dtype = np.float16 if fp16 else np.float32
    emb = Embedding(embedded_file, embed_dim, dtype)
    samples = (
        np.random.choice(len(emb), sample_size, replace=False)
        if len(emb) > sample_size
        else None  # only sample if file is over the sample_size
    )
    with emb.open_for_read(mode="memory") as data:
        with out_file.open("wb") as outfp:
            if samples is not None:
                data[samples].tofile(outfp)
            else:
                # otherwise take all the content
                data.tofile(outfp)
    return out_file


class SampleEmbeddingModule(StopesModule):
    def __init__(
        self, config: SampleEmbeddingModuleConfig = SampleEmbeddingModuleConfig()
    ):
        super().__init__(config, SampleEmbeddingModuleConfig)
        self.num_workers = max(
            int(self.config.max_num_workers),
            len(self.config.embedded_files),
        )

    def requirements(self):
        return DistributedRequirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=self.num_workers,
            timeout_min=24 * 60,
        )

    async def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        slurm_env = os.environ.get("SLURM_JOB_ID", None)
        tmp_dir = Path(self.config.tmp_dir)
        if slurm_env:
            tmp_dir = tmp_dir / slurm_env
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # TODO be smarter about shard size: The last shard is often smaller and
        # may not contain the sample_size? (later throwing an error in: np.random.choice()).
        # Maybe we could just use logic that if the last shard is < sample size,
        # take everything but in the case of two shards where the second shard is very
        #  small we might end up with close to sample_size/2.
        shard_sample_size = math.ceil(
            self.config.sample_size / len(self.config.embedded_files)
        )

        sample_cb = functools.partial(
            _sample_single,
            embed_dim=self.config.embedding_dimensions,
            fp16=self.config.fp16,
            sample_size=shard_sample_size,
        )

        sample_shards = joblib.Parallel(n_jobs=self.num_workers)(
            [
                joblib.delayed(sample_cb)(in_file, Path(tmp_dir) / f"shard_{idx}.npy")
                for idx, in_file in enumerate(self.config.embedded_files)
            ]
        )

        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"index_training_sample.{self.config.lang}"

        dtype = np.float16 if self.config.fp16 else np.float32

        with out_file.open("wb") as combined_fp:
            for f in sample_shards:
                emb = Embedding(f, self.config.embedding_dimensions, dtype)
                with emb.open_for_read(mode="memory") as in_data:
                    in_data.tofile(combined_fp)

        return str(out_file)

    def name(self):
        return f"sample_emb.{self.config.lang}-{len(self.config.embedded_files)}"
