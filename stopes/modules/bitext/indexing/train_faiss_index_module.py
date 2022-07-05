# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import typing as tp
from dataclasses import dataclass

import faiss
import numpy as np
from omegaconf.omegaconf import MISSING

from stopes.core.stopes_module import DistributedRequirements, StopesModule
from stopes.core.utils import ensure_dir
from stopes.modules.bitext.indexing.train_index import train_index
from stopes.utils.data_utils import DataConfig

logger = logging.getLogger("stopes.train_faiss_index")


@dataclass
class TrainFAISSIndexConfig:
    lang: str = MISSING
    embedding_file: str = MISSING
    index_type: str = MISSING
    data: DataConfig = MISSING
    output_dir: str = "ts.index.iteration_${iteration}"
    num_cpu: int = 40
    embedding_dimensions: int = 1024
    use_gpu: bool = True
    fp16_storage: bool = True
    sample_shards: bool = False
    sample_sz: int = 40_000_000


class TrainFAISSIndexModule(StopesModule):
    def __init__(self, config: TrainFAISSIndexConfig = TrainFAISSIndexConfig()):
        super().__init__(config, TrainFAISSIndexConfig)
        self.lang_output_dir = os.path.join(self.config.output_dir, self.config.lang)
        ensure_dir(self.lang_output_dir)

        self.index_type = self.config.index_type

        logger.info(f"lang={self.config.lang}, " f"index type={self.index_type}")

    def requirements(self):
        return DistributedRequirements(
            nodes=1,
            # mem_gb=700,
            tasks_per_node=1,
            gpus_per_node=1 if self.config.use_gpu else 0,
            cpus_per_task=self.config.num_cpu,
            timeout_min=1000,
            constraint="ib2",
        )

    async def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        index_output_file = os.path.abspath(
            os.path.join(
                self.lang_output_dir,
                f"{self.config.data.bname}.{self.index_type}.{self.config.lang}.train.idx",
            )
        )

        returned_index = train_index(
            self.config.embedding_file,
            self.index_type,
            self.config.embedding_dimensions,
            self.config.use_gpu,
            np.float16 if self.config.fp16_storage else np.float32,
        )
        if self.config.use_gpu:
            returned_index = faiss.index_gpu_to_cpu(returned_index)

        faiss.write_index(returned_index, str(index_output_file))

        index_output_file_path = str(index_output_file)
        logger.info(
            f"Trained index of type: {self.index_type} and lang: {self.config.lang}, can be found in output file: {index_output_file_path}"
        )

        return index_output_file

    def name(self):
        return f"index-train.{self.config.lang}.iteration_{self.config.data.iteration}"

    def comment(self):
        return (
            f"Creating FAISS index using student encoder v{self.config.data.iteration}"
        )

    def version(cls):
        return "0.1"
