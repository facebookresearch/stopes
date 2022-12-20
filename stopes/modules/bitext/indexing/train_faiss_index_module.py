# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf.omegaconf import MISSING

from stopes.core.stopes_module import Requirements, StopesModule
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
    fp16: bool = True


class TrainFAISSIndexModule(StopesModule):
    def __init__(self, config: TrainFAISSIndexConfig = TrainFAISSIndexConfig()):
        super().__init__(config, TrainFAISSIndexConfig)
        self.lang_output_dir = Path(self.config.output_dir) / self.config.lang
        self.lang_output_dir.mkdir(parents=True, exist_ok=True)

        self.index_type = self.config.index_type

        logger.info(f"lang={self.config.lang}, " f"index type={self.index_type}")

    def requirements(self):
        return Requirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=1 if self.config.use_gpu else 0,
            cpus_per_task=self.config.num_cpu,
            timeout_min=1000,
            constraint="ib2",
        )

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        import faiss

        from stopes.modules.bitext.indexing.train_index import train_index

        logger = logging.getLogger("stopes.train_faiss_index")
        index_output_file = (
            self.lang_output_dir
            / f"{self.config.data.bname}.{self.index_type}.{self.config.lang}.train.idx"
        ).resolve()

        returned_index = train_index(
            self.config.embedding_file,
            self.index_type,
            self.config.embedding_dimensions,
            self.config.use_gpu,
            np.float16 if self.config.fp16 else np.float32,
        )
        if self.config.use_gpu:
            returned_index = faiss.index_gpu_to_cpu(returned_index)

        faiss.write_index(returned_index, str(index_output_file))

        logger.info(
            f"Trained index of type: {self.index_type} and lang: {self.config.lang}, can be found in output file: {index_output_file}"
        )

        return index_output_file

    def name(self):
        return f"index-train.{self.config.lang}.iteration_{self.config.data.iteration}"

    def version(cls):
        return "0.2"
