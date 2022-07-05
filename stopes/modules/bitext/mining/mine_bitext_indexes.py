# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import MISSING

from stopes.core.stopes_module import DistributedRequirements, StopesModule
from stopes.core.utils import ensure_dir
from stopes.modules.bitext.mining.mine_bitext_indexes_utils import (  # noqa
    DISTANCES_FILE_SUFFIX,
    INDICES_FILE_SUFFIX,
    mine,
)
from stopes.utils.mining_utils import extract_shard_id

logger = logging.getLogger("mine_bitext_indexes")


@dataclass
class MineBitextConfig:
    src_lang: str = MISSING
    tgt_lang: str = MISSING

    src2tgt_dist_files: tp.List[str] = MISSING
    # index files is the list of indexes in the embedding, not
    #  the faiss index
    src2tgt_index_files: tp.List[str] = MISSING

    tgt2src_dist_files: tp.List[str] = MISSING
    # index files is the list of indexes in the embedding, not
    #  the faiss index
    tgt2src_index_files: tp.List[str] = MISSING

    index_type: str = MISSING

    output_dir: str = MISSING
    knn_dist: int = 16
    src_k: int = 16
    tgt_k: int = 16
    k_extract: int = 1
    margin: str = "ratio"
    margin_norm: str = "mean"
    num_probe: int = 128
    gpu_type: str = "fp16-shard"
    mine_threshold: float = 1.06


class MineBitextIndexesModule(StopesModule):
    def __init__(self, config):
        super().__init__(config, MineBitextConfig)
        ensure_dir(self.config.output_dir)

    def requirements(self):
        return DistributedRequirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=40,
            timeout_min=600,
        )

    async def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        # TODO: use Path to build path names
        out_base_name = os.path.abspath(
            os.path.join(
                self.config.output_dir,
                f"{self.config.src_lang}-{self.config.tgt_lang}"
                f".{self.config.index_type}.k{self.config.src_k}-{self.config.tgt_k}"
                f".{self.config.margin_norm}.np{self.config.num_probe}"
                f".{self.config.gpu_type}",
            )
        )

        # mining, extracting sentence indices
        scores, src_idx, trg_idx = mine(
            self.config.src2tgt_dist_files,
            self.config.tgt2src_dist_files,
            self.config.src2tgt_index_files,
            self.config.tgt2src_index_files,
            self.config.src_k,
            self.config.tgt_k,
            self.config.k_extract,
            self.config.mine_threshold,
            self.config.margin_norm == "last",
            logger,
        )

        # persisting results to disk
        meta = f"{out_base_name}.align"
        np.savez(meta, scores=scores, src_idx=src_idx, trg_idx=trg_idx)

        return meta

    def version(self):
        return "0.1"

    def name(self):
        return (
            f"mineD.{self.config.src_lang}-{self.config.tgt_lang}"
            f".{self.config.margin_norm}.meta"
        )
