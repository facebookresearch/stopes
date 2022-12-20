# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from omegaconf import MISSING

from stopes.core.stopes_module import Requirements, StopesModule
from stopes.core.utils import ensure_dir
from stopes.modules.bitext.mining.mine_bitext_indexes_utils import MineType, mine


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
    margin_type: str = "ratio"
    mine_type: str = "union"
    sort_neighbors: bool = False
    margin_norm: str = "mean"
    num_probe: int = 128
    gpu_type: str = "fp16-shard"
    mine_threshold: float = 1.06
    requirements: Requirements = field(
        default=Requirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=40,
            timeout_min=600,
        )
    )


class MineBitextIndexesModule(StopesModule):
    def __init__(self, config):
        super().__init__(config, MineBitextConfig)
        ensure_dir(self.config.output_dir)
        assert MineType.has_value(
            self.config.mine_type
        ), f"mine type: {self.config.mine_type} not supported"

    def requirements(self):
        return self.config.requirements

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        logger = logging.getLogger("stopes.mine_bitext_indexes")

        assert all(
            [os.path.exists(f) for f in self.config.src2tgt_dist_files]
        ), "src2tgt distance file missing"
        assert all(
            [os.path.exists(f) for f in self.config.src2tgt_index_files]
        ), "src2tgt index file missing"
        assert all(
            [os.path.exists(f) for f in self.config.tgt2src_dist_files]
        ), "tgt2src distance file missing"
        assert all(
            [os.path.exists(f) for f in self.config.tgt2src_index_files]
        ), "tgt2src index file missing"

        # mining, extracting sentence indices
        scores, src_idx, trg_idx, bwd_pos = mine(
            self.config.src2tgt_dist_files,
            self.config.tgt2src_dist_files,
            self.config.src2tgt_index_files,
            self.config.tgt2src_index_files,
            self.config.src_k,
            self.config.tgt_k,
            self.config.k_extract,
            self.config.mine_threshold,
            self.config.margin_norm == "last",
            self.config.margin_type,
            self.config.mine_type,
            self.config.sort_neighbors,
            logger,
        )

        meta = Path(self.config.output_dir) / (
            f"{self.config.src_lang}-{self.config.tgt_lang}"
            f".{self.config.index_type}.k{self.config.src_k}-{self.config.tgt_k}"
            f".{self.config.margin_norm}.np{self.config.num_probe}"
            f".{self.config.gpu_type}.align.npz"
        )
        # persisting results to disk
        np.savez(
            meta,
            scores=scores,
            src_idx=src_idx,
            trg_idx=trg_idx,
            bwd_pos=bwd_pos,
        )
        logger.info(f"written results to {meta}")

        return meta.resolve()

    def version(self):
        return "0.3"

    def name(self):
        return (
            f"mineD.{self.config.src_lang}-{self.config.tgt_lang}"
            f".{self.config.margin_norm}.meta"
        )
