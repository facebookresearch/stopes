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

from omegaconf import MISSING

from stopes.core.stopes_module import DistributedRequirements, StopesModule
from stopes.core.utils import ensure_dir, path_append_suffix
from stopes.modules.bitext.mining.mine_bitext_sentences_utils import Alignments  # noqa
from stopes.utils.data_utils import DataConfig
from stopes.utils.mining_utils import extract_shard_id

logger = logging.getLogger("mine_bitext_sentences")


@dataclass
class MineBitextSentencesConfig:
    src_lang: str = MISSING
    tgt_lang: str = MISSING

    src_text_files: tp.List[str] = MISSING
    src_meta_files: tp.List[str] = MISSING

    tgt_text_files: tp.List[str] = MISSING
    tgt_meta_files: tp.List[str] = MISSING

    alignment_file: str = MISSING  # the mined indexes & distances without npz extension
    data: DataConfig = MISSING
    output_dir: str = MISSING
    mine_threshold: float = 1.06
    score_max: float = 1.25
    dedup_bitexts: bool = True
    compress_output: bool = True


class MineBitextSentencesModule(StopesModule):
    def __init__(self, config):
        super().__init__(config, MineBitextSentencesConfig)
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
        out_base_name = os.path.abspath(
            os.path.join(
                self.config.output_dir,
                f"{self.config.src_lang}-{self.config.tgt_lang}"
                f".TH-{self.config.mine_threshold}",
            )
        )

        # TODO: ideally the .npz should not be hardcoded and we should be using
        # the filename given by the previous module directly

        # loading alignments from the previous step (mine_indexes)
        # also applies some filtering
        alignments = Alignments.from_npz(
            Path(f"{self.config.alignment_file}.npz"),
            self.config.mine_threshold,
            self.config.score_max,
        )

        # persisting the mined sentences & corresponding metadata to disk
        bitexts_tsv = path_append_suffix(Path(out_base_name), ".bitext.tsv")
        bimeta_tsv = path_append_suffix(Path(out_base_name), ".bimeta.tsv")
        if self.config.compress_output:
            bitexts_tsv = path_append_suffix(bitexts_tsv, ".gz")
            bimeta_tsv = path_append_suffix(bimeta_tsv, ".gz")
        alignments.save_texts(
            self.config.src_text_files,
            self.config.tgt_text_files,
            self.config.src_meta_files,
            self.config.tgt_meta_files,
            bitexts_tsv,
            bimeta_tsv,
            self.config.dedup_bitexts,
            logger,
        )

        return bitexts_tsv, bimeta_tsv

    def version(self):
        return "0.4"

    def name(self):
        return (
            f"mineD.{self.config.src_lang}-{self.config.tgt_lang}"
            f".TH-{self.config.mine_threshold}.sents"
        )
