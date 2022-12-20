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

from omegaconf import MISSING

from stopes.core.stopes_module import Requirements, StopesModule
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
    requirements: Requirements = field(
        default=Requirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=40,
            timeout_min=600,
        )
    )


class MineBitextSentencesModule(StopesModule):
    def __init__(self, config):
        super().__init__(config, MineBitextSentencesConfig)
        ensure_dir(self.config.output_dir)

    def requirements(self):
        return self.config.requirements

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        out_base_name = Path(self.config.output_dir) / (
            f"{self.config.src_lang}-{self.config.tgt_lang}"
            f".TH-{self.config.mine_threshold}"
        )

        # TODO: ideally the .npz should not be hardcoded and we should be using
        # the filename given by the previous module directly

        # loading alignments from the previous step (mine_indexes)
        # also applies some filtering
        alignment_file = Path(self.config.alignment_file)
        if alignment_file.suffix != ".npz":
            alignment_file = path_append_suffix(alignment_file, ".npz")

        alignments = Alignments.from_npz(
            alignment_file,
            self.config.mine_threshold,
            self.config.score_max,
        )

        # persisting the mined sentences & corresponding metadata to disk
        bitexts_tsv = path_append_suffix(out_base_name, ".bitext.tsv").resolve()
        bimeta_tsv = path_append_suffix(out_base_name, ".bimeta.tsv").resolve()
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
        return "0.6"

    def name(self):
        return (
            f"mineD.{self.config.src_lang}-{self.config.tgt_lang}"
            f".TH-{self.config.mine_threshold}.sents"
        )

    def validate(
        self,
        output: tp.Any,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> bool:
        (texts, meta) = output

        return Path(texts).exists() and (meta is None or Path(meta).exists())
