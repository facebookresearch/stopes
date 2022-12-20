# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import submitit
from omegaconf.omegaconf import MISSING

from stopes.core.stopes_module import Requirements, StopesModule
from stopes.core.utils import ensure_dir, measure, path_append_suffix
from stopes.utils.data_utils import DataConfig


@dataclass
class MergeFAISSIndexesConfig:
    indexes: tp.List[str] = MISSING
    lang: str = MISSING
    expected_line_count = MISSING
    index_type: str = MISSING
    data: DataConfig = MISSING
    output_dir: str = "index.${data.data_version}"
    enable_checkpointing: bool = False


class MergeFAISSIndexesModule(StopesModule):
    def __init__(
        self,
        config: MergeFAISSIndexesConfig = MergeFAISSIndexesConfig(),
        checkpoint_part: tp.Optional[Path] = None,
        checkpoint_file_idx: tp.Optional[int] = None,
    ):
        super().__init__(config)
        self.final_merged_index = (
            Path(self.config.output_dir)
            / f"{self.config.data.bname}.{self.config.index_type}.merged.{self.config.lang}.data.idx"
        ).resolve()
        self.partial_merge_file = path_append_suffix(self.final_merged_index, ".part")
        self.checkpoint_part = (
            checkpoint_part
            if checkpoint_part is not None
            else Path(self.config.indexes[0])
        )
        self.checkpoint_file_idx = (
            checkpoint_file_idx if checkpoint_file_idx is not None else 1
        )
        ensure_dir(self.config.output_dir)

    def requirements(self):
        return Requirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=10,
            timeout_min=1000,
        )

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        import faiss

        logger = logging.getLogger("stopes.merge_faiss_indexes")
        idx_merged = faiss.read_index(str(self.checkpoint_part))
        total_expected = idx_merged.ntotal
        logger.info(f" - {idx_merged.ntotal:12d}  {self.checkpoint_part.name}")
        for idx, fname in enumerate(
            self.config.indexes[
                self.checkpoint_file_idx :
            ],  # we start with the first one already loaded
            start=self.checkpoint_file_idx,
        ):
            index_shard_file = Path(fname)
            with measure(f"adding {index_shard_file}", logger):
                if index_shard_file.is_file():
                    index_shard = faiss.read_index(str(index_shard_file))
                    total_expected += index_shard.ntotal
                    logger.info(f" - {index_shard.ntotal:12d}")
                    if idx_merged:
                        faiss.merge_into(idx_merged, index_shard, True)
                        logger.info(f" - {idx_merged.ntotal:12d}  merged")
                    else:
                        idx_merged = index_shard
                elif idx_merged is not None:
                    logger.info(f" - MISSING")

                if self.config.enable_checkpointing:
                    with measure("checkpointing", logger):
                        faiss.write_index(idx_merged, str(self.partial_merge_file))
                        self.checkpoint_file_idx = (
                            idx + 1
                        )  # [self.checkpoint_file_idx:] above is inclusive

        assert idx_merged.ntotal > 0, "Merged index is empty"
        assert (
            idx_merged.ntotal == total_expected
        ), f"Merged index has the wrong size {idx_merged.ntotal} vs {total_expected}"
        with measure(f" saving merged index into {self.final_merged_index}", logger):
            faiss.write_index(idx_merged, str(self.final_merged_index))

        return self.final_merged_index

    def name(self):
        return f"idx_merge.{self.config.lang}.0-{len(self.config.indexes)-1}"

    def checkpoint(
        self, *args: tp.Any, **kwargs: tp.Any
    ) -> submitit.helpers.DelayedSubmission:
        return submitit.helpers.DelayedSubmission(
            MergeFAISSIndexesModule(
                config=self.config,
                checkpoint_part=self.partial_merge_file
                if self.config.enable_checkpointing
                else None,
                checkpoint_file_idx=self.checkpoint_file_idx
                if self.config.enable_checkpointing
                else None,
            ),
            *args,
            **kwargs,
        )  # submits to requeuing

    def version(self):
        "0.6"

    def validate(
        self,
        output: tp.Any,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> bool:
        import faiss

        expected_line_count = self.config.expected_line_count
        idx = faiss.read_index(str(output))
        assert (
            idx.ntotal == expected_line_count
        ), f"for {self.config.lang}, expected {expected_line_count} sentences in index after merge, but found {idx.ntotal}."
        return True
