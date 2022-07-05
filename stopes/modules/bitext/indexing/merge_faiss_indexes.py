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
import submitit
from omegaconf.omegaconf import MISSING

from stopes.core.stopes_module import DistributedRequirements, StopesModule
from stopes.core.utils import ensure_dir
from stopes.utils.data_utils import DataConfig
from stopes.utils.mining_utils import get_cached_line_count

logger = logging.getLogger("stopes.merge_faiss_indexes")


@dataclass
class MergeFAISSIndexesConfig:
    indexes: tp.List[str] = MISSING
    lang: str = MISSING
    index_type: str = MISSING
    data: DataConfig = MISSING
    output_dir: str = "index.${data.data_version}"


class MergeFAISSIndexesModule(StopesModule):
    def __init__(
        self,
        config: MergeFAISSIndexesConfig = MergeFAISSIndexesConfig(),
        checkpoint_part: tp.Optional[str] = None,
        checkpoint_file_idx: tp.Optional[int] = None,
    ):
        super().__init__(config)
        self.final_merged_index = os.path.abspath(
            os.path.join(
                self.config.output_dir,
                f"{self.config.data.bname}.{self.config.index_type}.0-{len(self.config.indexes)-1}.{self.config.lang}.data.idx",
            )
        )
        self.checkpoint_part = (
            checkpoint_part if checkpoint_part is not None else self.config.indexes[0]
        )
        self.checkpoint_file_idx = (
            checkpoint_file_idx if checkpoint_file_idx is not None else 1
        )
        ensure_dir(self.config.output_dir)

    def requirements(self):
        return DistributedRequirements(
            nodes=1,
            # mem_gb=480,
            tasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=10,
            timeout_min=1000,
        )

    async def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        idx_merged = faiss.read_index(self.checkpoint_part)
        print(
            " - {:12d}  {:s}".format(
                idx_merged.ntotal, os.path.basename(self.checkpoint_part)
            )
        )
        for idx, fname in enumerate(
            self.config.indexes[self.checkpoint_file_idx :],
            start=self.checkpoint_file_idx,
        ):
            if os.path.isfile(fname):
                index = faiss.read_index(fname)
                print(" - {:12d}  {:s}".format(index.ntotal, os.path.basename(fname)))
                if idx_merged:
                    faiss.merge_into(idx_merged, index, True)
                    print(" - {:12d}  merged".format(idx_merged.ntotal))
                else:
                    idx_merged = index
            elif idx_merged is not None:
                print(" - {:12s}  {:s}".format("MISSING", os.path.basename(fname)))
            faiss.write_index(idx_merged, self.final_merged_index + ".part")
            self.checkpoint_file_idx = (
                idx + 1
            )  # [self.checkpoint_file_idx:] above is inclusive

        assert idx_merged.ntotal > 0, "Merged index is empty"
        print(" saving merged index into {:s}".format(self.final_merged_index))
        faiss.write_index(idx_merged, self.final_merged_index)

        return self.final_merged_index

    def name(self):
        return f"idx_merge.{self.config.lang}.0-{len(self.config.indexes)-1}"

    def comment(self):
        return "Merging FAISS indexes"

    def checkpoint(
        self, *args: tp.Any, **kwargs: tp.Any
    ) -> submitit.helpers.DelayedSubmission:
        return submitit.helpers.DelayedSubmission(
            MergeFAISSIndexesModule(
                config=self.config,
                checkpoint_part=self.final_merged_index + ".part",
                checkpoint_file_idx=self.checkpoint_file_idx,
            ),
            *args,
            **kwargs,
        )  # submits to requeuing

    def version(self):
        "0.5"

    def validate(
        self,
        output: tp.Any,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> bool:
        expected_line_count = get_cached_line_count(
            self.config.lang,
            self.config.data,
        )
        idx = faiss.read_index(output)
        assert (
            idx.ntotal == expected_line_count
        ), f"for {self.config.lang}, expected {expected_line_count} sentences in index after merge, but found {idx.ntotal}."
        return True
