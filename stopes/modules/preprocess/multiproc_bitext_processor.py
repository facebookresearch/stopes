# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import shlex
import subprocess
import typing as tp
from abc import abstractmethod
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import hydra
from joblib import Parallel, delayed
from omegaconf import MISSING, DictConfig

from stopes.core import utils
from stopes.core.stopes_module import Requirements, StopesModule
from stopes.modules.preprocess.bitext_processor import (
    BitextProcessorCallback,
    BitextProcessorConfig,
)
from stopes.pipelines.filtering.dataset import Dataset, DatasetReader
from stopes.pipelines.monolingual.utils import slurm_tmp_maybe


class MultiprocBitextProcessorCallback(BitextProcessorCallback):
    """
    beware: line numbers will only be local to a chunk, not global to your file
    use self.offset_start in combination to know in what block that line number is.
    If the offset is None, you are in the merge mode
    """

    def __init__(
        self,
        outfile_prefix: str,
        src_input_file: str,
        tgt_input_file: str,
        input_files_idx: int,
        output_dir: str,
        line_offset_start: tp.Optional[int] = None,
        line_offset_end: tp.Optional[int] = None,
        outfile_postfix: str = "",
        merging: bool = False,
    ) -> None:
        super().__init__(
            outfile_prefix=outfile_prefix,
            outfile_postfix=outfile_postfix,
            src_input_file=src_input_file,
            tgt_input_file=tgt_input_file,
            input_files_idx=input_files_idx,
            output_dir=output_dir,
        )
        self.line_offset_start = line_offset_start
        self.line_offset_end = line_offset_end
        self.merging = merging

    @abstractmethod
    def merge_results(self, splits: tp.List[tp.Any]) -> tp.Any:
        """
        Multiple version of this CB will be used concurently to process splits of the original input. At the end
        we need to merge (reduce) the list of parallel results into a single results. The CB is responsible to do that.

        A separate instance will be created to do this.

        Takes a list of results from a single threaded LineProcessorCallback and merge them.
        """
        pass


def find_line_offsets(filename: str, num_chunks: int) -> tp.List[int]:
    """
    given a file and a number of chuncks, find the line offsets in the file
    to be able to chunk around full lines.
    """
    total_file_lines = utils.count_lines(filename)

    chunk_size = total_file_lines // num_chunks
    offsets = [chunk_size * x for x in range(num_chunks + 1)]
    offsets[-1] = total_file_lines

    return offsets


def _process_file_chunk(
    config: BitextProcessorConfig,
    src_input_file: str,
    tgt_input_file: str,
    input_files_idx: int,
    output_dir: str,
    line_offset_start: int,
    line_offset_end: int,
    corpus_name="",
):
    processor: MultiprocBitextProcessorCallback = hydra.utils.instantiate(
        config.bitext_processor,
        outfile_prefix=config.outfile_prefix,
        src_input_file=src_input_file,
        tgt_input_file=tgt_input_file,
        input_files_idx=input_files_idx,
        output_dir=output_dir,
        line_offset_start=line_offset_start,
        line_offset_end=line_offset_end,
        merging=False,
    )

    dataset = Dataset(src=src_input_file, tgt=tgt_input_file)
    with DatasetReader(
        dataset, corpus_name, line_offset_start, line_offset_end
    ) as inputs:
        with processor as enc:
            enc.process_lines(inputs)
    return processor.final_result()


@dataclass
class MultiprocBitextProcessorConfig(BitextProcessorConfig):
    tmp_dir: str = MISSING


class MultiprocBitextProcessorModule(StopesModule):
    def __init__(
        self,
        config: MultiprocBitextProcessorConfig = MultiprocBitextProcessorConfig(),
        processed_lines=0,
    ):
        super().__init__(config, MultiprocBitextProcessorConfig)
        self.config: MultiprocBitextProcessorConfig

    def array(self):
        return self.config.shards

    def requirements(self):
        reqs = self.config.requirements
        if not isinstance(reqs, Requirements):
            # Performe conversion if needed
            return Requirements(**reqs)
        return reqs

    def name(self):
        return getattr(
            self.config,
            "custom_name",
            "_".join(
                [
                    "MultiBitextProc",
                    self.config.bitext_processor._target_,
                    self.sha_key(),
                ]
            ),
        )

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        src_input_file = Path(iteration_value[0])
        tgt_input_file = Path(iteration_value[1])
        assert src_input_file.exists(), f"{src_input_file} does not exist"
        assert tgt_input_file.exists(), f"{tgt_input_file} does not exist"

        reqs = self.requirements()
        num_workers = reqs.cpus_per_task

        tmp_dir = slurm_tmp_maybe(Path(self.config.tmp_dir))

        decompressed_tmp = {}
        input_files = [src_input_file, tgt_input_file]

        read_from = None
        for idx, input_file in enumerate(input_files):
            decompressed_output = utils.expand_if_compressed(input_file, tmp_dir)
            if decompressed_output:
                decompressed_tmp[idx] = decompressed_output
                line_offsets = find_line_offsets(str(decompressed_tmp[0]), num_workers)
                read_from = decompressed_tmp
        if read_from is None:
            line_offsets = find_line_offsets(str(input_files[0]), num_workers)
            read_from = input_files

        # find_line_offsets returns a list of line numbers [line1, line2, line3, line4] but we would want pairs:
        # [(line1, line2), (line2, line3), (line3, line4)] to process the chunks with start/end info
        # we zip the list with itself shifted by one to get all the pairs.

        file_chunks = list(zip(line_offsets, line_offsets[1:]))

        proc_cb = partial(
            _process_file_chunk,
            self.config,
            str(read_from[0]),
            str(read_from[1]),
            iteration_index,
            str(tmp_dir),
        )

        print(f"starting jobs, num_workers: {num_workers}")
        final_results = Parallel(n_jobs=num_workers)(
            [delayed(proc_cb)(start, end) for (start, end) in file_chunks]
        )

        merging_processor: MultiprocBitextProcessorCallback = hydra.utils.instantiate(
            self.config.bitext_processor,
            outfile_prefix=self.config.outfile_prefix,
            src_input_file=str(src_input_file),
            tgt_input_file=str(tgt_input_file),
            input_files_idx=iteration_index,
            output_dir=self.config.output_dir,
            line_offset_start=None,
            line_offset_end=None,
            merging=True,
        )

        print(f"merging")
        full_rez = merging_processor.merge_results(final_results)

        if decompressed_tmp:
            for idx, file in decompressed_tmp.items():
                file.unlink()

        return full_rez


############
# Testing
############
