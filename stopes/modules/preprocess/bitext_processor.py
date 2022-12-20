# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from abc import abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

import hydra
from omegaconf import MISSING

from stopes.core.stopes_module import Requirements, StopesModule
from stopes.core.utils import ensure_dir
from stopes.pipelines.filtering.dataset import Dataset, DatasetLine, DatasetReader

logger = logging.getLogger("bitext_encoder")


class BitextProcessorCallback(AbstractContextManager):
    """
    a bitext processor callback is responsible for processing a batch of lines from two files and writing them to two output files.
    It's used as a context manager so that it can deal with opening/closing the output file resource properly.
    """

    def __init__(
        self,
        outfile_prefix: str,
        src_input_file: str,
        tgt_input_file: str,
        input_files_idx: int,
        output_dir: str,
        outfile_postfix: str = "",
    ) -> None:
        self.outfile_prefix = outfile_prefix
        self.outfile_postfix = outfile_postfix
        self.src_input_file = src_input_file
        self.tgt_input_file = tgt_input_file
        self.input_files_idx = input_files_idx

        self.output_dir = output_dir
        super().__init__()

    @abstractmethod
    def process_lines(
        self, dataset_reader: tp.Generator[DatasetLine, None, None]
    ) -> None:

        """
        process a batch of lines from two files and writes them to two output_files the way you want.
        The input are two iterators of lines with their line number in the input file
        """
        pass

    @abstractmethod
    def final_result(self) -> tp.Any:
        """
        return whatever this callback created, probably 2 file names
        """
        pass


@dataclass
class BitextProcessorConfig:
    bitext_processor: tp.Any = MISSING
    output_dir: str = MISSING
    outfile_prefix: str = "embed"
    outfile_postfix: str = ""
    shards: tp.List[tp.Any] = MISSING  # list of pairs of filenames
    # (it is actually tp.List[tp.Tuple[str, str]], see https://github.com/omry/omegaconf/issues/427)
    requirements: Requirements = Requirements(
        nodes=1,
        tasks_per_node=1,
        gpus_per_node=0,
        cpus_per_task=4,
        timeout_min=120,
    )
    custom_name: str = ""


class BitextProcessorModule(StopesModule):
    def __init__(
        self,
        config: BitextProcessorConfig = BitextProcessorConfig(),
        processed_lines=0,
        corpus_name="",
    ):
        super().__init__(config)
        # we do basic checkpointing with submitit Checkpointable which will store the state of this
        # callable. The basic idea here is to remember the last line processed
        self.processed_lines = processed_lines
        self.corpus_name = corpus_name
        ensure_dir(self.config.output_dir)

    def array(self):
        return self.config.shards

    def requirements(self):
        reqs = self.config.requirements
        if not isinstance(reqs, Requirements):
            # Performe conversion if needed
            return Requirements(**reqs)
        return reqs

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        src_input_file: str = iteration_value[0]  # type: ignore
        tgt_input_file: str = iteration_value[1]  # type: ignore

        assert Path(
            src_input_file
        ).exists(), f"input_file: {src_input_file} doesn't exist"
        assert Path(
            tgt_input_file
        ).exists(), f"input_file: {tgt_input_file} doesn't exist"

        kwargs = {
            "outfile_prefix": self.config.outfile_prefix,
            "src_input_file": src_input_file,
            "tgt_input_file": tgt_input_file,
            "input_files_idx": iteration_index,
            "output_dir": self.config.output_dir,
        }
        if hasattr(self.config, "outfile_postfix"):
            kwargs["outfile_postfix"] = self.config.outfile_postfix

        processor: BitextProcessorCallback = hydra.utils.instantiate(
            self.config.bitext_processor,
            **kwargs,
        )

        dataset = Dataset(src=src_input_file, tgt=tgt_input_file)
        with DatasetReader(dataset, self.corpus_name) as inputs:
            with processor as enc:
                enc.process_lines(inputs)
                # TODO For checkpointing, keep track of processed lines + slice initial buffer read
                # self.processed_lines += 1
        # TODO: this may not point to an existing file, making cache validation
        # impossible. However, the cache validate function can be adjusted accordingly
        return processor.final_result()

    @classmethod
    def version(cls):
        return "0.2"

    def name(self) -> str:
        return getattr(self.config, "custom_name", None) or "_".join(
            [
                "BitextProc",
                self.config.bitext_processor._target_.split(".")[-1],
                self.sha_key(),
            ]
        )

    # def checkpoint(
    #     self, *args: tp.Any, **kwargs: tp.Any
    # ) -> submitit.core.utils.DelayedSubmission:
    #     """Resubmits the same module with the same arguments"""
    #     return submitit.core.utils.DelayedSubmission(
    #         LineProcessorModule(
    #             config=self.config, processed_lines=self.processed_lines
    #         ),
    #         *args,
    #         **kwargs,
    #     )
