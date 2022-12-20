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

import stopes.core
from stopes.core.stopes_module import Requirements, StopesModule
from stopes.core.utils import ensure_dir

logger = logging.getLogger("text_encoder")


class LineProcessorCallback(AbstractContextManager):
    """
    a line processor callback is responsible for processing a batch of lines and writing them to an output file.
    It's used as a context manager so that it can deal with opening/closing the output file resource properly.
    """

    def __init__(
        self,
        outfile_prefix: str,
        input_file: str,
        input_file_idx: int,
        output_dir: str,
        outfile_postfix: str = "",
    ) -> None:
        self.outfile_prefix = outfile_prefix
        self.outfile_postfix = outfile_postfix
        self.input_file = input_file
        self.input_file_idx = input_file_idx
        self.output_dir = output_dir
        super().__init__()

    @abstractmethod
    def process_lines(self, lines_with_number: tp.Iterator[tp.Tuple[int, str]]) -> None:
        """
        process a batch of lines and write them to an output_file the way you want.
        The input is an iterator of lines with their line number in the input file
        """
        pass

    @abstractmethod
    def final_result(self) -> tp.Any:
        """
        return whatever this callback created, probably a file name
        """
        pass


def buffered_read(
    fp: tp.TextIO, buffer_size: int
) -> tp.Generator[tp.List[tp.Tuple[int, str]], None, None]:
    buffer = []
    for line_num, src_str in enumerate(fp):
        buffer.append((line_num, src_str.strip()))
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []

    if len(buffer) > 0:
        yield buffer


@dataclass
class LineProcessorConfig:
    line_processor: tp.Any = MISSING
    output_dir: str = MISSING
    outfile_prefix: str = "embed"
    outfile_postfix: str = ""
    shards: tp.List[str] = MISSING
    buffer_size: int = 10_000
    requirements: Requirements = Requirements(
        nodes=1,
        tasks_per_node=1,
        gpus_per_node=0,
        cpus_per_task=4,
        timeout_min=120,
    )
    custom_name: str = ""


class LineProcessorModule(StopesModule):
    def __init__(
        self,
        config: LineProcessorConfig = LineProcessorConfig(),
        processed_lines=0,
        validate_config: bool = False,
    ):
        super().__init__(
            config,
            # TODO: always validate that config is a LineProcessorConfig
            # This is not possible currently because several config files add extra args
            # to make it easier to type the config
            config_class=LineProcessorConfig if validate_config else None,
        )
        # we do basic checkpointing with submitit Checkpointable which will store the state of this
        # callable. The basic idea here is to remember the last line processed
        self.processed_lines = processed_lines
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
        input_file: str = iteration_value  # type: ignore
        assert Path(input_file).exists(), f"input_file: {input_file} doesn't exist"

        kwargs = {
            "outfile_prefix": self.config.outfile_prefix,
            "input_file": input_file,
            "input_file_idx": iteration_index,
            "output_dir": self.config.output_dir,
        }
        if hasattr(self.config, "outfile_postfix"):
            kwargs["outfile_postfix"] = self.config.outfile_postfix

        processor: LineProcessorCallback = hydra.utils.instantiate(
            self.config.line_processor,
            **kwargs,
        )
        with stopes.core.utils.open(input_file) as filep:
            with processor as enc:
                for lines_with_numbers in buffered_read(filep, self.config.buffer_size):
                    enc.process_lines(lines_with_numbers)
                    # TODO For checkpointing, keep track of processed lines + slice initial buffer read
        # TODO: this may not point to an existing file, making cache validation
        # impossible. However, the cache validate function can be adjusted accordingly
        return processor.final_result()

    @classmethod
    def version(cls):
        return "0.2"

    def name(self) -> str:
        return getattr(self.config, "custom_name", None) or "_".join(
            [
                "LineProc",
                self.config.line_processor._target_.split(".")[-1],
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
