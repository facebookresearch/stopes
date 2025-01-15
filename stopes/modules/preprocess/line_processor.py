# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import logging
import typing as tp
from abc import abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path

import hydra
from omegaconf import MISSING

import stopes.core
from stopes.core import Requirements, StopesModule, utils

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
        output_dir: Path,
        outfile_postfix: str = "",
    ) -> None:
        self.outfile_prefix = outfile_prefix
        self.outfile_postfix = outfile_postfix
        self.input_file = input_file
        self.input_file_idx = input_file_idx
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
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


@dataclass
class LineProcessorConfig:
    line_processor: tp.Any = MISSING
    output_dir: str = MISSING
    outfile_prefix: str = "embed"
    outfile_postfix: str = ""
    # shards is either a list of files or a glob string
    # if only hydra allowed, the right type would be tp.Union[str, tp.List[str]]
    shards: tp.Any = MISSING
    buffer_size: int = 10_000
    requirements: Requirements = field(
        default_factory=lambda: Requirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=4,
            timeout_min=120,
        )
    )
    custom_name: str = ""


class LineProcessorModule(StopesModule):
    def __init__(
        self,
        config: LineProcessorConfig = LineProcessorConfig(),
        processed_lines: int = 0,
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
        Path(config.output_dir).mkdir(exist_ok=True)

    def array(self) -> tp.List[str]:
        if isinstance(self.config.shards, str):
            return list(glob.glob(self.config.shards))
        return self.config.shards  # type: ignore[no-any-return]

    def requirements(self) -> Requirements:
        reqs = self.config.requirements
        if not isinstance(reqs, Requirements):
            # Performe conversion if needed
            return Requirements(**reqs)
        return reqs

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> tp.Any:
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

        # TODO: use StopesModule.build here to flatten the config
        processor: LineProcessorCallback = hydra.utils.instantiate(
            self.config.line_processor,
            **kwargs,
        )
        with stopes.core.utils.open(input_file) as f, processor as proc:
            for lines_with_numbers in utils.batch(
                enumerate(f), self.config.buffer_size
            ):
                proc.process_lines(lines_with_numbers)
                # TODO For checkpointing, keep track of processed lines + slice initial buffer read
        # TODO: this may not point to an existing file, making cache validation
        # impossible. However, the cache validate function can be adjusted accordingly
        return processor.final_result()

    @staticmethod
    def version() -> str:
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
