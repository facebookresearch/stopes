# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import subprocess
import sys
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import MISSING

from stopes.core.stopes_module import Requirements, StopesModule
from stopes.core.utils import ensure_dir
from stopes.modules.preprocess import moses_cli_module

logger = logging.getLogger("generate_multi_bleu_detok_module")


class FairseqGenerateOutputFileEmpty(Exception):
    """
    Exception raised when fairseq generate output file is empty
    """

    def __init__(self, file_path: Path):
        message = "Output file from fairseq generate is empty. "
        self.file_path = file_path
        self.message = message + f"File path is: {self.file_path}"
        super().__init__(self.message)


@dataclass
class GenerateMultiBleuDetokConfig:
    src_lang: str = MISSING
    tgt_lang: str = MISSING
    checkpoint_dir: Path = MISSING
    binarized_dir: Path = MISSING  # contains binarized files for each lang and split
    output_dir: Path = MISSING
    checkpoint_glob: str = "checkpoint[0-9]*.pt"
    beam: int = 5
    batch_size: int = 128
    batch_memory: int = 2
    splits: tp.List[str] = field(default_factory=lambda: ["valid", "test"])


@dataclass
class JobConfig:
    """
    GenerateMultiBleuDetokModule's run function is called for each checkpoint_file_path and for each split
    JobConfig stores split and checkpoint_file_path for a current run / current job
    """

    split: str
    checkpoint_file_path: Path


class GenerateMultiBleuDetokModule(StopesModule):
    """
    This module calls generate.py and multi-blue-detok.perl script to generate bleu scores
    """

    def __init__(self, config: GenerateMultiBleuDetokConfig):
        super().__init__(config)
        self.output_dir = Path(self.config.output_dir).resolve()
        self.binarized_dir = Path(self.config.binarized_dir)

        ensure_dir(self.output_dir)
        assert (
            self.binarized_dir.exists()
        ), f"binarized_dir doesn't exist: {self.binarized_dir.resolve()}"

        self.checkpoints = list(
            Path(self.config.checkpoint_dir).glob(self.config.checkpoint_glob)
        )

    def array(self):
        """
        GenerateMultiBleuDetokModule's run function is called for each checkpoint_file_path and for each split
        Returns array of JobConfig's
        """
        array = [
            JobConfig(split=split, checkpoint_file_path=Path(checkpoint_file))
            for checkpoint_file in self.checkpoints
            for split in self.config.splits
        ]
        return array

    def requirements(self):
        return Requirements(
            tasks_per_node=1,
            nodes=1,
            gpus_per_node=1,
            cpus_per_task=8,
            timeout_min=1000,
            constraint="volta32gb",
        )

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        checkpoint_file = iteration_value.checkpoint_file_path
        split = iteration_value.split

        fairseq_generate_output_file = fairseq_generate(
            src_lang=self.config.src_lang,
            tgt_lang=self.config.tgt_lang,
            split=split,
            checkpoint_file=checkpoint_file,
            output_dir=self.output_dir,
            binarized_dir=self.binarized_dir,
            batch_size=self.config.batch_size,
            beam=self.config.beam,
        )

        return multi_bleu_detok_call(fairseq_generate_output_file)

    def name(self):
        return f"generate_multi_bleu_detok.{self.config.src_lang}-{self.config.tgt_lang}.{self.sha_key()}"


def fairseq_generate(
    src_lang: str,
    tgt_lang: str,
    split: str,
    checkpoint_file: Path,
    output_dir: Path,
    binarized_dir: Path,
    batch_size: int,
    beam: int,
) -> Path:
    assert (
        src_lang != tgt_lang
    ), f"src_lang {src_lang} must not be identical to tgt_lang"

    output_file = output_dir / f"{checkpoint_file.stem}_{split}.out"

    fairseq_generate_command = [
        sys.executable,
        "-m",
        "fairseq_cli.generate",
        binarized_dir,
        "--gen-subset",
        split,
        "--source-lang",
        src_lang,
        "--target-lang",
        tgt_lang,
        "--no-progress-bar",
        "--path",
        checkpoint_file,
        "--batch-size",
        str(batch_size),
        "--beam",
        str(beam),
        "--post-process",
        "sentencepiece",
    ]

    logger.info(f"Fairseq Generate subprocess command: {fairseq_generate_command}")

    with open(output_file, "w") as out:
        subprocess.run(
            fairseq_generate_command,
            stdout=out,
            check=True,
        )

    if output_file.stat().st_size == 0:
        raise FairseqGenerateOutputFileEmpty(output_file)

    return output_file


def multi_bleu_detok_call(
    fairseq_generate_output_file: Path,
) -> Path:
    """
    Calls the multi-bleu-detok.perl script to generate bleu score
    Takes in outputted file from fairseq generate
    Returns bleu score file path
    """
    assert (
        fairseq_generate_output_file.exists()
    ), f"fairseq_generate_output_file: {fairseq_generate_output_file} for evaluation doesn't exist"

    bleu_score_file = (
        fairseq_generate_output_file.parent
        / f"{fairseq_generate_output_file.stem}.bleu"
    ).resolve()

    multi_bleu_detok_script = moses_cli_module.get_moses_script(
        "scripts/generic/multi-bleu-detok.perl"
    )

    output_ref_file = process_output_ref_hyp_file(fairseq_generate_output_file, "ref")
    output_hyp_file = process_output_ref_hyp_file(fairseq_generate_output_file, "hyp")

    multi_bleu_detok_command = [
        "perl",
        multi_bleu_detok_script,
        output_ref_file,
    ]

    logger.info(f"multi_bleu_detok command: {multi_bleu_detok_command}")
    with open(bleu_score_file, "w") as out, open(output_hyp_file, "r") as input:
        subprocess.run(
            multi_bleu_detok_command,
            stdin=input,
            stdout=out,
            check=True,
        )

    output_ref_file.unlink()
    output_hyp_file.unlink()

    return bleu_score_file


def process_output_ref_hyp_file(
    fairseq_generate_output_file: Path, file_type: str
) -> Path:
    """
    Helper function for multi_bleu_detok_call
    Takes in fairseq_generate_output_file and a file_type (i.e. one of ref or hyp)
    Extracts desired column from fairseq_generate_output_file and returns ref or hyp file
    """
    assert file_type == "ref" or file_type == "hyp"
    return_file = Path(f"{fairseq_generate_output_file}.{file_type}")
    desired_line_prefix = "T" if file_type == "ref" else "H"
    desired_col_number = 1 if file_type == "ref" else 2
    with open(fairseq_generate_output_file, "r", encoding="utf-8") as read_file, open(
        return_file, "w", encoding="utf-8"
    ) as write_file:
        for line in read_file:
            if line.startswith(desired_line_prefix):
                line = line.rstrip("\n")
                # col format for ref (T) and hyp (H) lines: ID [tab] score [tab] text
                desired_column = line.split("\t", 3)[desired_col_number]
                print(desired_column, file=write_file)
    return return_file


def get_checkpoint_files_list_from_checkpoints_dir(
    train_fairseq_checkpoints_dir: Path,
    minimum_epoch: int,
    maximum_epoch: int,
) -> tp.List[Path]:
    """
    Params:
    train_fairseq_checkpoints_dir: Path - this is the dir that contains all checkpoints produced by TrainFariseqModule
    minimum_epoch & maximum_epoch - minimum/maximum checkpoint number to be included in list

    This function returns a list of checkpoint file paths found in train_fairseq_checkpoints_dir
    The config for GenerateMultiBleuDetokModule requires all checkpoint paths in a list (not as one directory in which the checkpoints exist)
    Hence, this function can be called to help in creating the config for GenerateMultiBleuDetokModule
    """
    checkpoint_files_list = []
    for checkpoint_number in range(minimum_epoch, maximum_epoch + 1):
        #  The checkpoint files from TrainFairseq have a hardcoded naming scheme as below
        checkpoint_file = (
            train_fairseq_checkpoints_dir / f"checkpoint{checkpoint_number}.pt"
        )
        assert (
            checkpoint_file.exists()
        ), f"checkpoint_file: {checkpoint_file} doesn't exist"
        checkpoint_files_list.append(str(checkpoint_file))

    return checkpoint_files_list
