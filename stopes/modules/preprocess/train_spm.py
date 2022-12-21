# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import logging
import typing as tp
from pathlib import Path

import sentencepiece as spm
from omegaconf import MISSING

from stopes.core.stopes_module import Requirements, StopesModule

logger = logging.getLogger(__name__)


@dataclasses.dataclass()
class TrainSpmConfig:
    output_dir: str = MISSING
    train_data_file: str = MISSING
    vocab_size: int = 50_000
    training_lines: int = 5_000_000
    seed_sentencepiece_size: int = 5_000_000
    character_coverage: float = 0.999995
    model_type: str = "unigram"
    shuffle_input_sentence: bool = True
    num_threads: int = 4
    model_prefix_spm: str = ""  # optional value; if passed as empty, will be auto set based on train_data_file name


@dataclasses.dataclass()
class Vocab:
    model_file: Path
    vocab_file: Path

    @property
    def dict_file(self):
        dict_file = Path(f"{self.model_file.parent}/{self.model_file.stem}.dict.txt")
        if not dict_file.exists():
            # Convert .vocab file into a fairseq dictionary
            # Note: name scheme is expected like this strictly;
            # its a hardcoded requirement in fairseq/tasks/translation.py
            with open(dict_file, "w", encoding="utf-8") as o:
                with open(self.vocab_file, encoding="utf-8") as f:
                    for line in f:
                        word = line.split("\t", 1)[0]
                        # Those special tokens are added by fairseq
                        if word in ["<unk>", "<s>", "</s>"]:
                            continue
                        print(word, 1, file=o)
        return dict_file


class TrainSpmModule(StopesModule):
    """
    Train a SPM model using module API.

    python -m pdb launch_module.py train_spm.config.train_data_file=/checkpoint/guw/hmine/align/bel.v2.sents.bel.xz train_spm.config.output_dir=<output_dir> launcher.cluster=debug
    """

    def __init__(
        self,
        config: TrainSpmConfig,
    ):
        super().__init__(config, TrainSpmConfig)
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def requirements(self) -> Requirements:
        return Requirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=self.config.num_threads,
            timeout_min=2 * 24 * 60,
        )

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> Vocab:
        if self.config.model_prefix_spm == "":
            input_file_stem = Path(self.config.train_data_file).stem
            model_prefix_spm = (
                self.output_dir
                / f"spm_train.{input_file_stem}.{self.config.vocab_size}"
            )
        else:
            model_prefix_spm = self.output_dir / self.config.model_prefix_spm

        return train_spm(
            self.config, Path(self.config.train_data_file), model_prefix_spm
        )

    def validate(
        self,
        output: tp.Any,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> bool:
        vocab = output
        return (
            vocab.model_file.exists()
            and vocab.vocab_file.exists()
            and vocab.dict_file.exists()
        )

    def version(cls):
        return "0.3"

    def name(self):
        data = Path(self.config.train_data_file).stem
        return f"train_spm_{data}"


def train_spm(
    spm_config: TrainSpmConfig,
    train_data_file: Path,
    model_prefix_spm: Path,
) -> Vocab:
    logger.info(
        f"train_data_file: {train_data_file}, model_prefix_spm: {model_prefix_spm}; "
    )
    spm.SentencePieceTrainer.train(
        input=train_data_file,
        model_prefix=model_prefix_spm,
        vocab_size=int(spm_config.vocab_size),
        character_coverage=spm_config.character_coverage,
        model_type=spm_config.model_type,
        input_sentence_size=spm_config.training_lines,
        seed_sentencepiece_size=spm_config.seed_sentencepiece_size,
        shuffle_input_sentence=spm_config.shuffle_input_sentence,
        num_threads=spm_config.num_threads,
    )

    logger.info(f"SPM Training completed")
    model_file = Path(f"{model_prefix_spm}.model").resolve()
    vocab_file = Path(f"{model_prefix_spm}.vocab").resolve()
    return Vocab(model_file=model_file, vocab_file=vocab_file)
