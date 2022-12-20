# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import itertools
import logging
import os
from pathlib import Path

import hydra
from omegaconf import OmegaConf

import stopes.core
from stopes.core import utils
from stopes.modules import nmt_bitext_eval_utils
from stopes.modules.evaluation.generate_multi_bleu_detok_module import (
    GenerateMultiBleuDetokModule,
)
from stopes.modules.nmt_bitext_eval_utils import (
    NMTBitextEvalConfig,
    determine_valid_test_bleu,
    moses_preprocess,
    parse_generated_bleu_files,
    spm_train_encode_binarize,
)
from stopes.modules.train_fairseq_module import TrainFairseqModule

logger = logging.getLogger("NMT_bitext_eval")


class NMTBitextEval:
    """
    This pipeline takes a corpus of bitext, and train a MT system on it using Fairseq.

    Sample Usage:
        python nmt_bitext_eval.py src_lang=ben tgt_lang=hin bitext_tsv=<path here>

        Call this python script with required hydra overrides.
        These must include:
        - src_lang, tgt_lang
        - bitext_tsv (tsv file with {score} {src} {tgt})

        The pipeline is divided in 3 big steps:
            * Preprocessing:
                - Split each TSVs files, optionally remove bitext for a given threshold
                - Apply Moses preprocessing
                    (normalize-punctuation, lowercase, remove-non-printing-char, deescape-special-chars)
                    treat train, valid and test files
                - Train an SPM for each lang
                - use SPM to binarize data for Fairseq training
            * Train an NMT model using Fairseq

            * Evaluate: for each checkpoint of the model compute the detokenized BLEU.
    """

    def __init__(
        self,
        config: NMTBitextEvalConfig,
    ):
        self.config = config

        self.langs = [self.config.src_lang, self.config.tgt_lang]
        self.lang_pair = "-".join(self.langs)
        self.output_dir = Path(self.config.work_dir).resolve()
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.launcher = hydra.utils.instantiate(config.launcher)
        self.bitext_tsv = Path(self.config.bitext_tsv)
        logger.info(f"output_dir: {self.output_dir}")
        logger.info(f"input file: {self.output_dir}")

        assert (
            self.bitext_tsv.exists()
        ), f"The input mined data file: {self.bitext_tsv} doesn't exist and is required to start this pipeline \
             Please provide an existing file, or if you don't have one, generate it by running the global_mining_pipeline."

        OmegaConf.save(config=self.config, f=self.output_dir / "nmt_bitext_eval.yaml")

    def name(self) -> str:
        return f"{self.bitext_tsv.stem}.{self.config.bitext_threshold*100}"

    async def run(self):
        assert (
            self.config.src_lang != self.config.tgt_lang
        ), f"src_lang is {self.config.src_lang} must be different than tgt_lang"

        logger.info("Starting to Split TSV and process_lang")
        split_cfg = nmt_bitext_eval_utils.SplitConcatBitextFilesConfig(
            bitext_tsv=self.config.bitext_tsv,
            src_lang=self.config.src_lang,
            tgt_lang=self.config.tgt_lang,
            output_dir=self.output_dir / "data_raw",
            bitext_threshold=self.config.bitext_threshold,
            max_tsv_lines=self.config.max_tsv_lines,
            public_bitext_base_dir=self.config.public_bitext_base_dir,
            public_corpora_to_ignore=self.config.public_corpora_to_ignore,
        )
        src_corpus, tgt_corpus = await self.launcher.schedule(
            nmt_bitext_eval_utils.SplitConcatBitextFiles(split_cfg)
        )

        moses_outputs = await moses_preprocess(
            self.config,
            src_corpus,
            tgt_corpus,
            self.launcher,
        )

        # trains a spm model, spm-encodes, and then binarizes all splits
        binarized_src_files, binarized_tgt_files = await asyncio.gather(
            *[
                spm_train_encode_binarize(
                    self.lang_pair,
                    lang,
                    moses_output,
                    self.launcher,
                    self.config,
                )
                for lang, moses_output in zip(self.langs, moses_outputs)
            ]
        )

        if binarized_src_files.spm == binarized_tgt_files.spm:
            raise ValueError(
                "Your config generated the two spm at the same file ! Results will be corrupted because of this. Check your config."
            )

        logger.info("Starting Train Fairseq Module")
        data_dir = Path(self.config.train_fairseq.params.task.data)
        data_dir.mkdir(exist_ok=True)

        bin_files = {}
        for binarized_files in [binarized_src_files, binarized_tgt_files]:
            lang = binarized_files.lang
            stopes.core.utils.symlink(
                data_dir / f"dict.{lang}.txt", binarized_files.dict_file
            )
            stopes.core.utils.symlink(data_dir / f"{lang}.spm", binarized_files.spm)

            for split, ext in itertools.product(
                ("train", "valid", "test"),
                (".bin", ".idx"),
            ):
                simple_name = f"{split}.{self.lang_pair}.{lang}{ext}"
                # Binarizer will generate file with .000.{lang} suffix.
                # Create a symlink without the suffix, so that translation task find the data.
                file_with_shard = (
                    getattr(binarized_files, split).with_suffix(ext).resolve()
                )
                if file_with_shard in bin_files:
                    raise ValueError(
                        f"In {data_dir}, {simple_name} and {bin_files[file_with_shard]} both point to {file_with_shard} ! This is probably due to a bad config."
                    )
                bin_files[file_with_shard] = simple_name
                stopes.core.utils.symlink(data_dir / simple_name, file_with_shard)

        train_fairseq_module = TrainFairseqModule(self.config.train_fairseq)
        best_checkpoint_path = await self.launcher.schedule(train_fairseq_module)
        train_fairseq_checkpoints_dir = best_checkpoint_path.parent

        logger.info(f"Starting Evaluation")
        with utils.clone_config(self.config.eval) as eval_cfg:
            eval_cfg.binarized_dir = data_dir
            eval_cfg.checkpoint_dir = str(best_checkpoint_path.parent)

        bleu_module = GenerateMultiBleuDetokModule(eval_cfg)
        bleu_files = await self.launcher.schedule(bleu_module)

        logger.info(f"Evaluation completed, output dir: {self.output_dir}")

        results = parse_generated_bleu_files(bleu_files)

        valid, test = determine_valid_test_bleu(results)
        logger.info("SUMMARY OF RESULTS")
        if valid:
            logger.info(f"best (valid) bleu: {valid['bleu']} (ckpt: {valid['ckpt']})")
        if test:
            logger.info(f"test bleu: {test['bleu']} (ckpt: {test['ckpt']}")

        # set_of_bleu_score_output_dirs contains dirs of all generated bleu scores for the user to view
        set_of_bleu_score_output_dirs = set()
        for bleu_score_file in bleu_files:
            # All bleu scores will usually exist in same parent dir, but not if some results are cached
            set_of_bleu_score_output_dirs.add(bleu_score_file.parent)

        logger.info(
            "To inspect all generated bleu scores, please view .bleu files in these folder(s):"
        )
        for dir in set_of_bleu_score_output_dirs:
            logger.info(f"\t - {str(dir)}")


OmegaConf.register_new_resolver(
    "config_sha", stopes.core.utils.config_sha, replace=True
)


@hydra.main(config_path="conf", config_name="nmt_bitext_eval")
def main(config: NMTBitextEvalConfig) -> None:
    pipeline = NMTBitextEval(config)
    asyncio.run(pipeline.run())


if __name__ == "__main__":
    main()
