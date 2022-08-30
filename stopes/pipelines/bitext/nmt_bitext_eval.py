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
from stopes.core.utils import ensure_dir
from stopes.modules.evaluation.generate_multi_bleu_detok_module import (
    GenerateMultiBleuDetokModule,
)
from stopes.modules.nmt_bitext_eval_utils.preproc_binarized_mined_utils import (
    NMTBitextEvalConfig,
    clone_config,
    moses_preprocess,
    parse_generated_bleu_files,
    split_to_mono,
    spm_train_encode_binarize,
)
from stopes.modules.train_fairseq_module import TrainFairseqModule

logger = logging.getLogger("NMT_bitext_eval")


class NMTBitextEval:
    """
    How to call this script:
        Sample call:
        python nmt_bitext_eval.py src_lang=ben tgt_lang=hin input_file_mined_data_tsv=<path here>

        Call this python script with required hydra overrides.
        These must include:
        - src_lang, tgt_lang
        -input_file_mined_data_tsv (whether gzipped, xzipped, or not zipped)

        These may optionally include, based on your choice:
        - any other config fields

    What this script does: the functionalities have been segmented into steps:
        Preprocessing and Binarizing:
            - Split TSVs
            - process_lang function is called for each split (train, valid, test):
                for each lang:
                    Moses preprocessing applied (normalize-punctuation, lowercase, remove-non-printing-char, deescape-special-chars)
                    spm_training done (on train)
                    Spm Encode Binarize (FairseqBinarize Module)

        Train:
            - trainFairseqModule

        Evaluate: (per epoch)
            - generate.py script
            - multi_bleu_detok for generating bleu scores
    """

    def __init__(
        self,
        config: NMTBitextEvalConfig,
    ):
        self.config = config
        self.lang_pair = f"{self.config.src_lang}-{self.config.tgt_lang}"
        self.output_dir = os.path.abspath(self.config.output_dir)
        self.ensure_all_dirs()
        self.launcher = hydra.utils.instantiate(config.launcher)

        logger.info(f"output_dir: {self.output_dir}")
        logger.info(f"bin_dir: {self.config.bin_dir}")
        self.input_file_mined_data_tsv = Path(self.config.input_file_mined_data_tsv)

        assert (
            self.input_file_mined_data_tsv.exists()
        ), f"The input mined data file: {self.input_file_mined_data_tsv} doesn't exist and is required to start this pipeline \
             Please provide an existing file, or if you don't have one, generate it by running the global_mining_pipeline."

        OmegaConf.set_readonly(self.config, True)

    async def run(self):
        assert (
            self.config.src_lang != self.config.tgt_lang
        ), f"src_lang is {self.config.src_lang} must be different than tgt_lang"

        logger.info("Starting to Split TSV and process_lang")
        src_lang_file_after_split, tgt_lang_file_after_split = split_to_mono(
            self.input_file_mined_data_tsv,
            self.config.preproc_binarize_mined.max_number_of_tsv_lines_in_millions,
            self.config.preproc_binarize_mined.bitext_alignment_minimum_score_threshold,
            self.config.bin_dir,
            self.config.src_lang,
            self.config.tgt_lang,
            self.config.public_bitext_base_dir,
            self.config.public_bitext_margin,
            self.config.public_corpora_to_ignore,
        )

        moses_src_output, moses_tgt_output = await moses_preprocess(
            self.config,
            src_lang_file_after_split,
            tgt_lang_file_after_split,
            self.launcher,
        )

        # trains a spm model, spm-encodes, and then binarizes all splits
        binarized_src_lang_outputs, binarized_tgt_lang_outputs = await asyncio.gather(
            spm_train_encode_binarize(
                self.lang_pair,
                self.config.src_lang,
                moses_src_output,
                self.launcher,
                self.config,
            ),
            spm_train_encode_binarize(
                self.lang_pair,
                self.config.tgt_lang,
                moses_tgt_output,
                self.launcher,
                self.config,
            ),
        )

        logger.info("Starting Train Fairseq Module")
        cache_fixed_spm_and_binarize_output_dir = (
            self.config.spm_train_and_binarize_output_dir
        )
        # Symlinking: to remove shard id's from binarized data file names, as per hardcoded naming requirement for TrainFairseq Module
        for split, ext, lang in itertools.product(
            ("train", "valid", "test"),
            (".bin", ".idx"),
            (self.config.src_lang, self.config.tgt_lang),
        ):
            prefix: Path = f"{split}.{self.lang_pair}.{lang}"
            binarized_files = (
                binarized_src_lang_outputs
                if lang == self.config.src_lang
                else binarized_tgt_lang_outputs
            )
            binarized_files_parent_dir = (
                getattr(binarized_files, split).parent
            ).resolve()

            # Note the line below relies on the fact that ({prefix} + ".000" + {ext}) is the hardcoded naming convention within lineprocessor/fairseq binarizer module
            current_binarized_file_path: Path = binarized_files_parent_dir / str(
                prefix + ".000" + ext
            )

            symlinked_path: Path = Path(
                self.config.spm_train_and_binarize_output_dir
            ).resolve() / str(prefix + ext)

            stopes.core.utils.symlink(
                symlinked_path,
                current_binarized_file_path,
            )

            if (
                Path(self.config.spm_train_and_binarize_output_dir).resolve()
                == binarized_files_parent_dir
            ):
                # Since output dir prescribed in config == binarized files output dir, we know this 6 binarize call was not cached
                # Because otherwise, the binarized file output dir would be an old cached dir. Hence, we turn spm dir into a full resolved path
                # Why? in order to prevent train fairseq module from running from cache as well (passing an absolute path for spm dir to train fairseq config will change the config and ensure no cache)
                cache_fixed_spm_and_binarize_output_dir = str(
                    Path(cache_fixed_spm_and_binarize_output_dir).resolve()
                )

        with clone_config(
            self.config.train_fairseq.config
        ) as edited_train_fairseq_config:
            edited_train_fairseq_config.params.task = {
                "_name": "translation",
                "source_lang": self.config.src_lang,
                "target_lang": self.config.tgt_lang,
                "data": cache_fixed_spm_and_binarize_output_dir  # This points to the spm/binarize outputs;
                # Even if spm model/dict files + binarized files are cached and exist in a dir somewhere else, they have been symlinked into this config output dir so it will work properly
            }

            edited_train_fairseq_config.params.checkpoint.save_dir = str(
                Path(edited_train_fairseq_config.output_dir)
                / "nmt_train_fairseq"
                / f"{self.lang_pair}"
            )

        train_fairseq_module = TrainFairseqModule(edited_train_fairseq_config)
        best_checkpoint_path = await self.launcher.schedule(train_fairseq_module)
        train_fairseq_checkpoints_dir = best_checkpoint_path.parent

        logger.info(f"Starting Evaluation")
        with clone_config(self.config.eval.config) as edited_eval_config:
            edited_eval_config.binarized_dir = cache_fixed_spm_and_binarize_output_dir
            edited_eval_config.output_dir = str(train_fairseq_checkpoints_dir)
            edited_eval_config.checkpoint_dir = str(best_checkpoint_path.parent)

        generate_multi_bleu_detok_module = GenerateMultiBleuDetokModule(
            edited_eval_config
        )
        generated_bleu_score_files = await self.launcher.schedule(
            generate_multi_bleu_detok_module
        )

        logger.info(f"Evaluation completed, output dir: {self.output_dir}")

        results = parse_generated_bleu_files(generated_bleu_score_files)

        for split in results:
            logger.info(f"BEST BLEU ({split}): {max(results[split])}")

        # set_of_bleu_score_output_dirs contains dirs of all generated bleu scores for the user to view
        set_of_bleu_score_output_dirs = set()
        for bleu_score_file in generated_bleu_score_files:
            # All bleu scores will usually exist in same parent dir, but not if some results are cached
            set_of_bleu_score_output_dirs.add(bleu_score_file.parent)

        logger.info(
            "To inspect all generated bleu scores, please view .bleu files in these folder(s):"
        )
        for dir in set_of_bleu_score_output_dirs:
            logger.info(f"\t - {str(dir)}")

    def ensure_all_dirs(self):
        ensure_dir(self.output_dir)
        ensure_dir(self.config.bin_dir)
        ensure_dir(self.config.spm_train_and_binarize_output_dir)
        ensure_dir(self.config.moses_filter.output_dir)
        ensure_dir(self.config.spm.train.config.output_dir)
        ensure_dir(self.config.train_fairseq.config.output_dir)


@hydra.main(config_path="conf", config_name="nmt_bitext_eval")
def main(config: NMTBitextEvalConfig) -> None:
    pipeline = NMTBitextEval(config)
    asyncio.run(pipeline.run())


if __name__ == "__main__":
    main()
