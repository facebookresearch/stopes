# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import importlib
import logging
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import hydra
import wandb
from omegaconf import MISSING, DictConfig, OmegaConf

from stopes.core import utils
from stopes.core.launcher import Launcher
from stopes.core.stopes_module import Requirements
from stopes.modules.preprocess.bitext_processor import BitextProcessorConfig
from stopes.modules.preprocess.line_processor import LineProcessorModule
from stopes.modules.preprocess.multiproc_bitext_processor import (
    MultiprocBitextProcessorModule,
)
from stopes.modules.train_fairseq_module import TrainFairseqModule
from stopes.modules.translation.fairseq_generate import (
    FairseqGenerateConfig,
    FairseqGenerateModule,
)
from stopes.pipelines.bitext.shard_and_shuffle import (
    ShardAndShuffleConfig,
    shard_and_shuffle,
)
from stopes.pipelines.distillation.distillation_bitext_processor import (
    BitextSplitNormalizeFilterLID,
)
from stopes.pipelines.monolingual.monolingual_line_processor import LIDConfig
from stopes.pipelines.monolingual.monolingual_pipeline import (
    MonoLingualConfig,
    monolingual_cleaning,
)
from stopes.pipelines.monolingual.utils.predict_script import find_lang_script
from stopes.pipelines.monolingual.utils.sentence_split import map_lang
from stopes.utils.mining_utils import extract_shard_id


@dataclass
class DistillationConfig:
    launcher: DictConfig
    mono_pipeline: MonoLingualConfig
    dedup: DictConfig
    shard: ShardAndShuffleConfig
    generate: FairseqGenerateConfig
    lang_code_mapping: DictConfig
    binarize: DictConfig
    train_fairseq: DictConfig
    bitext_clean: BitextProcessorConfig
    lid: LIDConfig = LIDConfig()
    src_langs: tp.List[str] = MISSING
    tgt_langs: tp.List[str] = MISSING
    mono_data_dir: str = MISSING
    output_dir: str = MISSING
    min_lines_per_shard: int = None
    wandb: tp.Optional[DictConfig] = None


logger = logging.getLogger("distillation_pipeline")


async def bitext_clean_helper(
    config: BitextProcessorConfig, file_pair: tp.Tuple[Path, Path], launcher: Launcher
):
    src_lang = file_pair[0].name.split(".")[-2]
    tgt_lang = file_pair[1].name.split(".")[-2]
    lang_pair = file_pair[1].name.split(".")[-3]

    with importlib.resources.path(
        "stopes.pipelines.monolingual", config.language_script_filename
    ) as path:
        lang_script = find_lang_script(tgt_lang, path)

    with importlib.resources.path(
        "stopes.pipelines.monolingual", config.split_language_equivalences_filename
    ) as path:
        tgt_splitter_lang = map_lang(tgt_lang, path)
        src_splitter_lang = map_lang(src_lang, path)

    assert (
        lang_script
    ), f"couldn't find {tgt_lang} script in {config.language_script_filename}"

    logger.info(
        f"using script {lang_script} and splitter for {tgt_splitter_lang} for target lang {tgt_lang}, and splitter for {src_splitter_lang} for source lang {src_lang}."
    )
    with utils.clone_config(config) as bitext_config:
        bitext_config.custom_name = f"bitext_clean_{lang_pair}.{tgt_lang}"
        bitext_config.shards = [([str(file) for file in file_pair])]
        bitext_config.requirements = Requirements(**config.requirements)
        bitext_config.bitext_processor._target_ = (
            f"{BitextSplitNormalizeFilterLID.__module__}.BitextSplitNormalizeFilterLID"
        )
        bitext_config.bitext_processor.tgt_lang = tgt_lang
        bitext_config.bitext_processor.src_splitter_lang = src_splitter_lang
        bitext_config.bitext_processor.lang_script = lang_script
        bitext_config.bitext_processor.tgt_splitter_lang = tgt_splitter_lang

    processor = MultiprocBitextProcessorModule(bitext_config)
    processor_summaries = await launcher.schedule(processor)
    # TODO: perhaps add wandb
    return processor_summaries


class DistillationPipeline:
    """
    See README for details.
    """

    def __init__(
        self,
        config: DistillationConfig,
    ):
        self.config = config

        assert Path(
            self.config.mono_data_dir
        ).exists(), f"The input data directory: {self.config.mono_data_dir} doesn't exist and is required to start this pipeline \
             Please provide an existing monolingual data directory, or if you don't have one, download some data (see README)."

        self.output_dir = str(Path(self.config.output_dir).resolve())

        logger.info(f"Distillation pipeline output_dir: {self.output_dir}")

        self.config.mono_pipeline.output_dir = str(
            Path(self.config.output_dir) / "cleaned_mono_data"
        )

        self.config.shard.output_dir = str(
            Path(self.config.output_dir) / "training_shards"
        )
        self.config.shard.nb_shards = len(self.config.tgt_langs)

        self.config.bitext_clean.output_dir = str(
            Path(self.config.output_dir) / "bitext_clean"
        )

        self.config.binarize.output_dir = str(
            Path(self.config.output_dir) / "binarized_bitext"
        )

        self.config.train_fairseq.output_dir = str(Path(self.config.output_dir))

        self.config.bitext_clean.lid = self.config.lid
        self.config.mono_pipeline.lid = self.config.lid

        self.lang_pairs = [
            f"{src_lang}-{tgt_lang}"
            for src_lang in self.config.src_langs
            for tgt_lang in self.config.tgt_langs
        ]
        self.config.train_fairseq.params.task.lang_pairs = ",".join(
            self.lang_pairs  # comma separated list based on input src and tgt langs
        )

        langs_list = self.config.src_langs.copy()
        langs_list.extend(self.config.tgt_langs)
        self.config.train_fairseq.params.task.langs = langs_list
        self.config.train_fairseq.params.checkpoint.save_dir = str(
            Path(self.config.train_fairseq.output_dir) / "distilled_model"
        )

        self.ensure_all_dirs()

        # get a launcher as per the config
        self.launcher = hydra.utils.instantiate(self.config.launcher)
        OmegaConf.save(
            config=config,
            f=str(Path(self.launcher.config_dump_dir) / "distillation.yaml"),
        )

        OmegaConf.set_readonly(self.config, True)

    async def run(self):
        logger.info(f"Starting monolingual pipeline to clean/filter source data...")
        mono_prep_output_files = await monolingual_cleaning(self.config.mono_pipeline)

        logger.info(
            f"Starting to shard and shuffle filtered source data based on number of target languages..."
        )
        shard_jobs_list = []
        for i in range(len(mono_prep_output_files)):
            with utils.clone_config(self.config.shard) as shard_config:
                shard_config.input_file = mono_prep_output_files[i].as_posix()
                shard_config.outfile_prefix = (
                    f"source_shard.{str(mono_prep_output_files[i].name).split('_')[0]}"
                )
            shard_jobs_list.append(
                shard_and_shuffle(
                    shard_config, self.config.min_lines_per_shard, self.launcher
                )
            )

        source_shards = await asyncio.gather(*shard_jobs_list)
        source_shards = [shard for job in source_shards for shard in job[0]]

        # renames source_shards so each source data source has a shard corresponding to each target language
        for idx, shard in enumerate(source_shards):
            file_parts = str(shard.name).split(".")
            src_lang = file_parts[-3]
            tgt_lang = self.config.tgt_langs[int(file_parts[-2])]
            new_file_name = Path(str(shard.parent)) / str(
                ".".join(file_parts[:-3]) + f".{src_lang}-{tgt_lang}.{src_lang}.xz"
            )
            new_file_name = shard.rename(new_file_name)
            source_shards[idx] = new_file_name

        assert len(source_shards) == len(self.config.src_langs) * len(
            self.config.tgt_langs
        ), f"An error occurred during sharding - mismatch of output shard count and expected count"

        logger.info(f"Starting generation of target data...")
        # converts each shard into a tuple of format (file_path, src_lang, tgt_lang) based on naming convention from step above
        file_list = [
            (
                str(src_file),
                src_lang,
                src_file.name.split(".")[-3].split("-")[1],
            )
            for src_file in source_shards
            for src_lang in self.config.src_langs
            if src_file.name.split(".")[-3].startswith(f"{src_lang.split('_')[0]}")
        ]

        with utils.clone_config(self.config.generate) as generate_config:
            generate_config.file_list = file_list

        generate_module = FairseqGenerateModule(
            generate_config,
            Path(self.config.shard.output_dir),
        )

        generated = await self.launcher.schedule(generate_module)

        logger.info(
            "Starting to clean/filter the generated target data while also removing corresponding source sentences..."
        )
        bitext_clean_jobs = []
        for direction in generated:
            bitext_clean_jobs.append(
                bitext_clean_helper(self.config.bitext_clean, direction, self.launcher)
            )

        bitext_summaries = await asyncio.gather(*bitext_clean_jobs)
        cleaned_bitext = [
            [s[0].src_output_file, s[0].tgt_output_file] for s in bitext_summaries
        ]
        cleaned_bitext = [file for file_pair in cleaned_bitext for file in file_pair]

        logger.info(
            f"Starting to binarize and encode the cleaned source and target (bitext) data..."
        )
        split = "train"
        binarize_encode_jobs_list = []
        for file in cleaned_bitext:
            lang_pair = file.name.split(".")[-4]
            lang = file.name.split(".")[-3]
            outfile_prefix = f"{split}.{lang_pair}.{lang}"

            with utils.clone_config(self.config.binarize) as binarizer_config:
                binarizer_config.outfile_prefix = outfile_prefix
                binarizer_config.shards = [str(file)]

            binarize_module = LineProcessorModule(binarizer_config)
            binarized_shards = self.launcher.schedule(binarize_module)
            binarize_encode_jobs_list.append(binarized_shards)
        binarized = await asyncio.gather(*binarize_encode_jobs_list)

        train_shards: tp.Set[Path] = set()
        for shards in binarized:
            for file in shards:
                shard_idx = extract_shard_id(file.name)
                shard_dir = file.parent / f"shard{shard_idx:03d}"
                shard_dir.mkdir(exist_ok=True)
                new_file = file.name.replace(f".{shard_idx:03d}.", ".")
                utils.symlink(shard_dir / new_file, file)
                utils.symlink(
                    shard_dir / new_file.replace("bin", "idx"), file.with_suffix(".idx")
                )
                train_shards.add(shard_dir.absolute())

        train_shards: tp.List[Path] = sorted(train_shards)

        logger.info(
            f"Starting to train distilled student model based on binarized bitext data..."
        )
        # adds symlinks to SPM dictionaries to ${output_dir}/binarized_bitext directory
        for lang in self.config.train_fairseq.params.task.langs:
            dict_to_copy = (
                Path(self.config.binarize.spm_model_path).parents[1]
                / "data_bin"
                / "shard000"
                / f"dict.{lang}.txt"
            )
            for shard in train_shards:
                new_location = shard / f"dict.{lang}.txt"
                utils.symlink(new_location, dict_to_copy)

        with utils.clone_config(self.config.train_fairseq) as fairseq_config:
            fairseq_config.params.task.data = ":".join([str(s) for s in train_shards])

        train_fairseq_module = TrainFairseqModule(fairseq_config)

        best_checkpoint_path = await self.launcher.schedule(train_fairseq_module)

        train_fairseq_checkpoints_dir = best_checkpoint_path.parent
        logger.info(
            f"Distillation completed. The trained student model is located in the directory {train_fairseq_checkpoints_dir}."
        )

    def ensure_all_dirs(self):
        utils.ensure_dir(self.output_dir)
        utils.ensure_dir(self.config.shard.output_dir)
        utils.ensure_dir(self.config.bitext_clean.output_dir)
        utils.ensure_dir(self.config.binarize.output_dir)
        utils.ensure_dir(self.config.train_fairseq.params.checkpoint.save_dir)


@hydra.main(config_path="conf", config_name="distillation")
def main(config: DistillationConfig) -> None:
    if config.wandb is not None:
        run = wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            config=OmegaConf.to_container(config),
        )
        run.name = f'distil.[{",".join(config.src_langs)}].{run.name}'

    pipeline = DistillationPipeline(config)
    asyncio.run(pipeline.run())


if __name__ == "__main__":
    main()
