# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import importlib
import logging
import shlex
import subprocess
import typing as tp
from dataclasses import dataclass
from pathlib import Path
from string import Template

import hydra
import wandb
from omegaconf import MISSING, DictConfig, OmegaConf

from stopes.core import Launcher, utils
from stopes.core.stopes_module import Requirements
from stopes.modules.monolingual.monolingual_sort_dedup import DedupeWithMergeSort
from stopes.modules.preprocess.multiproc_line_processor import (
    MultiprocLineProcessorConfig,
    MultiprocLineProcessorModule,
)
from stopes.pipelines.monolingual.monolingual_line_processor import (
    FilterConfig,
    LIDConfig,
    MonolingualProcessResult,
    SplitNormalizeFilterLID,
)
from stopes.pipelines.monolingual.utils import slurm_tmp_maybe
from stopes.pipelines.monolingual.utils.predict_script import find_lang_script
from stopes.pipelines.monolingual.utils.sentence_split import map_lang


@dataclass
class MonoLingualConfig:
    launcher: DictConfig
    dedup: DictConfig
    langs: tp.List[str] = MISSING
    data_dir: str = MISSING
    output_dir: str = MISSING
    corpus_filter: str = ""
    # this is the name of the tsv resource in this package
    language_script_filename: str = MISSING
    # this is the name of the tsv resource in this package
    split_language_equivalences_filename: str = MISSING

    skip_dedup: bool = False

    # size argument passed down to `split`
    # from the man page: The SIZE argument is an integer and optional unit (example: 10K is 10*1024).
    #   Units are K,M,G,T,P,E,Z,Y (powers of 1024) or KB,MB,... (powers of 1000)
    max_shard_size: str = "500M"
    # /tmp will not work for distributed jobs, be careful
    dist_tmp_dir: str = "/tmp"
    # this one is used locally, so /tmp is ok even in distributed launchers
    local_tmp_dir: str = "/tmp"

    filter: FilterConfig = FilterConfig()
    lid: LIDConfig = LIDConfig()
    preprocess_buffer_size: int = 10_000
    preproces_requirements: Requirements = Requirements()
    # where to find the input files that we care about, useful to debug stuff
    # corpus might be empty if corpus_filter is empty. This uses string.Template
    input_file_glob_template: str = "$lang/$corpus*.$lang.xz"
    # config Weight and Biases
    wandb: tp.Optional[DictConfig] = None


logger = logging.getLogger("monolingual_pipeline")


async def launch_preprocessor(
    launcher: Launcher,
    raw_files: tp.List[Path],
    lang: str,
    config: MonoLingualConfig,
    out_dir: Path,
) -> tp.List[Path]:

    with importlib.resources.path(
        "stopes.pipelines.monolingual", config.language_script_filename
    ) as path:
        lang_script = find_lang_script(lang, path)

    with importlib.resources.path(
        "stopes.pipelines.monolingual", config.split_language_equivalences_filename
    ) as path:
        splitter_lang = map_lang(lang, path)

    assert (
        lang_script
    ), f"couldn't find {lang} script in {config.language_script_filename}"

    logger.info(
        f"found {len(raw_files)} file for {lang}, using script {lang_script} and splitter for {splitter_lang}."
    )

    tmp_dir = slurm_tmp_maybe(Path(config.dist_tmp_dir)) / lang

    # TODO this will cause trouble for the cache as the splits will always be different probably
    shards = list(utils.split_large_files(raw_files, config.max_shard_size, tmp_dir))

    logger.info(f"processing {len(shards)} splits.")

    file_processor = MultiprocLineProcessorModule(
        config=MultiprocLineProcessorConfig(
            line_processor=DictConfig(
                {
                    # this will eventually initialize SplitNormalizeFilterLID above
                    "_target_": f"{SplitNormalizeFilterLID.__module__}.SplitNormalizeFilterLID",
                    "lang": lang,
                    "split_algo": config.split_algo,
                    "filter_config": config.filter,
                    "lid_config": config.lid,
                    "lang_script": lang_script,
                    "splitter_lang": splitter_lang,
                    "num_cpu": config.preproces_requirements.cpus_per_task,
                    "local_tmp_dir": config.local_tmp_dir,
                    "_version": "0.3",
                    "skip_dedup": config.skip_dedup,
                }
            ),
            custom_name=f"monolingual_preproc_{lang}",
            output_dir=str(out_dir),
            outfile_prefix="",  # TODO from config?
            shards=shards,
            buffer_size=config.preprocess_buffer_size,
            requirements=Requirements(**config.preproces_requirements),
            tmp_dir=config.local_tmp_dir,
        )
    )

    preprocess_summaries = await launcher.schedule(file_processor)

    table = wandb.Table(columns=["lang"] + MonolingualProcessResult.table_columns())
    for s in preprocess_summaries:
        table.add_data(*([lang] + s.get_data_row()))
        logger.info(s)
    wandb.log({f"{lang}/per_file": table})

    try:
        # remove the temp shards
        for f in tmp_dir.iterdir():
            f.unlink()
        tmp_dir.rmdir()
    except Exception as e:
        logger.error(
            f"Could not clean up the temp shards in {tmp_dir}, perhaps because it was not necessary. Check it manually and clean if necessary.",
            exc_info=e,
        )

    return preprocess_summaries


async def process_lang(
    lang: str,
    config: MonoLingualConfig,
    launcher: Launcher,
):
    data_dir = Path(config.data_dir)
    raw_files = list(
        data_dir.glob(
            Template(config.input_file_glob_template).substitute(
                corpus=config.corpus_filter, lang=lang
            )
        )
    )

    out_dir = (Path(config.output_dir) / lang).resolve()
    utils.ensure_dir(out_dir)

    logger.info(f"outputs for {lang} going to {out_dir}.")

    preprocess_summaries = await launch_preprocessor(
        launcher=launcher,
        raw_files=raw_files,
        lang=lang,
        config=config,
        out_dir=out_dir,
    )

    processed_files = [s.output_file for s in preprocess_summaries]

    # Deduplicate if necessary
    if config.skip_dedup:
        # Skip deduplication
        logger.info(f"starting to merge unsorted files from {len(processed_files)}")
        output_file = Path(out_dir / f"{lang}_no_dedup.xz")
        # Concatenates the contents of the processed files (source data from different corpi) into one output_file without deduping the merge
        subprocess.run(
            utils.bash_pipefail(
                " ".join(
                    ["cat"]
                    + [str(f) for f in processed_files]
                    + [">"]
                    + [shlex.quote(str(output_file))]
                )
            ),
            shell=True,
            check=True,
        )
        logger.info(f"done merging unsorted files in {output_file}")
        return output_file
    else:
        # Deduplicate
        logger.info(f"starting to dedup from {len(processed_files)}")
        if len(processed_files) > 1:
            with utils.clone_config(config.dedup) as dedup_config:
                dedup_config.shards = [str(f.resolve()) for f in processed_files]
                dedup_config.output_file = str(out_dir / f"{lang}_all_dedup.xz")
                dedup_config.num_cpu = config.preproces_requirements.cpus_per_task
                dedup_config.tmp_dir = config.local_tmp_dir
            dedup_module = DedupeWithMergeSort(dedup_config)

            final = await launcher.schedule(dedup_module)
        else:
            final = processed_files[0]
        logger.info(f"done cleaning and deduplicating in {final}")
        return final


async def monolingual_cleaning(config: MonoLingualConfig):
    # get a launcher as per the config
    launcher = hydra.utils.instantiate(config.launcher)

    OmegaConf.save(
        config=config,
        f=str(Path(launcher.config_dump_dir) / "monolingual.yaml"),
    )

    finals = await asyncio.gather(
        *[process_lang(lang, config, launcher) for lang in config.langs]
    )
    return finals


@hydra.main(config_path="conf", config_name="monolingual")
def main(config: MonoLingualConfig) -> None:
    run = wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        config=OmegaConf.to_container(config),
    )
    run.name = f'mono.[{",".join(config.langs)}].{run.name}'
    asyncio.run(monolingual_cleaning(config))


if __name__ == "__main__":
    main()
