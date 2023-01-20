# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import shutil
import tempfile
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import hydra

from stopes.core.utils import split_large_files
from stopes.modules.translation.fairseq_generate import (
    FairseqGenerateConfig,
    FairseqGenerateModule,
)

logger = logging.getLogger("translation_pipeline")


@dataclass
class TranslationConfig:
    launcher: tp.Dict[str, tp.Any]
    generation: FairseqGenerateConfig
    max_size_per_shard: str = "10M"


async def run(config: TranslationConfig):
    assert len(config.generation.get("file_list", [])) and not config.generation.get(
        "src_text_file"
    ), (
        "This pipeline does not support the `src_text_file` generation config option. "
        "Please use the `file_list` option instead."
    )

    # Without the preserve_filenames option, the FairseqGenerateModule outputs
    # src_lang-tgt_lang.gen files. If we specify multiple generation jobs with the same
    # direction, they would overwrite each other. This pipeline is almost guaranteed to
    # have multiple files per direction due to sharding, so we require this option.
    assert config.generation.get(
        "preserve_filenames"
    ), "The preserve_filenames option of the generation module is required."

    # Sharding
    logger.info("Sharding files...")
    with tempfile.TemporaryDirectory(
        dir=config.generation.output_dir, prefix="splits_tmp"
    ) as tmp_dir:
        file_shards = {
            fpath: list(
                split_large_files([fpath], config.max_size_per_shard, Path(tmp_dir))
            )
            for fpath in set(Path(path) for path, _, _ in config.generation.file_list)
        }

        # Replace the generation file list by including the shards
        file_list = [
            (str(new_path), src_lang, tgt_lang)
            for orig_path, src_lang, tgt_lang in config.generation.file_list
            for new_path in file_shards[Path(orig_path)]
        ]
        config.generation.file_list = file_list

        # Translation
        launcher = hydra.utils.instantiate(config.launcher)
        generate_module = FairseqGenerateModule(config.generation)
        logger.info("Starting large scale translation pipeline...")
        await launcher.schedule(generate_module)

    logger.info(
        "All translation jobs completed. "
        f"The output can be found in {generate_module.output_dir}"
    )


@hydra.main(config_path="conf", config_name="example")
def main(config: TranslationConfig) -> None:
    asyncio.run(run(config))


if __name__ == "__main__":
    main()
