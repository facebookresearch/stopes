# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import os
import shlex
import subprocess
import tempfile
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import OmegaConf

from stopes.core.utils import split_large_files
from stopes.modules.translation.fairseq_generate import (
    FairseqGenerateModule,
    FairseqGenerateModuleConfig,
)

logger = logging.getLogger("translation_pipeline")
logger.setLevel(logging.DEBUG)


@dataclass
class TranslationConfig:
    output_dir: Path
    launcher: tp.Dict[str, tp.Any]
    generation: FairseqGenerateModuleConfig
    max_size_per_shard: str = "10M"


def _parse_tsv_column_path(raw_path: str):
    """Parse paths in the format `/some/tsv.file:column_name`"""
    assert raw_path.count(":") == 1
    file_path, col_name = raw_path.split(":")
    assert file_path.endswith(".tsv")
    return file_path, col_name


def _is_tsv_column_path(raw_path: str) -> bool:
    if raw_path.count(":") == 1 and ".tsv" in raw_path:
        file_path, _colname = _parse_tsv_column_path(raw_path)
        if file_path.endswith(".tsv"):
            return True
    return False


def _extract_tsv_column(raw_path: str, out_dir: Path) -> str:
    """Take a TSV:column path, extract the column to a tempfile and return its path"""
    assert _is_tsv_column_path(raw_path)
    fd, new_path = tempfile.mkstemp(dir=out_dir, text=True)
    file_path, col_name = _parse_tsv_column_path(raw_path)
    df = pd.read_csv(open(file_path), quoting=3, sep="\t", usecols=[col_name])
    with os.fdopen(fd, "wt") as fout:
        for line in df[col_name]:
            print(line, file=fout)
    return new_path


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
    assert (
        config.generation.get("preserve_filenames") is not None
    ), "The preserve_filenames option of the generation module is required."

    out_dir = Path(config.generation.output_dir)

    out_dir.mkdir(exist_ok=True, parents=True)

    # Sharding
    logger.info("Sharding files...")
    with tempfile.TemporaryDirectory(
        dir=out_dir,
        prefix="splits_tmp",
    ) as tmp_dir:
        file_list = []
        # Extract data from TSV files, if necessary
        for fpath, src_lang, tgt_lang in config.generation.file_list:
            if _is_tsv_column_path(fpath):
                fpath = _extract_tsv_column(fpath, Path(tmp_dir))
            file_list.append((fpath, src_lang, tgt_lang))

        # Shard data, if necessary
        file_shards = {
            fpath: list(
                split_large_files([fpath], config.max_size_per_shard, Path(tmp_dir))
            )
            for fpath in set(Path(path) for path, _, _ in file_list)
        }

        # Replace the generation file list by including the shards
        sharded_file_list = [
            (str(new_path), src_lang, tgt_lang)
            for orig_path, src_lang, tgt_lang in file_list
            for new_path in file_shards[Path(orig_path)]
        ]
        config.generation.file_list = sharded_file_list

        # These sharded outputs will need to be merged
        to_be_merged = {}
        for orig_path, src_lang, tgt_lang in file_list:
            orig_path = Path(orig_path)
            shards = file_shards[orig_path]
            if len(shards) > 1:
                to_be_merged[out_dir / f"{src_lang}-{tgt_lang}.{orig_path.name}"] = [
                    out_dir / f"{src_lang}-{tgt_lang}.{shard.name}" for shard in shards
                ]

        # Translation
        launcher = hydra.utils.instantiate(config.launcher)
        generate_module = FairseqGenerateModule(config.generation)
        logger.info("Starting large scale translation pipeline...")
        await launcher.schedule(generate_module)

    logger.info("Shard translation complete; merging any sharded outputs...")
    for name, shards in to_be_merged.items():
        subprocess.run(
            shlex.join(["cat"] + [str(f) for f in shards])
            + " > "
            + shlex.quote(str(name)),
            shell=True,
            check=True,
        )
        subprocess.run(
            shlex.join(["rm"] + [str(f) for f in shards]),
            shell=True,
            check=True,
        )

    logger.info(
        f"All translation jobs completed. The outputs can be found in {out_dir}"
    )


@hydra.main(version_base="1.1", config_path="conf", config_name="example")
def main(config: TranslationConfig) -> None:
    OmegaConf.save(
        config=config,
        f=str(Path.cwd() / "translation_config.yaml"),
    )
    asyncio.run(run(config))


if __name__ == "__main__":
    main()
