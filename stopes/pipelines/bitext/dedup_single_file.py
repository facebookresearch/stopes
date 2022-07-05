# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import glob
import logging
import shutil
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import hydra
import wandb
from omegaconf import MISSING, DictConfig, OmegaConf

from stopes.core import stopes_module
from stopes.core.launcher import Launcher
from stopes.modules.monolingual.monolingual_sort_dedup import DedupeWithMergeSort
from stopes.pipelines.monolingual.utils import split_list

logger = logging.getLogger("dedup_files")


@dataclass
class DedupConfig:
    output_file: str = MISSING
    input_file: str = MISSING
    num_cpu: int = 40
    mem_gb: int = 300
    timeout_min: int = 14400
    tmp_dir: str = "/tmp"
    field_def: str = "6"


async def launch_dedup(
    file: Path, output_file: Path, config: DedupConfig, launcher: Launcher
) -> Path:

    dedup_module = DedupeWithMergeSort(
        DictConfig(
            {
                "shards": [str(file.resolve())],  # Path are not supported
                "output_file": str(output_file),
                "num_cpu": config.num_cpu,
                "timeout_min": int(config.timeout_min),
                "mem_gb": config.mem_gb,
                "tmp_dir": config.tmp_dir,
                "field_def": config.field_def,
                "do_local_dedup": True,
            }
        )
    )
    deduped_file = await launcher.schedule(dedup_module)
    return Path(deduped_file)


async def dedup_single_file(config: DedupConfig):
    # get a launcher as per the config
    launcher = hydra.utils.instantiate(config.launcher)

    OmegaConf.save(
        config=config,
        f=str(Path(launcher.config_dump_dir) / "dedup_single_file.yaml"),
    )

    logger.info(f"deduping {config.input_file}")

    deduped_file = await launch_dedup(
        Path(config.input_file),
        Path(config.output_file),
        config,
        launcher,
    )
    logger.info(f"done deduping single file to {deduped_file}")


@hydra.main(config_path="conf", config_name="dedup_single_file")
def main(config: DedupConfig) -> None:
    asyncio.run(dedup_single_file(config))


if __name__ == "__main__":
    main()
