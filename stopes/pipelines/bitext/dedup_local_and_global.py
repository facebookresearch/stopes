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
from importlib.metadata import files
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
class DedupLocalAndGlobalConfig:
    output_file: str = MISSING
    input_files_glob: str = MISSING
    num_cpu: int = 40
    mem_gb: int = 300
    timeout_min: int = 14400
    tmp_dir: str = "/tmp"
    field_def: str = "6"


def build_dedup_module(
    files_to_dedup: tp.List[Path],
    do_local_dedup: bool,
    config: DedupLocalAndGlobalConfig,
) -> DedupeWithMergeSort:

    dedup_module = DedupeWithMergeSort(
        DictConfig(
            {
                "shards": files_to_dedup,  # Path are not supported
                "output_file": config.output_file,
                "num_cpu": config.num_cpu,
                "timeout_min": int(config.timeout_min),
                "mem_gb": config.mem_gb,
                "tmp_dir": config.tmp_dir,
                "field_def": config.field_def,
                "do_local_dedup": do_local_dedup,
            }
        )
    )

    return dedup_module


async def dedup_local_and_global(config: DedupLocalAndGlobalConfig):
    # get a launcher as per the config
    launcher = hydra.utils.instantiate(config.launcher)

    OmegaConf.save(
        config=config,
        f=str(Path(launcher.config_dump_dir) / "dedup_local_and_global.yaml"),
    )

    files_to_dedup = glob.glob(config.input_files_glob)
    logger.info(f"Found {len(files_to_dedup)} files: {files_to_dedup}")

    if config.do_local_dedup:
        logger.info(f"Starting local dedup")
        local_pass_module = build_dedup_module(
            files_to_dedup,
            True,
            config,
        )
        localy_deduped_files = await launcher.schedule(local_pass_module)
        logger.info(f"Local dedup complete, starting global dedup")

    else:
        localy_deduped_files = files_to_dedup

    logger.info(f"Starting global dedup")
    global_pass_module = build_dedup_module(
        localy_deduped_files,
        False,
        config,
    )
    globally_deduped_file = await launcher.schedule(global_pass_module)
    logger.info(f"Global dedup complete, results are in {globally_deduped_file}")


@hydra.main(config_path="conf", config_name="dedup_local_and_global")
def main(config: DedupLocalAndGlobalConfig) -> None:
    """
    This pipeline can handle both
    - pre-sorted input files (single step, merge at a global level)
    - non sorted input files (2 steps, local dedup then global merge)
    Behavior is controlled by do_local_dedup
    """

    asyncio.run(dedup_local_and_global(config))


if __name__ == "__main__":
    main()
