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

from stopes.core.launcher import Launcher
from stopes.modules.monolingual.monolingual_sort_dedup import DedupeWithMergeSort
from stopes.pipelines.monolingual.utils import split_list

logger = logging.getLogger("dedup_files")


@dataclass
class DedupConfig:
    output_file: str = MISSING
    glob: str = MISSING
    merge_dir: str = MISSING
    output_file: str = MISSING
    num_files: int = 10
    num_cpu: int = 40
    mem_gb: int = 300
    timeout_min: int = 14400
    tmp_dir: str = "/tmp"
    field_def: str = "6"
    resort_files: bool = False
    process_locally: bool = False


async def launch_dedup(
    files: tp.List[Path],
    output_file: Path,
    config: DedupConfig,
    launcher: Launcher,
    do_local_dedup: bool = False,
    process_locally: bool = False,
    compress: bool = True,
) -> tp.Union[tp.List[Path], Path]:

    dedup_module = DedupeWithMergeSort(
        DictConfig(
            {
                "shards": [str(f.resolve()) for f in files],  # Path are not supported
                "output_file": str(output_file),
                "num_cpu": config.num_cpu,
                "timeout_min": int(config.timeout_min),
                "mem_gb": config.mem_gb,
                "tmp_dir": config.tmp_dir,
                "field_def": config.field_def,
                "do_local_dedup": do_local_dedup,
                "process_locally": process_locally,
                "compress": compress,
            }
        )
    )
    try:
        deduped_file = await launcher.schedule(dedup_module)
    except Exception as e:
        logger.error(f"shards failed {files}", exc_info=e)
        return None
    if type(deduped_file) is list:
        return [Path(f) for f in deduped_file]

    return Path(deduped_file)


async def dedup_files(config: DedupConfig):
    # get a launcher as per the config
    launcher = hydra.utils.instantiate(config.launcher)

    OmegaConf.save(
        config=config,
        f=str(Path(launcher.config_dump_dir) / "dedup_files.yaml"),
    )

    all_files = [Path(f) for f in glob.glob(config.glob)]
    logger.info(f"found {len(all_files)} files")

    if len(all_files) == 1:
        return all_files[0]

    merge_dir = Path(config.merge_dir) / Path(config.output_file).stem
    merge_dir.mkdir(parents=True, exist_ok=True)

    if config.resort_files:
        all_files = await launch_dedup(
            all_files,
            merge_dir / f"partial_dedup.first-sort.txt",
            config,
            launcher,
            do_local_dedup=True,
            process_locally=config.process_locally,
            compress=False,
        )

    assert all([f is not None for f in all_files]), "some files dedup failed"

    current_chunks = list(split_list(all_files, config.num_files))

    loop_cnt = 0
    while len(current_chunks) > 1:
        logger.info(f"dedupping {len(current_chunks)} chunks in iteration {loop_cnt}.")
        new_files = await asyncio.gather(
            *[
                launch_dedup(
                    files,
                    merge_dir / f"partial_dedup.{loop_cnt}-{idx}.txt",
                    config,
                    launcher,
                    do_local_dedup=False,
                    process_locally=config.process_locally,
                    compress=False,
                )
                for idx, files in enumerate(current_chunks)
            ]
        )

        assert all([f is not None for f in new_files]), "some files dedup failed"
        current_chunks = list(split_list(new_files, config.num_files))
        loop_cnt += 1

    logger.info(f"done iterating after {loop_cnt} loops.")

    last_files = current_chunks[0]
    if len(last_files) == 1:
        end = last_files[0]
        logger.info(f"moving {end} to final output")
        shutil.move(end, config.output_file)
    else:
        logger.info(f"one last merge with {len(last_files)} files.")
        #  one last merge
        end = await launch_dedup(
            last_files,
            Path(config.output_file),
            config,
            launcher,
            compress=False,
        )

    logger.info(f"done iterative deduping to {end}")


@hydra.main(config_path="conf", config_name="dedup_files")
def main(config: DedupConfig) -> None:
    if config.wandb:
        run = wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            config=OmegaConf.to_container(config),
        )
        run.name = f"dedup_{run.name}"

    asyncio.run(dedup_files(config))


if __name__ == "__main__":
    main()
