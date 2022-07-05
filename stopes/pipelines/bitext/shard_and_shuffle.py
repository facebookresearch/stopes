# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import lzma
import random
import shutil
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

import hydra
from omegaconf import MISSING, DictConfig, OmegaConf

from stopes.core import stopes_module, utils
from stopes.core.launcher import Launcher
from stopes.modules.preprocess.multiproc_line_processor import (
    MultiprocLineProcessorCallback,
    MultiprocLineProcessorConfig,
    MultiprocLineProcessorModule,
)

logger = logging.getLogger("shard_and_shuffle")


@dataclass
class ShardAndShuffleConfig:
    launcher: DictConfig
    nb_shards: int
    output_dir: str = MISSING
    input_file: str = MISSING
    log_every: int = 10_000


@dataclass
class Shard:
    outfd: tp.TextIO
    path: Path
    cnt: int = 0


class ShardAndShuffleMC(MultiprocLineProcessorCallback):
    """
    This module splits a single input file into multiple shards,
    shuffling on the fly
    """

    def __init__(
        self,
        # set by LineProcessorModule
        outfile_prefix: str,
        input_file: str,
        input_file_idx: int,
        output_dir: str,
        offset_start: tp.Optional[int],
        offset_end: tp.Optional[int],
        merging: bool,
        # our params
        nb_shards: int,
        log_every: int = 10_000,
    ):
        super().__init__(
            outfile_prefix=outfile_prefix,
            input_file=input_file,
            input_file_idx=input_file_idx,
            output_dir=output_dir,
            offset_start=offset_start,
            offset_end=offset_end,
            merging=merging,
        )

        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = out_dir / (
            f"{self.outfile_prefix}.{input_file_idx:03d}.{offset_start}_{offset_end}.xz"
        )
        self.nb_shards = nb_shards
        self.log_every = log_every

        self.shards = []

    def __enter__(self):
        for i in range(self.nb_shards):
            shard = self.output_file.with_suffix(f".{i:03d}.xz")
            fp = utils.open(str(shard), mode="wt", encoding="utf-8")
            self.shards.append(
                Shard(
                    path=shard,
                    outfd=fp,
                )
            )
            print(f"Initializing shard {i} at {shard}")
        return self

    def __exit__(self, *exc):
        for idx, f in enumerate(self.shards):
            f.outfd.close()
            f.outfd = None

    def process_lines(self, lines_with_number: tp.Iterator[tp.Tuple[int, str]]) -> None:
        """
        process a batch of lines, filter them, dedup them locally
        and write them to the output file
        """
        random.seed(42)
        for idx, line in lines_with_number:
            r = random.randint(0, self.nb_shards - 1)
            self.shards[r].cnt += 1
            print(line.strip(), file=self.shards[r].outfd)
            if idx % self.log_every == 0:
                print(f"{self.output_file} Processed {idx} lines")

    def final_result(self) -> tp.List[Shard]:
        return self.shards

    def merge_results(self, splits: tp.List[tp.List[Shard]]) -> tp.List[Path]:

        merge = Path(self.output_dir) / (
            f"{self.outfile_prefix}.{self.input_file_idx:03d}"
        )
        final_shards = []
        for i in range(self.nb_shards):
            p = merge.with_suffix(f".{i:03d}.xz")

            final_shards.append(
                Shard(
                    path=p,
                    outfd=lzma.open(
                        str(p),
                        mode="wb",
                    ),
                )
            )
        for shards in splits:
            for idx, s in enumerate(shards):
                f_shard = final_shards[idx]
                f_shard.cnt += s.cnt
                with lzma.open(str(s.path), mode="rb") as shard_fd:
                    shutil.copyfileobj(shard_fd, f_shard.outfd)
                    f_shard.outfd.write("\n".encode("utf-8"))

        for idx, shard in enumerate(final_shards):
            shard.outfd.close()
            print(
                f"Shard nb {idx} - filename: {shard.path}, nb elements:" f" {shard.cnt}"
            )

        return [s.path for s in final_shards]


async def shard_and_shuffle(config: ShardAndShuffleConfig):
    launcher = hydra.utils.instantiate(config.launcher)

    OmegaConf.save(
        config=config,
        f=str(Path(launcher.config_dump_dir) / "shard_and_shuffle.yaml"),
    )

    file_processor = MultiprocLineProcessorModule(
        config=MultiprocLineProcessorConfig(
            line_processor=DictConfig(
                {
                    "_target_": "stopes.pipelines.bitext.shard_and_shuffle.ShardAndShuffleMC",
                    "nb_shards": config.nb_shards,
                    "log_every": config.log_every,
                }
            ),
            custom_name=f"shard_and_shuffle",
            output_dir=str(config.output_dir),
            outfile_prefix="shard",
            shards=[config.input_file],
            requirements=stopes_module.DistributedRequirements(
                nodes=1,
                mem_gb=getattr(config, "mem_gb", 1),
                tasks_per_node=1,
                cpus_per_task=getattr(config, "num_cpu", 40),
                gpus_per_node=0,
                timeout_min=getattr(config, "timeout_min", 14400),
            ),
            tmp_dir=str(config.tmp_dir),
        )
    )

    logger.info(f"Sharding and shuffling {config.input_file}")
    shards = await launcher.schedule(file_processor)
    logger.info(f"Done sharding and shuffling to shards")
    for f in shards:
        print(f"\t{f}")


@hydra.main(config_path="conf", config_name="shard_and_shuffle")
def main(config: ShardAndShuffleConfig) -> None:
    asyncio.run(shard_and_shuffle(config))


if __name__ == "__main__":
    main()
