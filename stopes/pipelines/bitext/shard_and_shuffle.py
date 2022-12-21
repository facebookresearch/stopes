# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import lzma
import math
import random
import shutil
import subprocess
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import numpy as np
from fairseq.file_chunker_utils import Chunker
from more_itertools import iterate
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
        min_lines_per_shard: int = None,  # optional param specifying a minimum number of lines per output shard - resulting in sampling with replacement
        total_file_lines: int = None,  # optional param which denotes total number of lines in input file and is used if min_lines_per_shard is not None
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

        self.min_lines_per_shard = min_lines_per_shard
        self.total_file_lines = total_file_lines

        if self.min_lines_per_shard is not None:
            if self.total_file_lines is None:
                self.total_file_lines = utils.count_lines(self.input_file)
            else:
                self.total_file_lines = total_file_lines

        self.chunk_num_lines = 0
        if (
            self.min_lines_per_shard is not None
            and self.offset_end is not None
            and self.offset_start is not None
        ):
            with Chunker(
                str(self.input_file), self.offset_start, self.offset_end
            ) as line_iterator:
                for _ in line_iterator:
                    self.chunk_num_lines += 1

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
        shards input file into output shards by sampling with replacement if self.min_lines_per_shard is defined, or otherwise
        randomly writes out each line in the input file to an output shard
        """
        random.seed(42)
        if self.chunk_num_lines == 0:
            return
        # Sampling with replacement to satisfy minimum lines per shard requirement.

        # Algorithm: We have `total_lines` number of sentences in the file and `num_shards` amount of target languages, and `chunk_num_lines` sentences in the
        # current chunk being processed by the LineProcessor, and want a total of `min_lines_per_shard` sentences in each output shard.
        # We want to sample `chunk_lines_per_shard` sentences from the current chunk for each output shard, computed as a
        # proportion of min_lines_per_shard.
        # chunk_lines_per_shard * T = total number of lines needed from current chunk
        # (chunk_lines_per_shard * T) / chunk_num_lines = how many times should each line in the current chunk be assigned
        shard_indices = np.arange(self.nb_shards)
        if self.min_lines_per_shard:
            chunk_lines_per_shard = (
                self.chunk_num_lines / self.total_file_lines
            ) * self.min_lines_per_shard
            num_shards_per_line = (
                chunk_lines_per_shard * self.nb_shards
            ) / self.chunk_num_lines
            round_up_freq = round(num_shards_per_line % 1, 3) * 1000
        for idx, line in lines_with_number:
            if self.min_lines_per_shard:
                # Select `num_shards_per_line` shards.
                # If that value is fractional, round up round_up_freq/10 of the time
                if idx % 1000 < round_up_freq:
                    n = math.ceil(num_shards_per_line)
                else:
                    n = math.floor(num_shards_per_line)
                n = min(n, len(shard_indices))
                selected_shards = np.random.permutation(shard_indices)[:n]
            else:
                selected_shards = [random.randint(0, self.nb_shards - 1)]
            # write out this line to selected_shards
            for index in selected_shards:
                self.shards[index].cnt += 1
                print(line.strip(), file=self.shards[index].outfd)
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
                    if s.cnt != 0:
                        shutil.copyfileobj(shard_fd, f_shard.outfd)
                        f_shard.outfd.write("\n".encode("utf-8"))

        for idx, shard in enumerate(final_shards):
            shard.outfd.close()
            print(
                f"Shard nb {idx} - filename: {shard.path}, nb elements:" f" {shard.cnt}"
            )

        return [s.path for s in final_shards]


async def shard_and_shuffle(
    config: ShardAndShuffleConfig,
    min_lines_per_shard: int = None,
    launcher: Launcher = None,
):
    if launcher is None:
        launcher = hydra.utils.instantiate(config.launcher)

    OmegaConf.save(
        config=config,
        f=str(Path(launcher.config_dump_dir) / "shard_and_shuffle.yaml"),
    )

    if min_lines_per_shard is not None:
        total_file_lines = utils.count_lines(config.input_file)

    file_processor = MultiprocLineProcessorModule(
        config=MultiprocLineProcessorConfig(
            line_processor=DictConfig(
                {
                    "_target_": "stopes.pipelines.bitext.shard_and_shuffle.ShardAndShuffleMC",
                    "nb_shards": config.nb_shards,
                    "log_every": config.log_every,
                    "min_lines_per_shard": min_lines_per_shard,
                    "total_file_lines": total_file_lines,
                }
            ),
            custom_name=f"shard_and_shuffle",
            output_dir=str(config.output_dir),
            outfile_prefix=config.outfile_prefix,
            shards=[config.input_file],
            requirements=stopes_module.Requirements(
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
    return shards


@hydra.main(config_path="conf", config_name="shard_and_shuffle")
def main(config: ShardAndShuffleConfig) -> None:
    asyncio.run(shard_and_shuffle(config))


if __name__ == "__main__":
    main()
