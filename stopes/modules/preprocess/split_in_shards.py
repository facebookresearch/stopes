# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import lzma
import random
import shutil
import typing as tp
from dataclasses import dataclass
from pathlib import Path

from stopes.core import utils
from stopes.modules.preprocess.multiproc_bitext_processor import (
    MultiprocBitextProcessorCallback,
)
from stopes.modules.preprocess.multiproc_line_processor import (
    MultiprocLineProcessorCallback,
)
from stopes.pipelines.filtering.dataset import DatasetLine

logger = logging.getLogger("split_in_shards")


@dataclass
class Shard:
    outfd: tp.TextIO
    path: Path
    cnt: int = 0


class SplitInShardsMC(MultiprocLineProcessorCallback):
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
        shards input file into output shards by randomly writing out each line in the input file to an output shard
        """
        random.seed(42)

        for idx, line in lines_with_number:
            index = random.randint(0, self.nb_shards - 1)
            self.shards[index].cnt += 1
            print(line.strip(), file=self.shards[index].outfd)
            if idx % self.log_every == 0:
                print(f"{self.output_file} Processed {idx} lines")

    def final_result(self) -> tp.List[Shard]:
        return self.shards

    def merge_results(self, splits: tp.List[tp.List[Shard]]) -> tp.List[Path]:
        merge = Path(self.output_dir) / (
            f"{self.outfile_prefix}.{self.input_file_idx:03d}.suffix_to_remove"  # Path.with_suffix will realy remove it
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

        for idx, shard in enumerate(final_shards):
            shard.outfd.close()
            print(
                f"Shard nb {idx} - filename: {shard.path}, nb elements:" f" {shard.cnt}"
            )

        return [s.path for s in final_shards], [s.cnt for s in final_shards]


class SplitInShardsParallelMC(MultiprocBitextProcessorCallback):
    """
    This module splits a pair of parallel input files into multiple shards,
    shuffling on the fly
    """

    def __init__(
        self,
        # set by LineProcessorModule
        outfile_prefix: str,
        src_input_file: str,
        tgt_input_file: str,
        input_files_idx: int,
        output_dir: str,
        line_offset_start: tp.Optional[int],
        line_offset_end: tp.Optional[int],
        outfile_postfix: str = "",
        merging: bool = False,
        # our params
        nb_shards: int = 1,
        log_every: int = 10_000,
    ):
        super().__init__(
            outfile_prefix=outfile_prefix,
            src_input_file=src_input_file,
            tgt_input_file=tgt_input_file,
            input_files_idx=input_files_idx,
            output_dir=output_dir,
            line_offset_start=line_offset_start,
            line_offset_end=line_offset_end,
            outfile_postfix=outfile_postfix,
            merging=merging,
        )

        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.output_files = [
            out_dir
            / (
                f"{self.outfile_prefix}.{input_files_idx:03d}_{j}.{line_offset_start}_{line_offset_end}.xz"
            )
            for j in range(2)
        ]
        self.nb_shards = nb_shards
        self.log_every = log_every

        self.shards = [[], []]

    def __enter__(self):
        for i in range(self.nb_shards):
            for j in range(2):
                shard = self.output_files[j].with_suffix(f".{i:03d}.xz")
                fp = utils.open(str(shard), mode="wt", encoding="utf-8")
                self.shards[j].append(Shard(path=shard, outfd=fp))
                print(f"Initializing shard {i}/{j} at {shard}")
        return self

    def __exit__(self, *exc):
        for shards in self.shards:
            for f in shards:
                f.outfd.close()
                f.outfd = None

    def process_lines(
        self, dataset_reader: tp.Generator[DatasetLine, None, None]
    ) -> None:
        """
        shards input file into output shards by randomly writing out each line in the input file to an output shard.
        """
        random.seed(42)

        for idx, line in enumerate(dataset_reader):
            index = random.randint(0, self.nb_shards - 1)
            self.shards[0][index].cnt += 1
            self.shards[1][index].cnt += 1
            print(line.src.strip(), file=self.shards[0][index].outfd)
            print(line.tgt.strip(), file=self.shards[1][index].outfd)
            if idx % self.log_every == 0:
                print(
                    f"{self.output_files[0]} Processed {idx} lines, e.g. '{line.src.strip()}' + '{line.tgt.strip()}'"
                )

    def final_result(self) -> tp.List[tp.List[Shard]]:
        return self.shards

    def merge_results(
        self, splits: tp.List[tp.List[tp.List[Shard]]]
    ) -> tp.List[tp.List[Path]]:
        # returns two lists of filenames
        merge = Path(self.output_dir) / (
            f"{self.outfile_prefix}.{self.input_files_idx:03d}.suffix_to_remove"  # Path.with_suffix will realy remove it
        )
        final_shards = [[], []]
        for i in range(self.nb_shards):
            for j in range(2):
                p = merge.with_suffix(f".{i:03d}.{j}.xz")
                final_shards[j].append(
                    Shard(path=p, outfd=lzma.open(str(p), mode="wb")),
                )
        for shards in splits:  # loop over workers
            for j in range(2):  # loop over src/tgt texts
                for idx, s in enumerate(shards[j]):  # loop over shards
                    f_shard = final_shards[j][idx]
                    f_shard.cnt += s.cnt
                    with lzma.open(str(s.path), mode="rb") as shard_fd:
                        shutil.copyfileobj(shard_fd, f_shard.outfd)

        for j in range(2):
            for idx, shard in enumerate(final_shards[j]):
                shard.outfd.close()
                print(
                    f"Shard nb {j}/{idx} - filename: {shard.path}, nb elements:"
                    f" {shard.cnt}"
                )
        main_files, meta_files = [[s.path for s in ss] for ss in final_shards]
        return main_files, meta_files, [s.cnt for s in final_shards[0]]
