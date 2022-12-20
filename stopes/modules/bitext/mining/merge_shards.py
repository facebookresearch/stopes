# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import contextlib
import logging
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import xxhash

from stopes.core import utils
from stopes.core.stopes_module import Requirements, StopesModule

logger = logging.getLogger("merge_shards")


@dataclass
class MergeShardsConfig:
    src_lang: str
    tgt_lang: str
    output_dir: str
    # `pairs` is the list of pairs of paths to text and meta files,
    # such as [("shard1.txt", "shard1_meta.txt"), ("shard2.txt", "shard2_meta.txt")]
    # If there is no metadata, the second element of each pair should be None.
    pairs: tp.List[tp.Any]  # tp.List[tp.Tuple[Path, tp.Optional[Path]]]
    remove_tmp_files: bool = False


class ShardForMerge:
    """Represent an input shard opened for merge.
    Both input and output shards are in decreasing order of match scores, and this object helps manage that."""

    def __init__(self, text_path: Path, meta_path: tp.Optional[Path]):
        self.text_path = text_path
        self.meta_path = meta_path
        self.text_file = None
        self.meta_file = None
        self.next_line = None
        self.next_score = None
        self.next_meta = None

    def __enter__(self):
        self.text_file = utils.open(self.text_path, "rt")
        self.meta_file = utils.open(self.meta_path, "rt") if self.meta_path else None
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()

    def try_read_next_line(self, seen: tp.Set[int]) -> bool:
        """Try to read and remember the next line with a unique bitext.
        In case of success, return True. Otherwise, close the files and return False."""
        while True:
            # reed both text and meta to keep the two cursors parallel
            line = self.text_file.readline()
            meta_line = (
                self.meta_file.readline() if self.meta_file is not None else None
            )
            # exit and close the files in case of end-of-file
            if not line:
                self.next_line = None
                self.next_meta = None
                self.next_score = None
                self.close()
                return False
            # continue reading in case of duplicates (including duplicates across shards)
            score, bitext = line.split("\t", 1)
            h = xxhash.xxh3_64_intdigest(bitext.strip())
            if h in seen:
                continue
            seen.add(h)
            break
        self.next_line = line
        self.next_score = float(score)
        self.next_meta = meta_line
        return True

    def copy_line(self, text_file: tp.TextIO, meta_file: tp.Optional[tp.TextIO]):
        text_file.write(self.next_line)
        if self.next_meta is not None:
            meta_file.write(self.next_meta)

    def close(self):
        self.text_file.close()
        if self.meta_file is not None:
            self.meta_file.close()


class MergeShardsModule(StopesModule):
    def __init__(
        self,
        config: MergeShardsConfig,
    ):
        super().__init__(config, MergeShardsConfig)
        self.config: MergeShardsConfig

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> tp.Tuple[Path, Path]:
        """
        Merge the sorted input shards into one sorted output shard:
        1. Open all the shards for read simultaneously;
        2. In a loop, copy the line from the shard with the largest score;
        3. When some input files are depleted, remove them.
        This is always a non-parallel run (without sharding).
        """
        first_text, first_meta = self.config.pairs[0]
        extension = Path(first_text).suffix
        prefix = str(
            Path(self.config.output_dir)
            / f"{self.config.src_lang}_{self.config.tgt_lang}"
        )
        out_path_text = Path(prefix + f"_bitext{extension}")
        out_path_meta = Path(prefix + f"_bimeta{extension}")
        out_path_text.parent.mkdir(parents=True, exist_ok=True)
        has_meta = all(pair[1] for pair in self.config.pairs)
        if not has_meta:
            out_path_meta = None

        with contextlib.ExitStack() as stack:
            shards_for_merge: tp.List[ShardForMerge] = [
                stack.enter_context(
                    ShardForMerge(text_path, meta_path if has_meta else None)
                )
                for text_path, meta_path in self.config.pairs
            ]
            seen_hashes: tp.Set[int] = set()
            # read the first line of each shard and remove the empty shards
            # (theoretically, for some shard we might have zero unique matched data)
            shards_for_merge = [
                shard
                for shard in shards_for_merge
                if shard.try_read_next_line(seen=seen_hashes)
            ]
            f_text = stack.enter_context(utils.open(out_path_text, "wt"))
            f_meta = stack.enter_context(
                utils.open(out_path_meta, "wt")
                if has_meta
                else contextlib.nullcontext()
            )
            while len(shards_for_merge) > 0:
                # choose the shard with the maximal score
                best_shard = shards_for_merge[0]
                for shard in shards_for_merge:
                    if shard.next_score > best_shard.next_score:
                        best_shard = shard
                # copy the line from the chosen shard and advance its cursor
                best_shard.copy_line(f_text, f_meta)
                if not best_shard.try_read_next_line(seen=seen_hashes):
                    shards_for_merge.remove(best_shard)
        return out_path_text, out_path_meta

    def version(self):
        return "0.1"

    def requirements(self):
        return Requirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=4,
            timeout_min=60 * 24,
        )
