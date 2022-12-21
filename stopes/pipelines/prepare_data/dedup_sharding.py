# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import contextlib
import itertools
import logging
import math
import random
import typing as tp
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import xxhash
from omegaconf.omegaconf import MISSING

from stopes.core import utils
from stopes.core.launcher import Launcher
from stopes.core.stopes_module import Requirements, StopesModule
from stopes.pipelines.filtering.dataset import Dataset
from stopes.pipelines.monolingual.utils.text_normalizer import normalize_for_dedup
from stopes.pipelines.prepare_data.configs import DedupConfig, DedupType, ShardingConfig


@dataclass
class DedupShardingJob:
    num_shards: int
    eval_datasets: tp.Optional[tp.List[Dataset]] = None
    train_datasets: tp.List[Dataset] = MISSING


@dataclass
class DedupShardingConfig:
    dedup_config: DedupConfig
    output_dir: Path
    dedup_sharding_jobs: tp.List[DedupShardingJob] = MISSING


class DedupSharding(StopesModule):
    def __init__(
        self,
        config: DedupShardingConfig,
    ):
        super().__init__(config, DedupShardingConfig)

    def array(self):
        return self.config.dedup_sharding_jobs

    def requirements(self) -> Requirements:
        return Requirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=1,
            timeout_min=2 * 24 * 60,
        )

    def _initialize_seen(self):
        self._seen_src: tp.Set[int] = set()
        self._seen_tgt: tp.Set[int] = set()
        self._seen_pair: tp.Set[int] = set()

    def _already_seen(
        self, src_line: str, tgt_line: str, dedup_type: DedupType
    ) -> bool:
        norm_src: str = normalize_for_dedup(src_line)
        norm_tgt: str = normalize_for_dedup(tgt_line)

        if len(norm_src) < 1 or len(norm_tgt) < 1:
            return True

        norm_pair: str = "\t".join((norm_src, norm_tgt))
        hashed_pair: int = xxhash.xxh3_64_intdigest(norm_pair)

        if hashed_pair in self._seen_pair:
            return True

        self._seen_pair.add(hashed_pair)

        # We only check the sentence pair for "neither".
        if dedup_type == DedupType.neither:
            return False

        hashed_src: int = xxhash.xxh3_64_intdigest(norm_src)
        hashed_tgt: int = xxhash.xxh3_64_intdigest(norm_tgt)

        already_seen = False
        if dedup_type == DedupType.both or dedup_type == DedupType.src:
            if hashed_src in self._seen_src:
                already_seen = True
            else:
                self._seen_src.add(hashed_src)
        if dedup_type == DedupType.both or dedup_type == DedupType.tgt:
            if hashed_tgt in self._seen_tgt:
                already_seen = True
            else:
                self._seen_tgt.add(hashed_tgt)
        return already_seen

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> tp.List[Dataset]:
        assert iteration_value is not None, "iteration value is null"
        self.logger = logging.getLogger("stopes.prepare_data.dedup_sharding")

        dedup_sharding_job: DedupShardingJob = iteration_value
        lang_dir = dedup_sharding_job.train_datasets[0].lang_dir
        src, tgt = lang_dir.split("-")
        num_shards = dedup_sharding_job.num_shards
        output_dir = self.config.output_dir

        self.logger.info(f"Dedup-sharding for {lang_dir} to {num_shards} shards.")

        # Initialize all the seen sets.
        self._initialize_seen()

        if dedup_sharding_job.eval_datasets:
            for eval_dataset in dedup_sharding_job.eval_datasets:
                with utils.open(eval_dataset.src, "rt") as s_f, utils.open(
                    eval_dataset.tgt, "rt"
                ) as t_f:
                    for src_line, tgt_line in zip(s_f, t_f):
                        self._already_seen(src_line, tgt_line, DedupType.both)

        sharded_datasets: tp.List[Dataset] = []
        for train_dataset in dedup_sharding_job.train_datasets:
            self.logger.info(f"Dedup-sharding for {train_dataset.fold}")

            with ExitStack() as stack:
                source_shards = []
                target_shards = []
                metadata_shards = []

                for i in range(num_shards):
                    src_outfile = (
                        output_dir
                        / f"{train_dataset.fold}.shard{i:03d}.{src}-{tgt}.{src}"
                    )
                    tgt_outfile = (
                        output_dir
                        / f"{train_dataset.fold}.shard{i:03d}.{src}-{tgt}.{tgt}"
                    )
                    metadata_outfile = None
                    source_shards.append(
                        stack.enter_context(utils.open(src_outfile, "wt"))
                    )
                    target_shards.append(
                        stack.enter_context(utils.open(tgt_outfile, "wt"))
                    )
                    if train_dataset.metadata:
                        metadata_outfile = (
                            output_dir
                            / f"{train_dataset.fold}.shard{i:03d}.{src}-{tgt}.metadata"
                        )
                        metadata_shards.append(
                            stack.enter_context(utils.open(metadata_outfile, "wt"))
                        )
                    sharded_datasets.append(
                        Dataset(
                            src=str(src_outfile),
                            tgt=str(tgt_outfile),
                            metadata=str(metadata_outfile)
                            if metadata_outfile
                            else None,
                            lang_dir=lang_dir,
                            fold=train_dataset.fold,
                        )
                    )

                random.seed(0)
                seen_lines = 0
                num_lines = 0
                with utils.open(train_dataset.src, "rt") as s_f, utils.open(
                    train_dataset.tgt, "rt"
                ) as t_f, utils.open(
                    train_dataset.metadata, "rt"
                ) if train_dataset.metadata else contextlib.nullcontext(
                    itertools.repeat(None)
                ) as m_f:
                    for src_line, tgt_line, metadata_line in zip(s_f, t_f, m_f):
                        shard_id = random.randint(0, num_shards - 1)
                        if not self._already_seen(
                            src_line,
                            tgt_line,
                            self.config.dedup_config.dedup_type,
                        ):
                            source_shards[shard_id].write(src_line)
                            target_shards[shard_id].write(tgt_line)
                            if metadata_line is not None:
                                metadata_shards[shard_id].write(metadata_line)
                        else:
                            seen_lines += 1
                        num_lines += 1

                self.logger.info(
                    f"Removed {seen_lines}/{num_lines} lines ({(seen_lines * 100 / num_lines):.2f} %)"
                )

        return sharded_datasets

    def validate(
        self,
        output: tp.Any,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> bool:
        sharded_datasets: tp.List[Dataset] = output
        for dataset in sharded_datasets:
            if not (Path(dataset.src).exists() and Path(dataset.tgt).exists()):
                return False
        return True


def get_num_shards_per_lang_dir(
    train_counts_map: tp.Dict[str, int],
    sharding_config: ShardingConfig,
) -> tp.Dict[str, int]:
    max_num_shards: int = math.ceil(
        sum(train_counts_map.values()) / sharding_config.max_examples_per_shard
    )
    num_shards_map: tp.Dict[str, int] = {
        lang_dir: min(
            math.ceil(num_lines / sharding_config.smallest_shard), max_num_shards
        )
        for lang_dir, num_lines in train_counts_map.items()
    }
    return num_shards_map


async def dedup_sharding(
    retrieved_datasets: tp.List[Dataset],
    train_counts_map: tp.Dict[str, int],
    dedup_config: DedupConfig,
    sharding_config: ShardingConfig,
    launcher: Launcher,
    output_dir: Path,
) -> tp.Tuple[tp.List[Dataset], tp.List[Dataset], int]:

    num_shards_per_lang_dir: tp.Dict[str, int] = get_num_shards_per_lang_dir(
        train_counts_map, sharding_config
    )
    max_num_shards = max(num_shards_per_lang_dir.values())

    train_datasets_per_lang_dir = defaultdict(list)
    eval_datasets_per_lang_dir = defaultdict(list)
    for dataset in retrieved_datasets:
        if dataset.fold.startswith("train"):
            train_datasets_per_lang_dir[dataset.lang_dir].append(dataset)
        else:
            eval_datasets_per_lang_dir[dataset.lang_dir].append(dataset)

    dedup_sharding_jobs = []
    for lang_dir, train_datasets in train_datasets_per_lang_dir.items():
        eval_datasets = eval_datasets_per_lang_dir.get(lang_dir)
        # Look for the reversed direction.
        if eval_datasets is None:
            src, tgt = lang_dir.split("-")
            reversed_lang_dir = f"{tgt}-{src}"
            eval_datasets = eval_datasets_per_lang_dir.get(reversed_lang_dir)

        num_shards = num_shards_per_lang_dir[lang_dir]
        if dedup_config.cross_fold:
            # dedup across folds together per lang_dir.
            dedup_sharding_jobs.append(
                DedupShardingJob(
                    num_shards=num_shards,
                    eval_datasets=eval_datasets,
                    train_datasets=train_datasets,
                )
            )
        else:
            # dedup each fold separately per lang_dir.
            for train_dataset in train_datasets:
                dedup_sharding_jobs.append(
                    DedupShardingJob(
                        num_shards=num_shards,
                        eval_datasets=eval_datasets,
                        train_datasets=[train_dataset],
                    )
                )

    dedup_sharding_module = DedupSharding(
        DedupShardingConfig(
            dedup_config=dedup_config,
            output_dir=output_dir,
            dedup_sharding_jobs=dedup_sharding_jobs,
        )
    )
    dedup_sharding_results = await launcher.schedule(dedup_sharding_module)

    # Flatten lists
    sharded_train_datasets: tp.List[Dataset] = [
        sharded_dataset
        for sharded_datasets in dedup_sharding_results
        for sharded_dataset in sharded_datasets
    ]
    retrieved_eval_datasets: tp.List[Dataset] = [
        dataset
        for datasets in eval_datasets_per_lang_dir.values()
        for dataset in datasets
    ]
    return sharded_train_datasets, retrieved_eval_datasets, max_num_shards
