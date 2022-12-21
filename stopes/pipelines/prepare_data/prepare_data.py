# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import shutil
import typing as tp
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from stopes.core import utils
from stopes.pipelines.filtering.dataset import Dataset
from stopes.pipelines.prepare_data.binarize import binarize
from stopes.pipelines.prepare_data.build_vocab import build_vocab
from stopes.pipelines.prepare_data.configs import CorporaConfig, PrepareDataConfig
from stopes.pipelines.prepare_data.dedup_sharding import dedup_sharding
from stopes.pipelines.prepare_data.retrieve_data import retrieve_data
from stopes.pipelines.prepare_data.validate import validate

logger = logging.getLogger("prepare_data")


class PrepareData:
    def __init__(self, config: PrepareDataConfig):
        self.config = config
        self.ensure_all_dirs()
        # Cache won't be re-used if you change the output_dir.
        self.config.launcher.cache.caching_dir = Path(self.output_dir) / "cache"
        self.launcher = hydra.utils.instantiate(self.config.launcher)
        self.datasets = self._get_datasets(self.config.corpora)
        self._check_files_exist(self.datasets)
        OmegaConf.save(
            config=config,
            f=str(self.output_dir / "prepare_data.yaml"),
        )
        OmegaConf.set_readonly(self.config, True)

    async def run(self):
        train_src_counts_map, train_tgt_counts_map, train_counts_map = await validate(
            self.datasets, self.launcher
        )
        retrieved_datasets: tp.List[Dataset] = await retrieve_data(
            self.datasets,
            self.config.preprocessing,
            self.launcher,
            self.retrieved_data_dir,
        )
        (src_vocab, tgt_vocab), (
            sharded_train_datasets,
            retrieved_eval_datasets,
            max_num_shards,
        ) = await asyncio.gather(
            build_vocab(
                retrieved_datasets,
                self.config.vocab,
                train_src_counts_map,
                train_tgt_counts_map,
                self.launcher,
                self.vocab_dir,
            ),
            dedup_sharding(
                retrieved_datasets,
                train_counts_map,
                self.config.dedup,
                self.config.sharding,
                self.launcher,
                self.tmp_dir / "sharded",
            ),
        )
        await binarize(
            sharded_train_datasets,
            retrieved_eval_datasets,
            src_vocab,
            tgt_vocab,
            max_num_shards,
            self.config.sharding.binarize_num_workers,
            train_src_counts_map,
            train_tgt_counts_map,
            self.launcher,
            self.tmp_dir / "binarized",
            self.data_bin,
        )

        # Delete tmp_dir.
        shutil.rmtree(self.tmp_dir)

    def ensure_all_dirs(self):
        self.output_dir = Path(self.config.output_dir).resolve()
        self.retrieved_data_dir = self.output_dir / "retrieved_data"
        self.vocab_dir = self.output_dir / "vocab_bin"
        self.data_bin = self.output_dir / "data_bin"
        self.tmp_dir = self.output_dir / "tmp"
        utils.ensure_dir(self.output_dir)
        utils.ensure_dir(self.retrieved_data_dir)
        utils.ensure_dir(self.vocab_dir)
        utils.ensure_dir(self.data_bin)
        utils.ensure_dir(self.tmp_dir)
        utils.ensure_dir(self.tmp_dir / "sharded")
        utils.ensure_dir(self.tmp_dir / "binarized")

    @staticmethod
    def _get_datasets(
        corpora_conf: CorporaConfig,
    ) -> tp.List[Dataset]:
        datasets = []
        for fold in corpora_conf:
            if corpora_conf[fold]:
                for lang_dir in corpora_conf[fold]:
                    for corpus in corpora_conf[fold][lang_dir]:
                        src_file = corpora_conf[fold][lang_dir][corpus].src
                        tgt_file = corpora_conf[fold][lang_dir][corpus].tgt
                        metadata = getattr(
                            corpora_conf[fold][lang_dir][corpus], "metadata", None
                        )
                        dataset = Dataset(
                            src=src_file,
                            tgt=tgt_file,
                            metadata=metadata,
                            lang_dir=lang_dir,
                            fold=fold,
                        )
                        datasets.append(dataset)
        return datasets

    @staticmethod
    def _check_files_exist(datasets: tp.List[Dataset]):
        for dataset in datasets:
            assert Path(dataset.src).exists(), f"Nonexistent source path: {dataset.src}"
            assert Path(dataset.tgt).exists(), f"Nonexistent target path: {dataset.tgt}"


@hydra.main(config_path="conf", config_name="prepare_data")
def main(config: PrepareDataConfig) -> None:
    pipeline = PrepareData(config)
    asyncio.run(pipeline.run())


if __name__ == "__main__":
    main()
