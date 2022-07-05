# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import glob
import logging
import os
import re
import typing as tp
from dataclasses import dataclass, field

import hydra
from omegaconf import DictConfig

from stopes.core.stopes_module import StopesModule
from stopes.modules.bitext.indexing.sample_embedding_module import (
    SampleEmbeddingModule,
    SampleEmbeddingModuleConfig,
)
from stopes.modules.bitext.mining.calculate_distances import DistanceType
from stopes.utils.data_utils import DataConfig
from stopes.utils.mining_utils import extract_shard_id, get_faiss_index_type

logger = logging.getLogger("global_mining")


@dataclass
class LangConfig:
    index_type: tp.Optional[str] = None
    existing_index_path: tp.Optional[str] = None
    existing_embedding_glob: tp.Optional[str] = None
    spm_model: tp.Optional[str] = None
    spm_vocab: tp.Optional[str] = None
    encoder_model: tp.Optional[str] = None


@dataclass
class GlobalMiningConfig:
    launcher: DictConfig
    data: DataConfig
    model_dir: str
    embed_text: DictConfig
    train_index: DictConfig
    populate_index: DictConfig
    merge_indexes: DictConfig
    embedding_sample: DictConfig
    src_lang: str
    tgt_lang: str
    lang_configs: tp.Dict[str, LangConfig] = field(default_factory=dict)


class GlobalMiningPipeline:
    def __init__(
        self,
        config: GlobalMiningConfig,
    ):
        self.config = config
        self.launcher = hydra.utils.instantiate(config.launcher)
        self.src_index_type = self._get_index_type(self.config.src_lang)
        self.tgt_index_type = self._get_index_type(self.config.tgt_lang)

    def _get_index_type(self, lang: str) -> str:
        lang_config = getattr(self.config.lang_configs, lang, None)
        idx_type = None
        if lang_config:
            idx_type = getattr(lang_config, "index_type", None)

        if idx_type:
            return idx_type

        # this required a precomputed line count file
        return get_faiss_index_type(
            lang=lang,
            data_cfg=self.config.data,
        )

    def _find_data_shards(
        self,
        lang: str,
    ) -> tp.List[str]:

        if hasattr(self.config.data, "shard_glob"):
            glob_with_replacement = self.config.data.shard_glob.format(lang=lang)
            shards = glob.glob(glob_with_replacement)
        else:
            shard_name = (
                f"{self.config.data.bname}.{self.config.data.shard_type}.{lang}"
            )
            # find text shards
            path_match = re.compile(f"{shard_name}[.][0-9]+[.][gx]z")

            shards = [
                os.path.join(self.config.data.data_shard_dir, seg)
                for seg in os.listdir(self.config.data.data_shard_dir)
                if re.match(path_match, seg)
            ]
        shards.sort(key=extract_shard_id)
        assert len(shards) > 0, f"no shards found for {lang}"
        return shards

    async def _process_lang(
        self,
        lang: str,
        index_type: str,
    ) -> tp.Tuple[tp.List[str], tp.List[str], str]:
        """
        prepare embeddings and indexes for a single language
        returns a tuple with the list of embedded files (the shards) and the final merged index for that lang
        """

        all_lang_configs = getattr(self.config, "lang_configs", {})
        lang_config = getattr(all_lang_configs, lang, {})

        embedded_files_glob = getattr(lang_config, "existing_embedding_glob", None)
        existing_index_path = getattr(lang_config, "existing_index_path", None)

        text_shards = self._find_data_shards(lang)

        if embedded_files_glob:
            # we already have precomputed the embedded files + merged index
            embedded_files = sorted(
                glob.glob(embedded_files_glob), key=extract_shard_id
            )
            assert len(embedded_files) > 0, f"couldn't find any embeddings for {lang}"
            logger.info(
                f"embeddings already provided for {lang}, found"
                f" {len(embedded_files)} embeddings"
            )
        else:
            logger.info(f"Number of shards for {lang}: {len(text_shards)}")
            self.config.embed_text.config.encoder.encoder_model = getattr(
                lang_config,
                "encoder_model",
                getattr(self.config.embed_text.config.encoder, "encoder_model", None),
            )
            self.config.embed_text.config.encoder.spm_model = getattr(
                lang_config,
                "spm_model",
                getattr(self.config.embed_text.config.encoder, "spm_model", None),
            )
            self.config.embed_text.config.encoder.spm_vocab = getattr(
                lang_config,
                "spm_vocab",
                getattr(self.config.embed_text.config.encoder, "spm_vocab", None),
            )
            embed_module = StopesModule.build(
                self.config.embed_text,
                lang=lang,
                shards=text_shards,
            )
            embedded_files = await self.launcher.schedule(embed_module)
            embedded_files = [str(f) for f in embedded_files]

        if existing_index_path:
            logger.info(f"index already provided for {lang}")

            return (text_shards, embedded_files, existing_index_path)

        else:
            sample_shards = getattr(
                self.config.train_index.config, "sample_shards", False
            )
            if sample_shards:
                logger.info(f"collecting index training sample for {lang}")
                sample_mod = SampleEmbeddingModule(
                    SampleEmbeddingModuleConfig(
                        embedded_files=embedded_files,
                        lang=lang,
                        data=self.config.data,
                        output_dir=self.config.train_index.config.output_dir,
                        embedding_dimensions=self.config.train_index.config.embedding_dimensions,
                        fp16=self.config.embed_text.config.encoder.fp16_storage,
                        sample_size=self.config.train_index.config.sample_sz,
                        tmp_dir=self.config.local_tmp_dir,
                        max_num_workers=self.config.embedding_sample.max_num_workers,
                    )
                )
                index_training_sample = await self.launcher.schedule(sample_mod)
            else:
                index_training_sample = embedded_files[0]
            train_index_module = StopesModule.build(
                self.config.train_index,
                index_type=index_type,
                data=self.config.data,
                embedding_file=index_training_sample,
                lang=lang,
                fp16_storage=self.config.embed_text.config.encoder.fp16_storage,
            )
            trained_index = await self.launcher.schedule(train_index_module)

            populate_index_module = StopesModule.build(
                self.config.populate_index,
                index=str(trained_index),
                embedding_files=embedded_files,
                lang=lang,
                index_type=index_type,
            )
            populated_indexes = await self.launcher.schedule(populate_index_module)

            populated_indexes = [
                str(idx) for idx in populated_indexes if idx is not None
            ]

            if len(populated_indexes) == 1:
                # there is only one index to start with, let's just use that instead of merging
                return (text_shards, embedded_files, populated_indexes[0])

            # otherwise, we need to run the merge
            merge_indexes_module = StopesModule.build(
                self.config.merge_indexes,
                data=self.config.data,
                indexes=sorted(populated_indexes, key=extract_shard_id),
                lang=lang,
                index_type=index_type,
            )
            merged = await self.launcher.schedule(merge_indexes_module)
            return (text_shards, embedded_files, str(merged))

    def run(self) -> None:
        loop = asyncio.get_event_loop()
        if self.config.launcher.cluster == "debug":
            loop.set_debug(True)
        loop.run_until_complete(self.arun())

    async def arun(self) -> None:
        logger.info(f"output: {os.path.abspath(self.config.output_dir)}")
        logger.info(f"working dir: {os.getcwd()}")

        (
            (src_text_shards, src_embeddings, src_merged_index),
            (
                tgt_text_shards,
                tgt_embeddings,
                tgt_merged_index,
            ),
        ) = await asyncio.gather(
            self._process_lang(
                lang=self.config.src_lang,
                index_type=self.src_index_type,
            ),
            self._process_lang(
                lang=self.config.tgt_lang,
                index_type=self.tgt_index_type,
            ),
        )

        src2tgt_calc_distances_module = StopesModule.build(
            self.config.calculate_distances,
            lang=self.config.src_lang,
            other_lang=self.config.tgt_lang,
            lang_embeddings=src_embeddings,
            distance_type=DistanceType.src2tgt,
            index_other_lang=tgt_merged_index,
        )
        tgt2src_calc_distances_module = StopesModule.build(
            self.config.calculate_distances,
            lang=self.config.tgt_lang,
            other_lang=self.config.src_lang,
            lang_embeddings=tgt_embeddings,
            distance_type=DistanceType.tgt2src,
            index_other_lang=src_merged_index,
        )
        src2tgt_dist, tgt2src_dist = await asyncio.gather(
            self.launcher.schedule(src2tgt_calc_distances_module),
            self.launcher.schedule(tgt2src_calc_distances_module),
        )

        # each schedule returns a list of dist+index files
        # we unzip that in two separate lists for each direction (four lists)
        (src2tgt_dist_files, src2tgt_index_files) = zip(*src2tgt_dist)
        (tgt2src_dist_files, tgt2src_index_files) = zip(*tgt2src_dist)
        src2tgt_dist_files = [str(path) for path in src2tgt_dist_files]
        tgt2src_dist_files = [str(path) for path in tgt2src_dist_files]

        mine_indexes_module = StopesModule.build(
            self.config.mine_indexes,
            index_type=self.src_index_type,
            src2tgt_dist_files=src2tgt_dist_files,
            src2tgt_index_files=src2tgt_index_files,
            tgt2src_dist_files=tgt2src_dist_files,
            tgt2src_index_files=tgt2src_index_files,
        )
        mined_indexes = await self.launcher.schedule(mine_indexes_module)

        src_meta_shards: tp.List[str] = []
        tgt_meta_shards: tp.List[str] = []
        if hasattr(self.config.data, "meta_glob"):
            glob_with_replacement = self.config.data.meta_glob.format(
                lang=self.config.src_lang
            )
            src_meta_shards = glob.glob(glob_with_replacement).sort(
                key=extract_shard_id
            )
            glob_with_replacement = self.config.data.meta_glob.format(
                lang=self.config.tgt_lang
            )
            tgt_meta_shards = glob.glob(glob_with_replacement).sort(
                key=extract_shard_id
            )

        mine_sentences_module = StopesModule.build(
            self.config.mine_sentences,
            src_text_files=src_text_shards,
            src_meta_files=src_meta_shards,
            tgt_text_files=tgt_text_shards,
            tgt_meta_files=tgt_meta_shards,
            alignment_file=mined_indexes,
        )
        mine_sentences = await self.launcher.schedule(mine_sentences_module)

        logger.info(f"Mining done, output is in {mine_sentences} .")


@hydra.main(config_path="conf", config_name="global_mining")
def main(config: GlobalMiningConfig) -> None:
    GlobalMiningPipeline(config).run()


if __name__ == "__main__":
    main()
