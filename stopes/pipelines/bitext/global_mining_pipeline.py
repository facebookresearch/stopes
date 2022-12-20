# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import glob
import logging
import math
import os
import re
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from stopes.core import utils
from stopes.core.stopes_module import Requirements, StopesModule
from stopes.modules.bitext.indexing.sample_embedding_module import (
    SampleEmbeddingModule,
    SampleEmbeddingModuleConfig,
)
from stopes.modules.bitext.mining.calculate_distances import DistanceType
from stopes.modules.bitext.mining.merge_shards import MergeShardsModule  # noqa
from stopes.modules.preprocess.multiproc_bitext_processor import (
    MultiprocBitextProcessorConfig,
    MultiprocBitextProcessorModule,
)
from stopes.modules.preprocess.multiproc_line_processor import (
    MultiprocLineProcessorConfig,
    MultiprocLineProcessorModule,
)
from stopes.utils.data_utils import DataConfig
from stopes.utils.mining_utils import determine_faiss_index_type, extract_shard_id

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
    output_dir: Path
    launcher: DictConfig
    data: DataConfig
    model_dir: str
    count_lines: DictConfig
    split_in_shards: DictConfig
    embed_text: DictConfig
    embedding_sample: DictConfig
    train_index: DictConfig
    populate_index: DictConfig
    merge_indexes: DictConfig
    mine_indexes: DictConfig
    mine_sentences: DictConfig
    calculate_distances: DictConfig
    src_lang: str
    tgt_lang: str
    lang_configs: tp.Dict[str, LangConfig] = field(default_factory=dict)
    # if a language has more lines than `max_shard_size`, it will be split into sub-languages to mine separately.
    max_shard_size: tp.Optional[int] = None
    # example of sharded_langs: {"eng": ["eng0", "eng1", "eng2"]}
    sharded_langs: tp.Optional[tp.Dict[str, tp.List[str]]] = None
    local_tmp_dir: str = "/tmp"


@dataclass
class Lang:
    # This representation of a language is used within the pipeline.
    # It is inherited from the config, but may change if the language is split.
    lang_name: str  # original language name (e.g. eng)
    split_name: str  # language split name (can be different for big languages, e.g. eng001)
    data_shards: tp.List[str]
    meta_shards: tp.Optional[tp.List[str]]
    shard_sizes: tp.List[int]
    # these fields may be filled later if they are not precomputed
    embeddings: tp.List[str] = field(default_factory=list)
    merged_index: tp.Optional[str] = None
    index_type: tp.Optional[str] = None


class GlobalMiningPipeline:
    def __init__(
        self,
        config: GlobalMiningConfig,
    ):
        self.config = config
        self.launcher = hydra.utils.instantiate(config.launcher)
        OmegaConf.save(
            config=config,
            f=str(Path(self.launcher.config_dump_dir) / "global_mining.yaml"),
        )

    def _find_data_shards(
        self,
        lang: str,
    ) -> tp.List[str]:
        shard_list = getattr(self.config.data, "shard_list", None)
        if shard_list:
            return [Path(f) for f in shard_list]

        if hasattr(self.config.data, "shard_glob"):
            glob_with_replacement = self.config.data.shard_glob.format(lang=lang)
            shards = glob.glob(glob_with_replacement)
        else:
            shard_name = (
                f"{self.config.data.bname}.{self.config.data.shard_type}.{lang}"
            )
            # find shards
            path_match = re.compile(f"{shard_name}[.][0-9]+[.][gx]z")

            shards = [
                os.path.join(self.config.data.data_shard_dir, seg)
                for seg in os.listdir(self.config.data.data_shard_dir)
                if re.match(path_match, seg)
            ]
        shards.sort(key=extract_shard_id)
        assert len(shards) > 0, f"no shards found for {lang}"

        return shards

    def _read_nl_file(
        self, lang: str, data_cfg: DictConfig
    ) -> tp.Optional[tp.List[int]]:
        """
        Load the line counts for a language from a .nl file, if it exists.
        We expect one .nl file per language, with one number per shard within the file, each number on a separate line.
        """
        if not getattr(data_cfg, "nl_file_template", None):
            return None
        nl_file = Path(data_cfg.data_shard_dir) / data_cfg.nl_file_template.format(
            lang=lang
        )
        if not nl_file.is_file():
            logger.info(f"no nl file for {lang} in {nl_file}")
            return None
        result = []
        with utils.open(nl_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                result.append(int(line))
        return result

    async def _segment_data_shards(self, shards, segment_audio_cfg):
        segment_audio_module = StopesModule.build(
            segment_audio_cfg,
            shards=shards,
        )
        segmented_audio = await self.launcher.schedule(segment_audio_module)

        return segmented_audio

    async def _process_lang(
        self,
        lang: str,
        check_size: bool = True,
    ) -> tp.List[Lang]:
        """
        Prepare embeddings and indexes for a single language
        (which might be split into multiple sub-languages if it is too large).
        Return a list of `Lang` objects that contain embedded files and final merged indexes for each sub-language.
        """
        # process the lang separately if it is pre-sharded
        sharded_langs = getattr(self.config, "sharded_langs", None) or {}
        sharded_lang_config = sharded_langs.get(lang)
        if sharded_lang_config is not None:
            precomputed_shards = sharded_lang_config
            logger.info(
                f"Processing precomputed splits for language {lang}: {precomputed_shards}"
            )
            processed_shards = await asyncio.gather(
                *[
                    self._process_lang(lang_shard, check_size=False)
                    for lang_shard in precomputed_shards
                ]
            )  # do not split the shards for the second time
            # flatten the list of lists
            results = [
                shard for inner_shards in processed_shards for shard in inner_shards
            ]
            # assign a common language name to all shards (it will be used for merging)
            for shard in results:
                shard.lang_name = lang
            return results
        data_shards = self._find_data_shards(lang)
        is_speech = getattr(self.config, "segment_audio", None)
        if is_speech:
            data_segment_shards = await self._segment_data_shards(
                data_shards, self.config.segment_audio
            )
            data_shards, shard_sizes = zip(*data_segment_shards)
        else:
            shard_sizes = self._read_nl_file(lang, self.config.data)
            if shard_sizes is not None and len(shard_sizes) == len(data_shards):
                logger.info(f"Using cached shard sizes from a .nl file for {lang}.")
            else:
                line_counter = StopesModule.build(
                    self.config.count_lines, shards=data_shards
                )
                shard_sizes = await self.launcher.schedule(line_counter)
        logger.debug(f"Shards are: {list(zip(data_shards, shard_sizes))}")
        nb_sent = sum(shard_sizes)

        meta_shards = None
        if hasattr(self.config.data, "meta_glob"):
            glob_with_replacement = self.config.data.meta_glob.format(lang=lang)
            meta_shards = sorted(glob.glob(glob_with_replacement), key=extract_shard_id)
        logger.info(f"for {lang}, meta shards are {meta_shards}")

        lng = Lang(
            lang_name=lang,
            split_name=lang,
            data_shards=data_shards,
            meta_shards=meta_shards,
            shard_sizes=shard_sizes,
        )

        if (
            check_size
            and self.config.max_shard_size
            and nb_sent > self.config.max_shard_size
        ):
            logger.info(
                f"Splitting the lang {lang}, because {nb_sent} > {self.config.max_shard_size}"
            )
            parts = await self._split_lang(
                lng=lng,
                max_shard_size=self.config.max_shard_size,
            )
        else:
            logger.info(
                f"Not splitting the lang {lang}: processing {nb_sent} as a whole"
            )
            parts = [lng]

        return await asyncio.gather(
            *[self._process_language_shard(part, is_speech) for part in parts]
        )

    async def _process_language_shard(
        self,
        lng: Lang,  # orig_lang, lang, data_shards, nb_sent, meta_shards,
        is_speech: bool,
    ) -> Lang:
        """Compute and index embeddings for a language and add them to the Lang structure.
        Parameter `orig_lang` is the name of the real language (e.g. "en"),
        while `lang` is the name of the language shard (e.g. "en001")
        """
        result = lng

        all_lang_configs = getattr(self.config, "lang_configs", {})
        lang_config = getattr(all_lang_configs, lng.lang_name, {})
        embedded_files_glob = getattr(lang_config, "existing_embedding_glob", None)
        result.merged_index = getattr(lang_config, "existing_index_path", None)

        result.index_type = getattr(
            lang_config, "index_type", None
        ) or determine_faiss_index_type(nb_sent=sum(lng.shard_sizes))

        if embedded_files_glob:
            # we already have precomputed the embedded files + merged index
            embedded_files = sorted(
                glob.glob(embedded_files_glob), key=extract_shard_id
            )
            assert (
                len(embedded_files) > 0
            ), f"couldn't find any embeddings for {lng.lang_name} with {embedded_files_glob}"
            logger.info(
                f"embeddings already provided for {lng.lang_name}, found"
                f" {len(embedded_files)} embeddings"
            )
        else:
            logger.info(f"Number of shards for {lng.lang_name}: {len(lng.data_shards)}")

            # TODO: refactor speech vs text encoders
            if not is_speech:
                self.config.embed_text.config.encoder.encoder_model = getattr(
                    lang_config,
                    "encoder_model",
                    getattr(
                        self.config.embed_text.config.encoder, "encoder_model", None
                    ),
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
                lang=lng.lang_name,
                lang_shard_name=lng.split_name,
                shards=lng.data_shards,
                # TODO: validate it with something like expected_shard_sizes=shard_sizes,
            )
            embedded_files = await self.launcher.schedule(embed_module)
            embedded_files = [str(f) for f in embedded_files]
        result.embeddings = embedded_files

        if lng.merged_index:
            assert os.path.exists(
                lng.merged_index
            ), f"existing index for {lng.lang_name} is missing: {lng.merged_index}"
            logger.info(f"index already provided for {lng.lang_name}")
            return result

        else:
            sample_shards = getattr(
                self.config.embedding_sample, "sample_shards", False
            )
            if sample_shards:
                logger.info(f"collecting index training sample for {lng.lang_name}")
                sample_mod = SampleEmbeddingModule(
                    SampleEmbeddingModuleConfig(
                        embedded_files=embedded_files,
                        lang=lng.lang_name,
                        data=self.config.data,
                        output_dir=self.config.train_index.config.output_dir,
                        embedding_dimensions=self.config.train_index.config.embedding_dimensions,
                        fp16=self.config.embed_text.config.encoder.fp16,
                        sample_size=self.config.embedding_sample.sample_sz,
                        tmp_dir=self.config.local_tmp_dir,
                        max_num_workers=self.config.embedding_sample.max_num_workers,
                    )
                )
                index_training_sample = await self.launcher.schedule(sample_mod)
                index_training_sample = str(index_training_sample)
            else:
                index_training_sample = embedded_files[0]
            train_index_module = StopesModule.build(
                self.config.train_index,
                index_type=lng.index_type,
                data=self.config.data,
                embedding_file=index_training_sample,
                lang=lng.split_name,
                fp16=self.config.embed_text.config.encoder.fp16,
            )
            trained_index = await self.launcher.schedule(train_index_module)

            populate_index_module = StopesModule.build(
                self.config.populate_index,
                index=str(trained_index),
                embedding_files=embedded_files,
                lang=lng.split_name,
                index_type=lng.index_type,
                data=self.config.data,
            )
            populated_indexes = await self.launcher.schedule(populate_index_module)

            populated_indexes = [
                str(idx) for idx in populated_indexes if idx is not None
            ]

            if len(populated_indexes) == 1:
                # there is only one index to start with, let's just use that instead of merging
                result.merged_index = populated_indexes[0]
                return result

            # otherwise, we need to run the merge
            merge_indexes_module = StopesModule.build(
                self.config.merge_indexes,
                data=self.config.data,
                indexes=sorted(populated_indexes, key=extract_shard_id),
                lang=lng.split_name,
                index_type=lng.index_type,
                expected_line_count=sum(lng.shard_sizes),
            )
            merged = await self.launcher.schedule(merge_indexes_module)
            result.merged_index = str(merged)
            return result

    def run(self) -> tp.Tuple[str, str]:
        loop = asyncio.get_event_loop()
        if self.config.launcher.cluster == "debug":
            loop.set_debug(True)
        return loop.run_until_complete(self.arun())

    async def arun(self) -> tp.Tuple[str, str]:
        """Run the global mining pipeline and return the paths of the mined text and metadata files"""
        logger.info(f"output: {os.path.abspath(self.config.output_dir)}")
        logger.info(f"working dir: {os.getcwd()}")

        (src_list, trg_list) = await asyncio.gather(
            self._process_lang(
                lang=self.config.src_lang,
            ),
            self._process_lang(
                lang=self.config.tgt_lang,
            ),
        )
        mined_pairs = await asyncio.gather(
            *[self._process_pair(l1, l2) for l1 in src_list for l2 in trg_list]
        )
        logger.info(
            f"After mining, need to merge {len(mined_pairs)} language pairs: {mined_pairs}."
        )
        if len(mined_pairs) == 1:
            mine_sentences, mine_meta = mined_pairs[0]
        else:
            mine_sentences, mine_meta = await self._merge_pairs(mined_pairs)
        logger.info(f"Mining done, output is in {mine_sentences}")
        logger.info(f"Mined metadata is in {mine_meta}")
        return mine_sentences, mine_meta

    async def _process_pair(self, src: Lang, tgt: Lang) -> tp.Tuple[Path, Path]:
        src2tgt_calc_distances_module = StopesModule.build(
            self.config.calculate_distances,
            lang=src.split_name,
            other_lang=tgt.split_name,
            lang_embeddings=src.embeddings,
            distance_type=DistanceType.src2tgt,
            index_other_lang=tgt.merged_index,
            index_other_lang_type=tgt.index_type,
        )
        tgt2src_calc_distances_module = StopesModule.build(
            self.config.calculate_distances,
            lang=tgt.split_name,
            other_lang=src.split_name,
            lang_embeddings=tgt.embeddings,
            distance_type=DistanceType.tgt2src,
            index_other_lang=src.merged_index,
            index_other_lang_type=src.index_type,
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
        src2tgt_index_files = [str(path) for path in src2tgt_index_files]
        tgt2src_index_files = [str(path) for path in tgt2src_index_files]

        mine_indexes_module = StopesModule.build(
            self.config.mine_indexes,
            index_type=src.index_type,
            src_lang=src.split_name,
            tgt_lang=tgt.split_name,
            src2tgt_dist_files=src2tgt_dist_files,
            src2tgt_index_files=src2tgt_index_files,
            tgt2src_dist_files=tgt2src_dist_files,
            tgt2src_index_files=tgt2src_index_files,
        )
        mined_indexes = await self.launcher.schedule(mine_indexes_module)

        src_meta = src.meta_shards if src.meta_shards is not None else []
        tgt_meta = tgt.meta_shards if tgt.meta_shards is not None else []
        mine_sentences_module = StopesModule.build(
            self.config.mine_sentences,
            src_lang=src.split_name,
            tgt_lang=tgt.split_name,
            src_text_files=src.data_shards,
            src_meta_files=src_meta,
            tgt_text_files=tgt.data_shards,
            tgt_meta_files=tgt_meta,
            alignment_file=mined_indexes,
        )
        mine_sentences, mine_meta = await self.launcher.schedule(mine_sentences_module)
        return mine_sentences, mine_meta

    async def _split_lang(self, lng: Lang, max_shard_size: int) -> tp.List[Lang]:
        """Split the sentences and the corresponding metadata"""
        nb_shards = int(math.ceil(sum(lng.shard_sizes) / max_shard_size))
        logger.info(f"Preparing split of {lng.shard_sizes} lines in {nb_shards} shards")
        cfg = self.config.split_in_shards.config
        outfile_prefix = f"shards_{lng.lang_name}"
        req_args = cfg.get("requirements") or {}
        reqs = Requirements(**req_args)
        if (
            lng.meta_shards
        ):  # TODO: merge these two branches of code after the line processor is unified.
            bitext = True
            logger.info(
                f"splitting {lng.lang_name} as bitext, becase there are metadata: {lng.meta_shards}"
            )
            file_processor = MultiprocBitextProcessorModule(
                config=MultiprocBitextProcessorConfig(
                    bitext_processor=DictConfig(
                        {
                            "_target_": "stopes.modules.preprocess.split_in_shards.SplitInShardsParallelMC",
                            "nb_shards": nb_shards,
                        }
                    ),
                    custom_name="split_in_shards",
                    output_dir=cfg.output_dir,
                    outfile_prefix=outfile_prefix,
                    shards=[list(p) for p in zip(lng.data_shards, lng.meta_shards)],
                    requirements=reqs,
                    tmp_dir=self.config.local_tmp_dir,
                )
            )
        else:
            bitext = False
            logger.info(
                f"splitting as monotext, becase there is no meta files: {lng.meta_shards}"
            )
            file_processor = MultiprocLineProcessorModule(
                config=MultiprocLineProcessorConfig(
                    line_processor=DictConfig(
                        {
                            "_target_": "stopes.modules.preprocess.split_in_shards.SplitInShardsMC",
                            "nb_shards": nb_shards,
                        }
                    ),
                    custom_name="split_in_shards",
                    output_dir=cfg.output_dir,
                    outfile_prefix=outfile_prefix,
                    shards=lng.data_shards,
                    requirements=reqs,
                    tmp_dir=self.config.local_tmp_dir,
                )
            )

        logger.info(f"Sharding and shuffling files {lng.data_shards}")
        results = await self.launcher.schedule(file_processor)
        logger.info("Done sharding and shuffling to shards")
        # reshape [old_shards, data_type, new_shards] into [data_type, new_shards, old_shards]
        results = [
            [list(old_shards) for old_shards in zip(*data_type)]
            for data_type in zip(*results)
        ]
        logger.info(f"results transposed are {results}")
        if bitext:
            out_shards, out_meta, out_counts = results
        else:
            out_shards, out_counts = results
            out_meta = [None for _ in out_shards]

        return [
            Lang(
                lang_name=lng.lang_name,
                split_name=f"{lng.lang_name}_{i:03d}",
                data_shards=[str(text) for text in texts],
                meta_shards=[str(meta) for meta in metas]
                if metas is not None
                else None,
                shard_sizes=sizes,
            )
            for i, (texts, sizes, metas) in enumerate(
                zip(out_shards, out_counts, out_meta)
            )
        ]

    async def _merge_pairs(
        self, mined_pairs: tp.List[tp.Tuple[Path, Path]]
    ) -> tp.Tuple[Path, Path]:
        """Merge the pairs of sentences and the corresponding metadata"""
        merge_module = StopesModule.build(self.config.merge_shards, pairs=mined_pairs)
        merged_text, merged_meta = await self.launcher.schedule(merge_module)
        return merged_text, merged_meta


@hydra.main(config_path="conf", config_name="global_mining", version_base=None)
def main(config: GlobalMiningConfig) -> None:
    return GlobalMiningPipeline(config).run()


if __name__ == "__main__":
    main()
