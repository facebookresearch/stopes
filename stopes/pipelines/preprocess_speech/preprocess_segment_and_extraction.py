# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import hydra
from omegaconf import MISSING

from stopes.core import Requirements, StopesModule, utils
from stopes.modules.audio_preprocess.audio_segment import AudioSegmentationConfig
from stopes.modules.audio_preprocess.erg_extraction import ErgExtractionConfig
from stopes.modules.audio_preprocess.f0_extraction import F0ExtractionConfig
from stopes.modules.audio_preprocess.hubert_extraction import HubertExtractionConfig
from stopes.pipelines.preprocess_speech.audio_pipelines import (
    audio_segment_pipeline,
    erg_extraction_pipeline,
    f0_extraction_pipeline,
    hubert_extraction_pipeline,
)

logger = logging.getLogger("preprocess_mined")

import io
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def zipfilepath_and_byteio(df, feats):
    for sample_id in tqdm(df.id):
        if sample_id not in feats.keys():
            logger.info(f"{sample_id} not found. Skip!")
            continue
        if feats[sample_id] is None:
            logger.info(f"No feat available for {sample_id}. Skip!")
            continue
        zipfile_name = f"{sample_id}.npy"
        byteio = io.BytesIO()
        np.save(byteio, feats[sample_id])
        byteio.seek(0)
        yield (zipfile_name, byteio)


@dataclass
class InputConfig:
    input_manifest: str = MISSING
    output_folder: str = MISSING
    src_lang: str = MISSING
    tgt_lang: str = MISSING
    nshards: int = MISSING
    min_duration_in_sec: float = 0.5


class InputModule(StopesModule):
    def __init__(
        self,
        config: InputConfig = InputConfig(),
        validate_config: bool = False,
    ):
        super().__init__(
            config,
            # TODO: always validate that config is a LineProcessorConfig
            # This is not possible currently because several config files add extra args
            # to make it easier to type the config
            config_class=InputConfig if validate_config else None,
        )
        # we do basic checkpointing with submitit Checkpointable which will store the state of this
        # callable. The basic idea here is to remember the last line processed
        # self.processed_lines = processed_lines
        Path(config.output_folder).mkdir(exist_ok=True)

    def requirements(self) -> Requirements:
        reqs = self.config.requirements
        if not isinstance(reqs, Requirements):
            # Performe conversion if needed
            return Requirements(**reqs)
        return reqs

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> tp.Any:
        df = pd.read_csv(self.config.input_manifest, sep="\t", quoting=3)
        result_paths = {}
        langs = [
            lang
            for lang in [self.config.src_lang, self.config.tgt_lang]
            if lang is not None
        ]
        for lang in langs:
            os.makedirs(self.config.output_folder + f"/{lang}", exist_ok=True)
            lang_df = df[
                [
                    f"{lang}_id",
                    f"{lang}_audio",
                    f"{lang}_audio_start",
                    f"{lang}_audio_duration",
                ]
            ].rename(
                columns={
                    f"{lang}_id": "id",
                    f"{lang}_audio": "audio",
                    f"{lang}_audio_start": "audio_start",
                    f"{lang}_audio_duration": "audio_duration",
                }
            )
            lang_df["src_ofst_and_len"] = lang_df.apply(
                lambda r: f"{r['audio_start']}:{r['audio_duration']}", axis=1
            )
            # There could be duplicated id due to matching to different utterance
            # But we only need to process the duplicated audio once
            logger.info(f"Before drop duplicates: {len(lang_df)} rows")
            lang_df = lang_df.drop_duplicates(subset="id")
            logger.info(
                f"Before filtering out short utterance (threadhold = {self.config.min_duration_in_sec}): {len(lang_df)} rows"
            )
            lang_df = lang_df[
                lang_df["audio_duration"] >= self.config.min_duration_in_sec
            ]
            output_path = f"{self.config.output_folder}/{lang}/audio_input.tsv"
            lang_df.to_csv(output_path, sep="\t", quoting=3, index=None)
            logger.info(f"Output audio input to {output_path} with {len(lang_df)} rows")
            result_paths[lang] = Path(output_path)
        return result_paths

    @staticmethod
    def version() -> str:
        return "0.6"


async def pipeline_for_one_language(
    config, input_manifest: Path, output_root_path: Path, nshards: int
):
    # setup a launcher to connect jobs together
    launcher = hydra.utils.instantiate(config.launcher)

    # setup local launcher for local job only
    with utils.clone_config(config.launcher) as local_launcher_config:
        local_launcher_config.cluster = "local"
    local_launcher = hydra.utils.instantiate(local_launcher_config)

    with utils.clone_config(config.audio_segmentation) as audio_seg_config:
        audio_seg_config.input_manifest = str(input_manifest)
        audio_seg_config.output_folder = str(output_root_path) + "/audios"
        audio_seg_config.nshards = nshards

    audio_seg_output_manifest = await audio_segment_pipeline(
        audio_seg_config,
        str(output_root_path / "audios.tsv"),
        launcher,
        local_launcher,
    )

    logger.info(f"Start extract audio files")

    with utils.clone_config(config.f0extraction) as f0_config:
        f0_config.input_manifest = str(audio_seg_output_manifest)
        f0_config.audio_column_name = audio_seg_config.audio_column_name
        f0_config.output_folder = str(output_root_path)
        f0_config.nshards = audio_seg_config.nshards // 3 + 1
    with utils.clone_config(config.ergextraction) as erg_config:
        erg_config.input_manifest = str(audio_seg_output_manifest)
        erg_config.audio_column_name = audio_seg_config.audio_column_name
        erg_config.output_folder = str(output_root_path)
        erg_config.nshards = audio_seg_config.nshards // 3 + 1
    with utils.clone_config(config.hubertextraction) as hubert_config:
        hubert_config.input_tsv_path = str(audio_seg_output_manifest)
        hubert_config.input_column_name = audio_seg_config.audio_column_name
        hubert_config.output_tsv_path = str(
            Path(output_root_path) / "unit" / "unit_orig.tsv"
        )
        hubert_config.nshards = audio_seg_config.nshards // 3 + 1

    (f0_extraction, erg_extraction, hubert_extraction) = await asyncio.gather(
        f0_extraction_pipeline(
            f0_config, str(output_root_path / "f0.tsv"), launcher, local_launcher
        ),
        erg_extraction_pipeline(
            erg_config, str(output_root_path / "erg.tsv"), launcher, local_launcher
        ),
        hubert_extraction_pipeline(
            hubert_config, str(output_root_path / "unit"), launcher, local_launcher
        ),
    )


#
#    (f0_extraction, erg_extraction, hubert_extraction) = await asyncio.gather(
#        launcher.schedule(F0ExtractionModule(
#                config=f0_config
#            )
#        ),
#        launcher.schedule(ErgExtractionModule(
#                config=erg_config
#            )
#        ),
#        launcher.schedule(HubertExtractionModule(
#                config=hubert_config
#            )
#        ),
#    )
#
#    # merge f0
#    logger.info(f"Compile feature f0...")
#    compile_feat(
#        audio_seg_output_manifest,
#        Path(f0_extraction[0]).parent,
#        "f0",
#        f0_config.nshards,
#    )
#
#    # merge erg
#    logger.info(f"Compile feature erg...")
#    compile_feat(
#        audio_seg_output_manifest,
#        Path(erg_extraction[0]).parent,
#        "erg",
#        erg_config.nshards,
#    )
#
#    # merge hubert and generate reduced unit
#    logger.info("Merging HuBERT units tsv")
#    hubert_tsv = merge_tsvs_if_sharded(
#        hubert_extraction,
#        Path(hubert_extraction[0]).parent / ("_".join(Path(hubert_extraction[0]).stem.split("_")[:-2]) + ".tsv"),
#    )
#    dedup_unit_tsv(
#        hubert_tsv,
#        (hubert_tsv.parent / ("_".join(hubert_tsv.stem.split("_")[:-3]) + "_dedup.tsv")),
#        unit_column="text",
#    )


@dataclass
class PreprocessMinedConfig:
    launcher: tp.Dict[str, tp.Any]
    input_config: InputConfig
    audio_segmentation: AudioSegmentationConfig
    f0extraction: F0ExtractionConfig
    ergextraction: ErgExtractionConfig
    hubertextraction: HubertExtractionConfig


async def pipeline(config):
    os.makedirs(config.audio_segmentation.output_folder, exist_ok=True)

    # setup a launcher to connect jobs together
    # launcher = hydra.utils.instantiate(config.launcher)

    # setup local launcher for local job only
    with utils.clone_config(config.launcher) as local_launcher_config:
        local_launcher_config.cluster = "local"
    local_launcher = hydra.utils.instantiate(local_launcher_config)

    input_paths = await local_launcher.schedule(InputModule(config=config.input_config))
    logger.info(input_paths)

    results = await asyncio.gather(
        *[
            pipeline_for_one_language(
                config,
                input_path,
                Path(config.input_config.output_folder + "/" + lang),
                config.input_config.nshards,
            )
            for lang, input_path in input_paths.items()
        ]
    )


# @hydra.main(config_path="conf", config_name="preprocess_mined")
# @hydra.main(config_path="conf", config_name="test")
@hydra.main(config_path="conf", config_name="monolingual_expressive_dataset")
def main(config: PreprocessMinedConfig) -> None:
    asyncio.run(pipeline(config))


if __name__ == "__main__":
    main()
