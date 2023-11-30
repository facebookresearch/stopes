# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
This file inlcudes audio preprocess handy pipelines
Mostly to combine sharding and merging logics into one pipeline function
"""

import asyncio
import logging
import typing as tp
from pathlib import Path

from omegaconf import MISSING

from stopes.modules.audio_preprocess.audio_segment import (
    AudioSegmentationConfig,
    AudioSegmentationModule,
)
from stopes.modules.audio_preprocess.erg_extraction import (
    ErgExtractionConfig,
    ErgExtractionModule,
)
from stopes.modules.audio_preprocess.f0_extraction import (
    F0ExtractionConfig,
    F0ExtractionModule,
)
from stopes.modules.audio_preprocess.hubert_extraction import (
    HubertExtractionConfig,
    HubertExtractionModule,
    dedup_unit_tsv,
)

logger = logging.getLogger("audio_segment_pipeline")

import os

from omegaconf import MISSING

from stopes.modules.audio_preprocess.audio_segment import (
    AudioSegmentationConfig,
    AudioSegmentationModule,
)
from stopes.modules.audio_preprocess.utils import (
    CompileFeatConfig,
    CompileFeatModule,
    MergeTSVsConfig,
    MergeTSVsModule,
)


async def audio_segment_pipeline(
    audio_segment_config: AudioSegmentationConfig,
    output_merged_manifest: str,
    launcher,
    local_launcher,
):
    os.makedirs(audio_segment_config.output_folder, exist_ok=True)

    audio_segment_tsvs = await asyncio.gather(
        launcher.schedule(
            AudioSegmentationModule(
                config=audio_segment_config,
            )
        ),
    )
    logger.info(audio_segment_tsvs)
    # merge tsvs if they are sharded
    audio_seg_output_manifest = await local_launcher.schedule(
        MergeTSVsModule(
            config=MergeTSVsConfig(
                tsvs=audio_segment_tsvs[0],
                output_manifest=output_merged_manifest,
            )
        )
    )

    return audio_seg_output_manifest


async def f0_extraction_pipeline(
    config: F0ExtractionConfig,
    output_merged_manifest: str,
    launcher,
    local_launcher,
):
    f0_results = await launcher.schedule(
        F0ExtractionModule(
            config=config,
        )
    )

    compile_feat = await local_launcher.schedule(
        CompileFeatModule(
            config=CompileFeatConfig(
                input_manifest=config.input_manifest,
                output_manifest=output_merged_manifest,
                feat_folder=str(Path(f0_results[0]).parent),
                feat_column_name="f0",
                nshards=config.nshards,
            ),
        )
    )
    return compile_feat


async def erg_extraction_pipeline(
    config: ErgExtractionConfig,
    output_merged_manifest: str,
    launcher,
    local_launcher,
):
    erg_results = await launcher.schedule(
        ErgExtractionModule(
            config=config,
        )
    )

    compile_feat = await local_launcher.schedule(
        CompileFeatModule(
            config=CompileFeatConfig(
                input_manifest=config.input_manifest,
                output_manifest=output_merged_manifest,
                feat_folder=str(Path(erg_results[0]).parent),
                feat_column_name="erg",
                nshards=config.nshards,
            ),
        )
    )
    return compile_feat


async def hubert_extraction_pipeline(
    config: HubertExtractionConfig,
    output_merged_manifest_prefix: str,
    launcher,
    local_launcher,
):
    hubert_extraction = await launcher.schedule(
        HubertExtractionModule(
            config=config,
        )
    )

    # merge hubert and generate reduced unit
    logger.info("Merging HuBERT units tsv")
    hubert_tsv = await local_launcher.schedule(
        MergeTSVsModule(
            MergeTSVsConfig(
                tsvs=hubert_extraction,
                output_manifest=str(output_merged_manifest_prefix) + "_orig.tsv",
            )
        )
    )
    output_tsv = Path(str(output_merged_manifest_prefix) + "_dedup.tsv")
    dedup_unit_tsv(
        hubert_tsv,
        output_tsv,
        unit_column="text",
    )

    return output_tsv
