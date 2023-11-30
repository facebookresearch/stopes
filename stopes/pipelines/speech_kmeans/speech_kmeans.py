# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

import hydra
from omegaconf import MISSING

from stopes.core import utils
from stopes.core.stopes_module import Requirements
from stopes.modules.speech.speech_kmeans import (
    SpeechKMeans,
    SpeechKMeansConfig,
    SpeechKMeansJob,
)

logger = logging.getLogger("speech_kmeans.speech_kmeans")


@dataclass
class SpeechKMeansMainConfig:
    launcher: tp.Any
    requirements: Requirements
    niter: int
    max_points_per_centroid: int = 10000000
    layers: tp.List[int] = field(default_factory=lambda: [35])
    feat_paths: tp.List[str] = MISSING
    km_sizes: tp.List[int] = MISSING
    out_dir: str = MISSING


async def speech_kmeans(config: SpeechKMeansMainConfig) -> None:
    logger.info(config)
    assert len(config.layers) == len(
        config.feat_paths
    ), "Layers and feature paths don't have 1:1 correspondence."
    launcher = hydra.utils.instantiate(config.launcher)
    speech_kmeans_jobs: tp.List[SpeechKMeansJob] = []
    for layer, feat_path in zip(config.layers, config.feat_paths):
        for km_size in config.km_sizes:
            km_dir = Path(config.out_dir) / f"layer_{layer}"
            utils.ensure_dir(km_dir)
            km_path = str(
                km_dir
                / f"km{km_size}_niter{config.niter}_mpc_{config.max_points_per_centroid}.npy"
            )
            speech_kmeans_jobs.append(
                SpeechKMeansJob(
                    layer=layer,
                    km_size=km_size,
                    km_path=km_path,
                    feat_path=feat_path,
                )
            )
    out_km_paths = await launcher.schedule(
        SpeechKMeans(
            SpeechKMeansConfig(
                niter=config.niter,
                max_points_per_centroid=config.max_points_per_centroid,
                requirements=config.requirements,
                speech_kmeans_jobs=speech_kmeans_jobs,
            )
        )
    )
    logger.info(out_km_paths)


@hydra.main(config_path="conf", config_name="speech_kmeans", version_base="1.1")
def main(config: SpeechKMeansMainConfig) -> None:
    asyncio.run(speech_kmeans(config))


if __name__ == "__main__":
    main()
