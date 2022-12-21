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

from stopes.core import utils
from stopes.core.stopes_module import Requirements, StopesModule
from stopes.modules.preprocess.wav2vec_laser_speech_encoder import (
    LaserEmbeddingConfig,
    LaserFileAudioEncoder,
)
from stopes.pipelines.speech.utils import split_tsv_files

logger = logging.getLogger("speech.compute_laser_embeddings")


@dataclass
class ComputeEmbeddingConfig:
    checkpoint_file: str
    manifest_file: Path
    out_file: Path
    checkpoint_dir: Path
    max_tokens: int


class ComputeEmbedding(StopesModule):
    def __init__(
        self,
        config: ComputeEmbeddingConfig,
    ):
        super().__init__(config, ComputeEmbeddingConfig)
        self.config: ComputeEmbeddingConfig

    def requirements(self):
        return Requirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=1,
            cpus_per_task=10,
            timeout_min=24 * 60,
            constraint="volta32gb",
        )

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        logger = logging.getLogger("stopes.speech.compute_embedding")
        logger.info(
            f"Encoding {self.config.manifest_file.stem} with {self.config.checkpoint_dir / self.config.checkpoint_file}"
        )
        laser_embedding = LaserFileAudioEncoder(
            self.config.checkpoint_dir,
            self.config.checkpoint_file,
            self.config.max_tokens,
            logger,
        )
        laser_embedding.encode_file(self.config.manifest_file, self.config.out_file)
        return self.config.out_file


def get_checkpoint_file(lang: str) -> str:
    # TODO: Write clauses for all other languages.
    if lang == "en":
        checkpoint_file = "english.pt"
    elif lang in ("ru", "cs", "pl", "sk", "hr"):
        checkpoint_file = "slavic.pt"
    elif lang in ("ca", "es", "fr", "it", "pt", "ro"):
        checkpoint_file = "romance.pt"
    return checkpoint_file


async def compute_laser_embeddings(config: LaserEmbeddingConfig) -> None:
    logger.info(config)
    split_tsv_files(config.data_dir, config.lang_dirs, config.num_chunks)
    utils.ensure_dir(config.out_dir)
    compute_embedding_jobs = []
    launcher = hydra.utils.instantiate(config.launcher)
    for lang_dir in config.lang_dirs.split(","):
        src, tgt = lang_dir.split("-")
        for lang in (src, tgt):
            checkpoint_file = get_checkpoint_file(lang)
            for chunk_id in range(config.num_chunks):
                infile = Path(config.data_dir) / f"{lang_dir}_{lang}_{chunk_id}.tsv"
                outfile = (
                    Path(config.out_dir) / f"{lang_dir}_{lang}_{chunk_id}.embeddings"
                )
                compute_embedding_jobs.append(
                    launcher.schedule(
                        ComputeEmbedding(
                            ComputeEmbeddingConfig(
                                checkpoint_dir=config.checkpoint_dir,
                                max_tokens=config.max_tokens,
                                checkpoint_file=checkpoint_file,
                                manifest_file=infile,
                                out_file=outfile,
                            )
                        )
                    )
                )

    return await asyncio.gather(*compute_embedding_jobs)


@hydra.main(
    config_path="conf", config_name="compute_laser_embeddings", version_base="1.1"
)
def main(config: LaserEmbeddingConfig) -> None:
    asyncio.run(compute_laser_embeddings(config))


if __name__ == "__main__":
    main()
