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
from stopes.modules.preprocess.laser_speech_embedding import (
    LaserEmbeddingConfig,
    LaserSpeechEmbedding,
)
from stopes.pipelines.speech.utils import split_tsv_files

logger = logging.getLogger("speech.compute_laser_embeddings")


@dataclass
class ComputeEmbeddingConfig:
    compute_embedding_jobs: tp.List[tp.Any]
    checkpoint_dir: Path
    max_tokens: int


class ComputeEmbedding(StopesModule):
    def __init__(
        self,
        config: ComputeEmbeddingConfig,
    ):
        super().__init__(config, ComputeEmbeddingConfig)

    def array(self):
        return self.config.compute_embedding_jobs

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
        assert iteration_value is not None, "iteration value is null"
        logger = logging.getLogger("stopes.speech.compute_embedding")
        checkpoint_file, infile, outfile = iteration_value
        logger.info(f"Encoding {infile.stem}")
        laser_embedding = LaserSpeechEmbedding(
            self.config.checkpoint_dir, checkpoint_file, self.config.max_tokens, logger
        )
        laser_embedding.encode_file(infile, outfile)


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
    for lang_dir in config.lang_dirs.split(","):
        src, tgt = lang_dir.split("-")
        for lang in (src, tgt):
            checkpoint_file = get_checkpoint_file(lang)
            for chunk_id in range(config.num_chunks):
                infile = Path(config.data_dir) / f"{lang_dir}_{lang}_{chunk_id}.tsv"
                outfile = (
                    Path(config.out_dir) / f"{lang_dir}_{lang}_{chunk_id}.embeddings"
                )
                compute_embedding_jobs.append((checkpoint_file, infile, outfile))

    launcher = hydra.utils.instantiate(config.launcher)
    await launcher.schedule(
        ComputeEmbedding(
            ComputeEmbeddingConfig(
                compute_embedding_jobs=compute_embedding_jobs,
                checkpoint_dir=config.checkpoint_dir,
                max_tokens=config.max_tokens,
            )
        )
    )


@hydra.main(config_path="conf", config_name="compute_laser_embeddings")
def main(config: LaserEmbeddingConfig) -> None:
    asyncio.run(compute_laser_embeddings(config))


if __name__ == "__main__":
    main()
