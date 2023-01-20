# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import dataclasses
import logging
import typing as tp
from pathlib import Path

import hydra
from omegaconf import MISSING, DictConfig

from stopes.core import utils as core_utils
from stopes.core.launcher import Launcher
from stopes.eval.blaser.score import BlaserEvalConfig, PytorchModelConfig
from stopes.modules.evaluation.blaser_module import BlaserEvalModule
from stopes.pipelines.speech.compute_laser_embeddings import (
    ComputeEmbedding,
    ComputeEmbeddingConfig,
)

logger = logging.getLogger("stopes.eval_blaser")


@dataclasses.dataclass
class BlaserEvalPipelineConfig:
    output_dir: Path = MISSING
    launcher: tp.Any = MISSING

    src_manifest: Path = MISSING
    tgt_manifest: Path = MISSING
    # the reference to evaluate against
    ref_manifest: tp.Optional[Path] = None

    src_lang: str = MISSING
    tgt_lang: str = MISSING
    ref_lang: tp.Optional[str] = None

    # encoders
    checkpoint_dir: Path = MISSING
    # dict from lang to checkpoint file
    checkpoints: tp.Dict[str, str] = MISSING
    max_tokens: int = 2_560_000

    # blaser
    blaser_model: PytorchModelConfig = MISSING
    use_gpu: bool = True
    batch_size: int = 16


async def eval_blaser(
    launcher: Launcher, config: BlaserEvalPipelineConfig
) -> tp.Tuple[tp.Optional[float], Path, float]:
    """
    compute embeddings and return:
    - pearson score if requested
    - path of score per segment/sentence
    - average score
    """

    src_enc = getattr(config.checkpoints, config.src_lang, None)
    tgt_enc = getattr(config.checkpoints, config.tgt_lang, None)
    ref_enc = (
        getattr(config.checkpoints, config.ref_lang, None)
        if config.ref_lang is not None
        else None
    )

    assert src_enc and tgt_enc, "source and target encoder langs need a speech encoder."
    if config.ref_manifest:
        assert (
            ref_enc
        ), "if you want to use a reference, you need an encoder for that language."

    emb_out_dir = config.output_dir / "emb"
    emb_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. compute embeddings
    src_emb, tgt_emb, ref_emb = await core_utils.gather_optionals(
        launcher.schedule(
            ComputeEmbedding(
                ComputeEmbeddingConfig(
                    checkpoint_file=src_enc,
                    manifest_file=config.src_manifest,
                    out_file=emb_out_dir / f"source-{config.src_lang}-emb.npy",
                    checkpoint_dir=config.checkpoint_dir,
                    max_tokens=config.max_tokens,
                )
            )
        ),
        launcher.schedule(
            ComputeEmbedding(
                ComputeEmbeddingConfig(
                    checkpoint_file=tgt_enc,
                    manifest_file=config.tgt_manifest,
                    out_file=emb_out_dir / f"target-{config.tgt_lang}-emb.npy",
                    checkpoint_dir=config.checkpoint_dir,
                    max_tokens=config.max_tokens,
                )
            )
        ),
        launcher.schedule(
            ComputeEmbedding(
                ComputeEmbeddingConfig(
                    checkpoint_file=ref_enc,
                    manifest_file=config.ref_manifest,
                    out_file=emb_out_dir / f"reference-{config.ref_lang}-emb.npy",
                    checkpoint_dir=config.checkpoint_dir,
                    max_tokens=config.max_tokens,
                )
            )
        )
        if config.ref_manifest
        else None,
    )

    # 2. call blaser
    (correlation, score_file) = await launcher.schedule(
        BlaserEvalModule(
            BlaserEvalConfig(
                output_dir=config.output_dir / "results",
                model=config.blaser_model,
                src_emb_files=[src_emb],
                ref_emb_files=[ref_emb] if ref_emb else None,
                mt_emb_files=[tgt_emb],
                use_gpu=config.use_gpu,
                batch_size=config.batch_size,
            )
        )
    )

    with core_utils.open(score_file, "r") as scores:
        tot = 0.0
        cnt = 0
        for line in scores:
            tot += float(line.strip())
            cnt += 1

    avg = tot / cnt

    logger.info(f"scores can be found in {score_file}")
    logger.info(f"average score: {avg}")

    return correlation, score_file, avg


@hydra.main(config_path="conf", config_name="eval_blaser", version_base="1.1")
def main(config: DictConfig) -> tp.Tuple[tp.Optional[float], Path, float]:
    typed_config = core_utils.promote_config(config, BlaserEvalPipelineConfig)
    launcher = hydra.utils.instantiate(config.launcher)
    return asyncio.run(eval_blaser(launcher, typed_config))


if __name__ == "__main__":
    main()
