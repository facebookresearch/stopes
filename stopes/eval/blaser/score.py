# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from stopes.core import utils as core_utils
from stopes.eval.blaser.model import BLASER, unsupervised_blaser
from stopes.eval.blaser.utils import (
    batchify,
    get_model_pred,
    get_pearson_corr,
    load_emb,
)

logger = logging.getLogger(__name__)


@dataclass
class PytorchModelConfig:
    config_file: Path
    model_checkpoint: Path


@dataclass
class BlaserEvalConfig:
    output_dir: Path
    src_emb_files: tp.List[Path]
    mt_emb_files: tp.List[Path]
    ref_emb_files: tp.Optional[tp.List[Path]] = None
    label_files: tp.Optional[tp.List[Path]] = None
    model: tp.Optional[PytorchModelConfig] = None
    use_gpu: bool = True
    batch_size: int = 16


def run(config: BlaserEvalConfig) -> tp.Tuple[tp.Optional[float], Path]:
    output_dir = config.output_dir
    output_dir.mkdir(exist_ok=True)
    if config.model is not None and config.ref_emb_files is not None:
        model_config = torch.load(config.model.config_file)
        model = BLASER(
            idim=model_config["idim"],
            odim=model_config["odim"],
            nhid=model_config["nhid"],
            dropout=model_config["dropout"],
            input_form=model_config["input_form"],
            output_act=model_config["output_act"],
            activation=model_config["activation"],
            norm_emb=model_config["norm_emb"],
            use_gpu=config.use_gpu,
        )
        model.load_from_ckpt_file(config.model.model_checkpoint)
    else:
        logger.info("using unsupervised BLASER")
        model = unsupervised_blaser

    src_batch = batchify(
        load_emb(config.src_emb_files),
        config.batch_size,
    )
    ref_batch = (
        [None] * len(src_batch)
        if config.ref_emb_files is None
        else batchify(
            load_emb(config.ref_emb_files),
            config.batch_size,
        )
    )
    mt_batch = batchify(
        load_emb(config.mt_emb_files),
        config.batch_size,
    )

    logger.info(f"test data size: {sum([len(b) for b in src_batch])}")

    model_preds = []
    for src, ref, mt in zip(src_batch, ref_batch, mt_batch):
        model_pred = get_model_pred(model, src, ref, mt, config.use_gpu)
        model_preds.append(model_pred.squeeze(1).cpu())
    model_preds = torch.cat(model_preds)
    logger.info(f"output pred size: {len(model_preds)}")
    model_pred_file = output_dir / "blaser_scores.txt"
    cnt = 0
    with open(model_pred_file, "w") as fp:
        for x in model_preds:
            cnt += 1
            print(float(x), file=fp)
    logger.info(f"wrote {cnt} lines")
    logger.info(f"model predictions saved to {model_pred_file.resolve()}")
    pearson_corr = None
    if config.label_files is not None:
        test_label = load_emb(config.label_files)
        pearson_corr = get_pearson_corr(
            model=model,
            src=None,
            ref=None,
            mt=None,
            label=test_label,
            model_pred=model_preds,
            use_gpu=config.use_gpu,
        )
        logger.info(f"pearson corr: {pearson_corr:.4f}")
    return pearson_corr, model_pred_file


@hydra.main(config_path="conf", config_name="score")
def main(config: DictConfig) -> None:
    typed_config = core_utils.promote_config(config, BlaserEvalConfig)
    run(typed_config)


if __name__ == "__main__":
    main()
