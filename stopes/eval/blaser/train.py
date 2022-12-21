# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from stopes.core import utils as core_utils
from stopes.eval.blaser.model import BLASER
from stopes.eval.blaser.utils import (
    get_linear_schedule_with_warmup,
    get_pearson_corr,
    load_emb,
    shuffle_data,
)

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    src_emb_file: Path
    ref_emb_file: Path
    mt_emb_file: Path
    label_file: Path


@dataclass
class BlaserTrainConfig:
    train_files: tp.List[DataConfig]
    val_files: tp.List[DataConfig]
    output_dir: Path
    embedding_dimension: int = 1024
    output_dim: int = 1
    batch_size: int = 16
    norm_emb: bool = True
    input_form: str = "comet"
    learning_rate: float = 5e-5
    learning_rate_scheduler: bool = True
    output_act: bool = False
    train_epoch: int = 20
    nhid: tp.List = field(default_factory=lambda: [3072, 1536])
    dropout: float = 0.1
    activation: str = "TANH"
    use_gpu: bool = True


def run(config: BlaserTrainConfig) -> tp.Tuple[Path, Path]:
    output_dir = config.output_dir
    output_dir.mkdir(exist_ok=True)
    log_save_path = output_dir / f"blaser_train.log"
    logger.addHandler(logging.FileHandler(log_save_path))
    logger.info(f"log saving to {log_save_path.resolve()}")
    model = BLASER(
        idim=config.embedding_dimension,
        odim=config.output_dim,
        nhid=config.nhid,
        dropout=config.dropout,
        input_form=config.input_form,
        output_act=config.output_act,
        activation=config.activation,
        norm_emb=config.norm_emb,
        use_gpu=config.use_gpu,
    )
    if (output_dir / f"{model.filename}.pt").is_file():
        logger.info(f"model already exists at {output_dir.resolve()}")
    else:
        train_src, train_ref, train_mt, train_label = [], [], [], []
        for train_file in config.train_files:
            train_src.append(train_file.src_emb_file)
            train_ref.append(train_file.ref_emb_file)
            train_mt.append(train_file.mt_emb_file)
            train_label.append(train_file.label_file)
        train_src = load_emb(train_src)
        train_ref = load_emb(train_ref)
        train_mt = load_emb(train_mt)
        train_label = load_emb(train_label).float()

        val_src, val_ref, val_mt, val_label = [], [], [], []
        for val_file in config.val_files:
            val_src.append(val_file.src_emb_file)
            val_ref.append(val_file.ref_emb_file)
            val_mt.append(val_file.mt_emb_file)
            val_label.append(val_file.label_file)
        val_src = load_emb(val_src)
        val_ref = load_emb(val_ref)
        val_mt = load_emb(val_mt)
        val_label = load_emb(val_label).float()

        logger.info(f"training data size: {len(train_ref)}")
        logger.info(f"validation data size: {len(val_ref)}")

        train_list = [train_src, train_ref, train_mt, train_label]
        val_list = [val_src, val_ref, val_mt, val_label]

        train_iterations(
            model,
            config.train_epoch,
            config.batch_size,
            config.learning_rate,
            config.learning_rate_scheduler,
            train_list,
            val_list,
            config.use_gpu,
        )
        model.save(output_dir)

    return (output_dir / f"{model.filename}.pt").resolve(), (
        output_dir / f"{model.filename}.config"
    ).resolve()


def train_iterations(
    model: torch.nn.Module,
    train_epoch: int,
    batch_size: int,
    learning_rate: float,
    learning_rate_scheduler: bool,
    train_dset: tp.List[torch.tensor],
    val_dset: tp.List[torch.tensor],
    use_gpu: bool,
):
    opt = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
    )
    logger.info(f"classifier filename: {model.filename}")

    if learning_rate_scheduler:

        opt_scheduler = get_linear_schedule_with_warmup(
            opt, 0, 30000 * 100 // batch_size
        )

    best_corr = -1
    best_mlp = model.mlp
    for epoch in range(train_epoch):
        train_batch = shuffle_data(train_dset, batch_size)
        for step, (src, ref, mt, label) in enumerate(zip(*train_batch)):
            model.train(mode=True)
            if use_gpu:
                label = label.cuda()
            loss = F.mse_loss(model(src, ref, mt).squeeze(1), label)
            loss.backward()
            opt.step()
            if learning_rate_scheduler:
                opt_scheduler.step()
            opt.zero_grad()
            if (step + 1) % 50 == 0:
                logger.info(
                    f"epoch: {epoch}, step: {step}, loss: {loss.cpu().item():.4f}"
                )
            if val_dset is not None and (
                (step + 1) % 200 == 0 or step + 1 == len(train_dset[0]) // batch_size
            ):
                [val_src, val_ref, val_mt, val_label] = val_dset
                curr_corr = get_pearson_corr(
                    model,
                    src=val_src,
                    ref=val_ref,
                    mt=val_mt,
                    label=val_label,
                    use_gpu=use_gpu,
                )
                logger.info(
                    f"curr val corr: {curr_corr:.4f}. best val corr: {best_corr:.4f}"
                )
                if best_corr < curr_corr:
                    best_mlp = copy.deepcopy(model.mlp)
                    best_corr = curr_corr
                    logger.info(f"best val corr updated to {best_corr:.4f}")
    if val_dset is not None:
        model.mlp = best_mlp


@hydra.main(config_path="conf", config_name="train")
def main(config: DictConfig) -> None:
    typed_config = core_utils.promote_config(config, BlaserTrainConfig)
    run(typed_config)


if __name__ == "__main__":
    main()
