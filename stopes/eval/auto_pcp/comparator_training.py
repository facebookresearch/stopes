# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import typing as tp

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from tqdm.auto import trange

from stopes.eval.blaser.utils import get_linear_schedule_with_warmup, shuffle_data

from .audio_comparator import Comparator, get_model_pred


def train_comparator(
    blaser_cfg,
    train_dataset,
    val_dataset=None,
    train_epoch=50,
    batch_size=64,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=1000,
    verbose=True,
    loss_fn="mse",
    shuffler=None,
    steps_per_epoch=None,
    early_stopping=False,
):
    comparator = Comparator(**blaser_cfg)
    comparator.cuda()
    if steps_per_epoch is None:
        steps_per_epoch = int(np.ceil(train_dataset[0].shape[0] / batch_size))

    assert (
        early_stopping is False or val_dataset is not None
    ), "With early stopping, you need a validation dataset."

    total_steps = steps_per_epoch * train_epoch

    opt = torch.optim.AdamW(
        params=filter(lambda p: p.requires_grad, comparator.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    opt_scheduler = get_linear_schedule_with_warmup(
        opt,
        warmup_steps,
        total_steps,
    )

    if loss_fn == "mse":
        loss_fn = F.mse_loss
    if shuffler is None:
        shuffler = shuffle_data

    mean_losses, corrs = [], []
    losses = []

    # setting early stopping
    best_val_corr, best_model_params, best_epoch = -np.infty, {}, -1

    for epoch in trange(train_epoch):
        train_batch = shuffler(train_dataset, batch_size)
        n_steps = len(train_batch[0])
        for step, (src, mt, label) in enumerate(zip(*train_batch)):
            comparator.train(mode=True)
            if comparator.use_gpu:
                label = label.cuda()
            loss = loss_fn(comparator(src=src, mt=mt, ref=mt).squeeze(1), label)
            loss.backward()
            losses.append(loss.item())
            opt.step()
            opt_scheduler.step()
            opt.zero_grad()
        if val_dataset:
            [val_src, val_mt, val_label] = val_dataset
            curr_pred = get_model_pred(
                comparator,
                src=val_src,
                ref=val_mt,
                mt=val_mt,
                use_gpu=comparator.use_gpu,
            )
            if len(val_label.shape) == 1:
                val_label = val_label.unsqueeze(-1)
            curr_corr = spearmanr(
                curr_pred[:, 0].cpu().numpy(), val_label[:, 0].numpy()
            ).correlation
            if early_stopping and curr_corr > best_val_corr:
                best_val_corr, best_epoch = curr_corr, epoch
                best_model_params = comparator.state_dict()
        else:
            cur_corr = 0
        curr_loss = np.mean(losses[-n_steps:])
        if verbose:
            print(
                f"epoch: {epoch}, step: {step}, loss: {curr_loss:.4f}, corr: {curr_corr:.4f}"
            )
        mean_losses.append(curr_loss)
        corrs.append(curr_corr)
    if early_stopping:
        print(
            f"Best model was from epoch {best_epoch}, validation score {best_val_corr}"
        )
        comparator.load_state_dict(best_model_params)
    comparator.train(mode=False)
    return comparator, mean_losses, corrs


def split_data(x1, x2, y, f, dev_size=100, double_train_pairs=True):
    # split into train and val
    train_x1, dev_x1, train_x2, dev_x2, train_y, dev_y = train_test_split(
        x1[f], x2[f], y[f], test_size=dev_size, random_state=1
    )

    if double_train_pairs:
        # swap x1 and x2 to increase the dataset size and diversity
        train_x1, train_x2 = torch.cat([train_x1, train_x2]), torch.cat(
            [train_x2, train_x1]
        )
        train_y = torch.cat([train_y, train_y])
        dev_x1, dev_x2 = torch.cat([dev_x1, dev_x2]), torch.cat([dev_x2, dev_x1])
        dev_y = torch.cat([dev_y, dev_y])

    test_x1 = x1[~f]
    test_x2 = x2[~f]
    test_y = y[~f]

    return (
        (train_x1, train_x2, train_y),
        (dev_x1, dev_x2, dev_y),
        (test_x1, test_x2, test_y),
    )
