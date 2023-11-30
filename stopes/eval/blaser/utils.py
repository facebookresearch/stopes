# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random
import typing as tp
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr
from tqdm.auto import tqdm

from stopes.utils.embedding_utils import Embedding

logger = logging.getLogger(__name__)


def load_emb(
    files: tp.List[Path],
) -> torch.Tensor:
    emb_npy = None
    for f in files:
        logger.info(f"loading {f}")
        with Embedding(f).open_for_read() as data:
            shard: np.ndarray = data.astype(np.float32)
        if emb_npy is None:
            emb_npy = shard
        else:
            emb_npy = np.cat(emb_npy, shard)
    return torch.from_numpy(emb_npy)


def batchify(emb: torch.Tensor, batch_size: int) -> tp.List[torch.Tensor]:
    return [emb[s_idx : s_idx + batch_size] for s_idx in range(0, len(emb), batch_size)]


def shuffle_data(datalists: tp.List[torch.Tensor], batch_size: int):
    idxlist = list(range(len(datalists[0])))
    random.shuffle(idxlist)
    return [batchify(d[idxlist], batch_size) for d in datalists]


def get_model_pred(
    model, src, ref, mt, use_gpu: bool, batch_size: int = None, progress_bar=True
):
    if batch_size and batch_size < src.shape[0]:
        batches = [batchify(d, batch_size) for d in [src, ref, mt]]
        itr = zip(*batches)
        if progress_bar:
            itr = tqdm(itr)
    else:
        batches = [[src], [ref], [mt]]
        itr = zip(*batches)
    results = []
    with torch.no_grad():
        for src, ref, mt in itr:
            if isinstance(model, torch.nn.Module):
                model.train(mode=False)
                pred = model(src, ref, mt)
                model.train(mode=True)
            else:
                pred = model(src, ref, mt, use_gpu)
            results.append(pred)
    return torch.cat(results)


def get_pearson_corr(
    model, src, ref, mt, label, use_gpu: bool, model_pred=None
) -> float:
    if model_pred is None:
        model_pred = get_model_pred(model, src, ref, mt, use_gpu)
    if len(model_pred.shape) > 1:
        model_pred = model_pred.squeeze(-1)
    if len(label.shape) > 1:
        label = label.squeeze(-1)
    pearson_score = pearsonr(model_pred.tolist(), label.tolist())[0]
    return pearson_score


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    """Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    ref: https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py
    """
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)
