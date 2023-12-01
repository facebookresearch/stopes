# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
This module contains utils for contrastive and mixed training of prosody comparators.
"""

import random
import typing as tp
from dataclasses import dataclass

import faiss
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm, trange

from stopes.eval.blaser.utils import get_linear_schedule_with_warmup, shuffle_data


def get_neighbors(
    queries: np.ndarray,
    keys: np.ndarray,
    k: int = 50,
    nlist: int = 300,
    nprobe: int = 4,
    max_index_train_size: int = 100_000,
) -> tp.Set[tp.Tuple[int, int]]:
    """For each vector in the set of queries, extract k nearest neighbours (except itself) in the set of keys."""
    n, d = queries.shape
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)

    if max_index_train_size and max_index_train_size < n:
        index.train(keys[random.sample(range(n), max_index_train_size)])
    else:
        index.train(keys)
    index.nprobe = nprobe
    index.add(keys)

    pairs: tp.Set[tp.Tuple[int, int]] = set()
    for i in trange(n):
        distances, indices = index.search(queries[[i]], k)
        for j in indices[0]:
            pairs.update(
                {tuple(sorted([i, j])) for j in indices[0]}.difference({(i, i)})  # type: ignore
            )

    return pairs


def find_similar_texts_ids(
    texts1: tp.List[str],
    texts2: tp.List[str],
    pairs_per_item: int = 10,
    encoder_name: str = "sentence-transformers/LaBSE",
) -> tp.List[tp.Tuple[int, int]]:
    """
    From a parallel dataset of texts, find some unrelated (i.e. unaligned) pairs with high similarity,
    to use as hard negative examples. They are returned as a list of pairs of ints.
    The similarity is computed as cosine similarity of a SentenceTransformer (LaBSE by default).
    To keep the dataset balanced, we extract approximately the same number of neighbors per each item.
    """
    encoder = SentenceTransformer(encoder_name)
    encoder.cuda()
    with torch.inference_mode():
        embs1 = encoder.encode(texts1)
        embs2 = encoder.encode(texts2)

    # TODO: implement correction of similarities by adjusted texts length ratio
    pairs1 = get_neighbors(embs1, embs2, k=pairs_per_item)
    pairs2 = get_neighbors(embs2, embs1, k=pairs_per_item)
    pairs_all = pairs1.union(pairs2)
    return list(pairs_all)


def random_swap_(data1, data2):
    """Swap a random half of rows in two matrices, inplace."""
    n = data1.shape[0]
    idx = random.sample(range(n), n // 2)
    data1[idx], data2[idx] = data2[idx], data1[idx]


def get_contrastive_sample(
    emb_x,
    emb_y,
    hard_negative_pairs: tp.Optional[tp.List[tp.Tuple[int, int]]],
    batch_size: int = 64,
    swap: bool = True,
):
    """
    Sample batch_size / 2 positive pairs and recombine them as batch_size / 2 negative pairs.
    If the list of hard negative pairs is not provided, negatives are sampled randomly.
    """
    sample_size = batch_size // 4
    if hard_negative_pairs is None:
        # Sample negatives randomly
        ids = random.sample(range(0, emb_x.shape[0] - 1), sample_size * 2)
        idx1 = ids[:sample_size]
        idx2 = ids[sample_size:]
    else:
        # Sample negatives from the list of hard negatives
        pairs = random.sample(hard_negative_pairs, sample_size)
        idx1 = [id1 for id1, id2 in pairs]
        idx2 = [id2 for id1, id2 in pairs]
    # In any case, positives would be just the correct pairs for the negatives

    x1 = torch.cat(
        [
            emb_x[idx1],  # first-pair positives
            emb_x[idx2],  # second-pair positives
            emb_x[idx1],  # reuse as negatives
            emb_x[idx2],  # reuse as negatives
        ]
    )
    x2 = torch.cat(
        [
            emb_y[idx1],
            emb_y[idx2],
            emb_y[idx2],
            emb_y[idx1],
        ]
    )
    y = torch.cat([torch.ones(sample_size * 2), torch.zeros(sample_size * 2)])

    assert x1.shape[0] == x2.shape[0]
    assert x1.shape[0] == y.shape[0]

    # In some cases, we would like the comparator model to be approximately symmetric w.r.t. x1 and x2.
    # Thus, if there is a systematic difference, we try to offset it by swapping a random half of x1 and x2,
    # to prevent the model from learning this difference.
    if swap:
        random_swap_(x1, x2)
    return x1, x2, y


def ranking_loss(logits, labels, margin=0.5, decay=0.01, expected_mean=2.5):
    """
    Penalize the pairs of samples when logits and labels are different in opposite directions, with a hinge-like loss.
    Additionally, regularize the problem by pulling the predictions towards `expected_mean`.
    Expect logits of shape (batch_size, 1) and labels of shape (batch_size,).
    Hyperparameters:
        - margin: the minimal desired gap in predictions for items with different labels
        - decay: weight of the L2 loss that pulls the predictions towards expected_mean.
    """
    if len(logits.shape) == 1:
        logits = logits.unsqueeze(-1)
    b = logits.shape[0]
    assert len(logits.shape) == 2
    assert logits.shape[1] == 1

    if len(labels.shape) == 1:
        labels = labels.unsqueeze(-1)
    assert len(labels.shape) == 2

    loss = (
        # column prediction (plus margin) is larger than row pred
        torch.clamp(logits.repeat([1, b]).T - logits.repeat([1, b]) + margin, 0, None)
        # but row target is larger than column target
        * ((labels.repeat([1, b]) - labels.repeat([1, b]).T) > 0)
    ).mean()
    if decay > 0:
        loss = loss + ((logits - expected_mean) ** 2).mean() * decay
    return loss


@dataclass
class ComparatorTrainingParams:
    batch_size: int = 64
    paired_proportion: tp.Optional[float] = None
    negative_target: tp.Optional[float] = None
    positive_target: tp.Optional[float] = None
    quasi_regression_weight: float = 1.0
    epochs: int = 50
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    verbose: bool = True
    n_targets: int = 5
    rank_column_id: int = 0
    loss_margin: float = 0.5
    loss_decay: float = 0.0

    @property
    def has_quasi_regression(self) -> int:
        """Determine whether an artificial regression objective should be used for paired data"""
        return self.negative_target is not None and self.positive_target is not None


def get_n_batches(
    labelled_data,
    paired_data,
    cfg: ComparatorTrainingParams,
) -> tp.Tuple[int, int]:
    """Compute the number of batches per epoch for labelled and paired data"""
    size_labelled = labelled_data[0].shape[0] if labelled_data is not None else 0
    size_paired = paired_data[0].shape[0] if paired_data is not None else 0

    if cfg.paired_proportion == 1 or size_labelled == 0:
        n_labelled = 0
        n_paired = int(size_paired / cfg.batch_size)
    else:
        n_labelled = int(size_labelled / cfg.batch_size)
        if cfg.paired_proportion == 0 or size_paired == 0:
            n_paired = 0
        elif cfg.paired_proportion is None:
            n_paired = int(size_paired / cfg.batch_size)
        else:
            n_paired = int(
                n_labelled * cfg.paired_proportion / (1 - cfg.paired_proportion)
            )
    return n_labelled, n_paired


def shuffler(
    labelled_data,
    paired_data,
    cfg: ComparatorTrainingParams,
    negative_pairs: tp.Optional[tp.List[tp.Tuple[int, int]]] = None,
) -> tp.List[tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]]:
    """
    Create and mix batches of regression data and of contrastive data.
    Each batch is a tuple: (x_emb, y_emb, labels, is_supervised)
    """
    if cfg.paired_proportion == 1:
        batches = []
    else:
        batchified = shuffle_data(labelled_data, batch_size=cfg.batch_size)
        batchified.append([True] * len(batchified[0]))
        batches = list(zip(*batchified))

    _, n_paired_batches = get_n_batches(labelled_data, paired_data, cfg)

    for _ in range(n_paired_batches):
        x1, x2, y = get_contrastive_sample(
            paired_data[0],
            paired_data[1],
            hard_negative_pairs=negative_pairs,
            batch_size=cfg.batch_size,
        )
        if cfg.negative_target is not None and cfg.positive_target is not None:
            y = cfg.negative_target + y * (cfg.positive_target - cfg.negative_target)
        y = y.unsqueeze(1).repeat(1, cfg.n_targets)  # make 5 targets instead of 1
        batches.append((x1, x2, y, False))
    random.shuffle(batches)
    return batches


def combined_training(
    cfg: ComparatorTrainingParams,
    labelled_data: tp.Optional[tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    paired_data: tp.Optional[tp.Tuple[torch.Tensor, torch.Tensor]],
    negative_pairs: tp.Optional[tp.List[tp.Tuple[int, int]]],
    comparator: torch.nn.Module,
) -> tp.Tuple[torch.nn.Module, tp.Tuple[tp.List[float], tp.List[float]]]:
    """
    Train a model (typically, a Comparator) with either or both tasks:
    1) Minimize MSE on labelled_data=(features1, features2, target_comparator_scores)
    2) Minimize contrastive loss on paired_data=(features1, features2)
    For the contrastive part, negative pairs are sampled from the `negative_pairs` or just randomly.
    The `cfg` argument describes how the input data are combined and how losses are computed.
    """
    losses_reg: tp.List[float] = []
    losses_rank: tp.List[float] = []

    opt = torch.optim.AdamW(
        params=filter(lambda p: p.requires_grad, comparator.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    labelled_batches, paired_batches = get_n_batches(labelled_data, paired_data, cfg)
    total_steps = (labelled_batches + paired_batches) * cfg.epochs
    if cfg.verbose:
        print(f"Predicted {total_steps} total steps")
    opt_scheduler = get_linear_schedule_with_warmup(
        opt,
        cfg.warmup_steps,
        total_steps,
    )

    if cfg.verbose:
        fig, axes = plt.subplots(ncols=2)
        dh = ipd.display(fig, display_id=True)

    comparator.train(mode=True)
    for epoch in trange(cfg.epochs):
        train_batches = shuffler(
            labelled_data,
            paired_data,
            negative_pairs=negative_pairs,
            cfg=cfg,
        )
        for step, (src, mt, label, batch_is_supervised) in enumerate(train_batches):
            comparator.train(mode=True)
            if comparator.use_gpu:
                label = label.cuda()
                src = src.cuda()
                mt = mt.cuda()

            predictions = comparator(src=src, mt=mt, ref=mt)

            loss = ranking_loss(
                predictions[:, cfg.rank_column_id],
                label[:, cfg.rank_column_id],
                decay=cfg.loss_decay,
                margin=cfg.loss_margin,
            )
            total_loss = loss
            losses_rank.append(loss.item())

            loss = F.mse_loss(predictions, label)
            if cfg.has_quasi_regression or batch_is_supervised:
                total_loss = total_loss + loss * (
                    1 if batch_is_supervised else cfg.quasi_regression_weight
                )
            losses_reg.append(loss.item())

            total_loss.backward()
            opt.step()
            opt_scheduler.step()
            opt.zero_grad()

        if cfg.verbose:
            [a.clear() for a in axes]
            pd.Series(losses_rank).ewm(100).mean()[-5000:].plot(ax=axes[0])
            pd.Series(losses_reg).ewm(100).mean()[-5000:].plot(ax=axes[1])
            dh.update(fig)

    comparator.train(mode=False)

    return comparator, (losses_reg, losses_rank)
