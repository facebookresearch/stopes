# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import typing as tp
from functools import lru_cache

import numpy as np
import ot  # pip install POT
import pandas as pd
from tqdm.auto import tqdm, trange


def norm64(items):
    results = []
    for item in items:
        item = item.astype(np.float64)
        item = item / item.sum()
        results.append(item)
    return results


def wass2unif(att_map, drop_eos=False, drop_bos=False):
    if len(att_map.shape) == 2:
        tgt_len, src_len = att_map.shape
        avg_vector = att_map.mean(0)
    else:
        (src_len,) = att_map.shape
        avg_vector = att_map
    if drop_eos:
        src_len -= 1
        avg_vector = avg_vector[:-1] / avg_vector[:-1].sum()
    if drop_bos:
        src_len -= 1
        avg_vector = avg_vector[1:] / avg_vector[1:].sum()
    return 0.5 * np.abs(avg_vector - np.ones(src_len) / src_len).sum()


@lru_cache(maxsize=1000)
def get_cost_matrix(m, n, smooth=False):
    if not smooth:
        a, b = np.arange(m), np.arange(n)
        res = np.abs(a[:, np.newaxis].repeat(n, 1) - b[np.newaxis, :].repeat(m, 0))
    else:
        mn = (m + n) / 2
        a, b = np.arange(m) / (m - 1) * mn, np.arange(n) / (n - 1) * mn
        res = np.abs(a[:, np.newaxis].repeat(n, 1) - b[np.newaxis, :].repeat(m, 0))
    return res.astype(np.float64)


def get_emd(vec1, vec2):
    m, n = len(vec1), len(vec2)
    if m < n:
        vec1 = np.concatenate([vec1, np.zeros(n - m)])
    if m > n:
        vec2 = np.concatenate([vec2, np.zeros(m - n)])
    m, n = len(vec1), len(vec2)
    cost_matrix = get_cost_matrix(m, n)
    return (ot.emd(vec1, vec2, cost_matrix) * cost_matrix).sum()


def get_best_dist(
    query: np.ndarray,
    len2means: tp.Dict[
        int, tp.List[np.ndarray]
    ],  # mapping from lengths to lists of reference attention maps
    k=4,
    r_max=1000,
    max_delta=0.1,
    verbose=True,
    return_mean=True,
    random_downsample=True,
    verbose_size=20,
    random_seed=None,
):
    if random_seed:
        random.seed(random_seed)
    m = len(query)
    cands = len2means.get(m, [])
    orig_len = len(cands)
    for i in range(1, 100):
        if len(cands) >= r_max:
            break
        if len(cands) > 0 and i > max_delta * m:
            break
        cands.extend(len2means.get(m - i, []))
        cands.extend(len2means.get(m + i, []))

    if len(cands) > r_max:
        if random_downsample:
            cands = random.sample(cands, r_max)
        else:
            cands = cands[:r_max]

    if orig_len < verbose_size and verbose:
        print(
            f"number of candidats is small for {m}: only {orig_len} => extended to {len(cands)}"
        )

    costs = np.array([get_emd(query, cand) for cand in cands])

    if len(costs) <= k:
        smallest_costs = costs
    else:
        smallest_costs = costs[np.argpartition(costs, k)[:k]]
    if return_mean:
        return smallest_costs.mean()
    return smallest_costs


def get_combo_score(
    wass_to_uniform,
    wass_to_data,
    w2u_threshold=0.8225450243337734,
    b_rescale=2.4869281337438,
    a_rescale=-1.2354565181293435,
):
    cutoff = np.array(wass_to_uniform) > w2u_threshold
    wass_combo = cutoff * (np.array(wass_to_uniform) * b_rescale + a_rescale) + (
        1 - cutoff
    ) * np.array(wass_to_data)
    return wass_combo


def compute_combo_parameters(
    filtered_attmaps_by_dir_and_len, threshold_quantile=0.999, rescale_quantile=0.99
):
    """
    This function explains how we computed the parameters of the get_combo_score function above.
    """
    dir2samples = {}
    dir2samplew2u = {}
    dir2samplew2d = {}
    for direction, len2maps in filtered_attmaps_by_dir_and_len.items():
        print(direction)
        sample = random.sample([v for part in len2maps.values() for v in part], 1000)
        dir2samples[direction] = sample
        dir2samplew2u[direction] = [wass2unif(am) for am in tqdm(sample)]
        dir2samplew2d[direction] = [
            get_best_dist(am, filtered_attmaps_by_dir_and_len[direction])
            for am in tqdm(sample)
        ]
    tmp_sampled = pd.DataFrame(
        {
            "w2u": [x for v in dir2samplew2u.values() for x in v],
            "w2d": [x for v in dir2samplew2d.values() for x in v],
        }
    )
    w2u_threshold = tmp_sampled["w2u"].quantile(threshold_quantile)
    print("threshold:", w2u_threshold)
    q0min, q0max = tmp_sampled["w2u"].quantile([1 - rescale_quantile, rescale_quantile])
    q1min, q1max = tmp_sampled["w2d"].quantile([1 - rescale_quantile, rescale_quantile])
    b_rescale = (q1max - q1min) / (q0max - q0min)
    a_rescale = q1min - q0min * b_rescale
    print("rescale parameters: ", b_rescale, a_rescale)
    return w2u_threshold, b_rescale, a_rescale
