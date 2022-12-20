# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Unit test for mine() in stopes/modules/bitext/mining/mine_bitext_indexes_utils.py
# Usage: pytest test_mine_index_utils.py

import logging
import typing as tp
from pathlib import Path

import numpy as np

from stopes.modules.bitext.mining.mine_bitext_indexes_utils import mine

# import psutil

###################################
# begin parameters
###################################
k_src = 3
k_tgt = 3
threshold = 1.04
k_extract = 1
margin_type = "ratio"
mean_is_last = False
sort_neighbors = True
dists_x2y = np.array(
    [
        [0.2, 0.3, 0.6],
        [0.7, 0.2, 0.4],
        [0.1, 0.0, 1.0],
    ]
)
indices_x2y = np.array(
    [
        [0, 1, 2],
        [1, 2, 0],
        [0, 2, 1],
    ],
    dtype=np.uint32,
)
dists_y2x = np.array(
    [
        [0.8, 0.3, 0.1],
        [0.7, 0.2, 0.9],
        [0.1, 0.0, 1.0],
    ]
)
indices_y2x = np.array(
    [
        [1, 2, 0],
        [0, 2, 1],
        [0, 1, 2],
    ],
    dtype=np.uint32,
)
logger = logging.getLogger(__name__)
mine_expected_results = {
    "forward": {"src_idx": np.array([2, 1, 0]), "tgt_idx": np.array([2, 2, 1])},
    "backward": {"src_idx": np.array([1, 2, 0]), "tgt_idx": np.array([2, 1, 0])},
    "union": {
        "src_idx": np.array([1, 2, 2, 0, 1, 0]),
        "tgt_idx": np.array([2, 2, 1, 0, 2, 1]),
    },
}
k_extract_expected_results = {
    "forward": {
        2: {
            "src_idx": np.array([2, 2, 1, 0, 0]),
            "tgt_idx": np.array([2, 0, 2, 1, 0]),
        }
    }
}
## 'ratio' distances for x2y
# array([[1.11340206, 1.12087912, 0.85714286],
#        [0.87640449, 1.125     , 1.01052632],
#        [1.17525773, 1.2244898 , 0.65934066]])

## 'ratio' distances for y2x
# array([[0.75789474, 1.05154639, 1.17525773],
#        [0.85714286, 1.18681319, 0.74157303],
#        [1.16326531, 1.25      , 0.6122449 ]])
###################################
# end parameters
###################################


def _create_test_data(path_dir) -> tp.List[str]:
    out_dist_xy = str(path_dir / "dists_x2y")
    out_dist_yx = str(path_dir / "dists_y2x")
    out_idx_xy = str(path_dir / "indices_x2y")
    out_idx_yx = str(path_dir / "indices_y2x")

    np.save(out_dist_xy, dists_x2y)
    np.save(out_idx_xy, indices_x2y)
    np.save(out_dist_yx, dists_y2x)
    np.save(out_idx_yx, indices_y2x)

    return [
        f"{out_dist_xy}.npy",
        f"{out_dist_yx}.npy",
        f"{out_idx_xy}.npy",
        f"{out_idx_yx}.npy",
    ]


def test_mine_type(tmp_path: Path):
    dists_x2y, dists_y2x, indices_x2y, indices_y2x = _create_test_data(tmp_path)
    for mine_type in ["forward", "backward", "union"]:
        alignments = mine(
            [dists_x2y],
            [dists_y2x],
            [indices_x2y],
            [indices_y2x],
            k_src,
            k_tgt,
            k_extract,
            threshold,
            mean_is_last,
            margin_type,
            mine_type,
            sort_neighbors,
            logger,
        )
        print(alignments)
        assert (alignments.src_idx == mine_expected_results[mine_type]["src_idx"]).all()
        assert (alignments.tgt_idx == mine_expected_results[mine_type]["tgt_idx"]).all()


def test_k_extract(tmp_path: Path):
    dists_x2y, dists_y2x, indices_x2y, indices_y2x = _create_test_data(tmp_path)
    mine_type = "forward"
    k_extract = 2
    alignments = mine(
        [dists_x2y],
        [dists_y2x],
        [indices_x2y],
        [indices_y2x],
        k_src,
        k_tgt,
        k_extract,
        threshold,
        mean_is_last,
        margin_type,
        mine_type,
        sort_neighbors,
        logger,
    )
    print(alignments)
    assert (
        alignments.src_idx
        == k_extract_expected_results[mine_type][k_extract]["src_idx"]
    ).all()
    assert (
        alignments.tgt_idx
        == k_extract_expected_results[mine_type][k_extract]["tgt_idx"]
    ).all()
