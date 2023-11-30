# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Different helper functions reused across multiple modules, focused on math and working with numpy.
"""


import numpy as np


def normalize_l2(emb: np.ndarray) -> np.ndarray:
    return emb / np.linalg.norm(emb, ord=2, axis=1, keepdims=True)


def pairwise_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity for each emb1 row with each emb2 row.
    Shapes:
    - emb1: [size1, emb_dim]
    - emb2: [size2, emb_dim]
    - output: [size1, size2]
    """
    return np.dot(normalize_l2(emb1), normalize_l2(emb2).T)


def rowwise_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity for each emb1 row with the corresponding emb2 row.
    Shapes:
    - emb1: [size, emb_dim]
    - emb2: [size, emb_dim]
    - output: [size]
    """
    return (normalize_l2(emb1) * normalize_l2(emb2)).sum(-1)
