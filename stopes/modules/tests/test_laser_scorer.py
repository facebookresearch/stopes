# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import numpy as np
import numpy.testing

from stopes.modules.bitext import laser_scorer


def test_corpus():
    x = laser_scorer.Corpus([], [])
    cache: tp.Dict[str, int] = {}

    original = ["a", "b", "a", "c", "b"]
    for sent in original:
        x.append(sent, cache)

    assert x.indices == [0, 1, 0, 2, 1]
    assert [x.sent(i) for i in range(len(original))] == original


def test_batch_knn():
    n, d, k = 1024, 32, 4
    x = np.random.uniform(-1, 1, (n, d)).astype(np.float32)
    y = np.random.uniform(-1, 1, (n, d)).astype(np.float32)
    full_knn = laser_scorer.faiss_knn(x, y, k)
    batch_size = 8
    batched_knn = laser_scorer.knn_batched(x, y, k, batch_size=batch_size, gpu=False)

    np.testing.assert_equal(full_knn.indices, batched_knn.indices)
    # Note: since we're creating different faiss indexes in the two algorithms,
    # the approximated faiss distances will be slightly different.
    np.testing.assert_allclose(full_knn.distances, batched_knn.distances, rtol=1e-6)
