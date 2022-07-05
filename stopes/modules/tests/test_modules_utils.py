# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

test_length = 10_000
test_dim = 16
test_dtype = np.float16
test_lang = "bn"
test_idx_type = "OPQ8,IVF16,PQ8"


def generate_embedding(
    file=None, emb_length: int = test_length, dim: int = test_dim, dtype=test_dtype
):
    data = np.random.randn(emb_length, dim).astype(dtype)
    if file is not None:
        data.tofile(file)
    return data
