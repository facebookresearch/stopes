# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from stopes.eval.vocal_style_similarity.ecapa import Ecapa
from stopes.eval.vocal_style_similarity.valle_sv import ValleEncoder


def get_embedder(model_name: str, model_path: str, use_cuda=True):
    if model_name == "ecapa":
        model = Ecapa(model_path, use_cuda=use_cuda)
    elif model_name == "valle":
        model = ValleEncoder(model_path, use_cuda=use_cuda)  # type: ignore
    else:
        raise NotImplementedError
    return model


def compute_cosine_similarity(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    xs and ys arrays can be multi-dimensional; the similarity is computed along the last dimension.
    Their expected dimension is [batch_size, embedding_dim].
    """
    # xs/ys: [N, F]
    xnorm, ynorm = np.linalg.norm(xs, axis=-1), np.linalg.norm(ys, axis=-1)
    result = (
        np.matmul(np.expand_dims(xs, axis=-2), np.expand_dims(ys, axis=-1))
        .squeeze(axis=-1)
        .squeeze(axis=-1)
    )
    result = result / np.multiply(xnorm, ynorm)
    return result
