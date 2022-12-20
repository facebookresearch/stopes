# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from stopes.core import utils


def test_batch():
    items = list(range(10))
    listify = lambda items: [list(item) for item in items]

    assert listify(utils.batch([], 1)) == []
    assert listify(utils.batch([], 10)) == []
    # fmt: off
    assert listify(utils.batch(items, 1)) == [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
    # fmt: on
    assert listify(utils.batch(items, 2)) == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    assert listify(utils.batch(items, 3)) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert listify(utils.batch(items, 4)) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
