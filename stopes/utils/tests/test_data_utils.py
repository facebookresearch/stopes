# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import MutableMapping

from stopes.utils.data_utils import DictWithOverrides


def test_stopes_dict():
    d = DictWithOverrides(a=1, b=2)
    d.register_key_hooks(str.upper)
    d.register_value_hooks(lambda x: x**2)
    d["hello"] = 3

    assert d.type == dict
    assert isinstance(d, MutableMapping)

    # keys that are added before the add_key_func are not transformed
    assert "a" in d and "A" not in d
    assert "HELLO" in d and d["HELLO"] == 9
