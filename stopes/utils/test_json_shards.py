# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, List

from stopes.utils.sharding.abstract_shards import BatchFormat
from stopes.utils.sharding.json_shards import JSONShard, JSONShardConfig


def _default_json_encoder(o: Any) -> Any:
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, Enum):
        return o.value
    if callable(o):
        return repr(o)
    return json.JSONEncoder().default(o)


def test_json_shard_config(tmp_path):
    ids = range(10)
    data = list("helloworld")

    json_data = [{"id": i, "char": d} for i, d in zip(ids, data)]
    test_file = tmp_path.joinpath("test.json")
    with open(test_file, encoding="utf-8", mode="w") as o:
        for item in json_data:
            o.write(json.dumps(item, default=_default_json_encoder) + "\n")

    input_config = JSONShardConfig(input_file=test_file, num_shards=3)

    shards: List[JSONShard] = input_config.make_shards()  # type: ignore

    # Test batch API
    text = ""
    for shard in shards:
        for batch in shard.to_batches(batch_size=3):
            text += "".join(batch["char"])
    assert text == "helloworld"

    # test __iter__ API . This API mains the context explicitly, just
    # like text_shards
    text = ""
    for shard in shards:
        with shard as shard_context:
            text += "".join([item["char"] for item in iter(shard_context)])
    assert text == "helloworld"

    # Test json sharding with nrows option
    input_config_nrows = JSONShardConfig(input_file=test_file, num_shards=2, nrows=5)
    small_shards = input_config_nrows.make_shards()
    text = ""
    for shard in small_shards:  # type: ignore
        for batch in shard.to_batches(batch_size=3):
            text += "".join(batch["char"])
    assert text == "hello"
