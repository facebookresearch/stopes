# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from stopes.utils.sharding.hf_shards import HFInputConfig, HFShard

# TODO: Hard code this to test if there are changes in HF datasets API
first_item_id = 7


def test_shard_iteration():
    shard = HFShard(
        filter=None,
        path_or_name="Fraser/mnist-text-small",
        split="test",
        index=0,
        num_shards=50,
        trust_remote_code=True,
    )
    with shard:
        item = next(iter(shard))
        assert isinstance(item, dict)
        assert "label" in item
        assert item["label"] == first_item_id

    with shard as progress:
        batch_iter = progress.to_batches(batch_size=4)
        item = next(batch_iter)
        assert item["label"][0].as_py() == first_item_id  # type: ignore


def test_input_config():
    input_config = HFInputConfig(
        input_file="Fraser/mnist-text-small",
        split="test",
        num_shards=50,
        trust_remote_code=True,
    )
    shards = input_config.make_shards()
    first_shard = shards[0]
    with first_shard:
        item = next(iter(first_shard))
        assert item["label"] == first_item_id
