# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from stopes.utils.sharding.hf_shards import HFInputConfig, HFShard

# TODO: Hard code this to test if there are changes in HF datasets API
expected_first_four = [
    1,
    0,
    1,
    0,
]  # contemmcm/rotten_tomatoes first 4 reviewState values


def test_shard_iteration():
    shard = HFShard(
        filter=None,
        path_or_name="contemmcm/rotten_tomatoes",
        split="complete",
        index=0,
        num_shards=50,
    )
    with shard:
        item = next(iter(shard))
        assert isinstance(item, dict)
        assert "reviewState" in item
        assert item["reviewState"] == expected_first_four[0]

    with shard as progress:
        batch_iter = progress.to_batches(batch_size=4)
        batch = next(batch_iter)
        # Verify first 4 items match expected pattern [1,0,1,0]
        for i in range(4):
            assert batch["reviewState"][i].as_py() == expected_first_four[i]  # type: ignore


def test_input_config():
    input_config = HFInputConfig(
        input_file="contemmcm/rotten_tomatoes",
        split="complete",
        num_shards=50,
    )
    shards = input_config.make_shards()
    first_shard = shards[0]
    with first_shard:
        item = next(iter(first_shard))
        assert item["reviewState"] == expected_first_four[0]
