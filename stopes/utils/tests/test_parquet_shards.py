# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import shutil
import tempfile

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from conftest import get_random_table, permutationally_equal_dataframes

from stopes.utils.sharding.abstract_shards import BatchFormat
from stopes.utils.sharding.parquet_shards import (
    ParquetOutputConfig,
    ParquetShard,
    ParquetShardingConfig,
)


def test_parquet_to_batches(single_file_dataset):
    input_config = ParquetShardingConfig(
        input_file=single_file_dataset,
        fragment_group_size=1,
        batch_size=11,
        batch_format=BatchFormat.ARROW,
    )
    shard = input_config.make_shards()[0]
    with shard:
        bb = next(shard.to_batches(4, batch_format=BatchFormat.ARROW))

    bb_head = input_config.head(4)
    assert isinstance(bb, pa.Table)
    assert bb.equals(bb_head)

    assert len(bb.schema.names) == 3
    assert len(bb) == 4


def test_parquet_sharding_config_single_file(single_file_dataset):
    psc = ParquetShardingConfig(
        input_file=str(single_file_dataset),
        batch_size=11,
        columns=["score", "cat", "name"],
        batch_format=BatchFormat.ARROW,
        filters_expr="(ds.field('score') < 0) & (ds.field('cat') >= 5)",
    )

    assert psc.partition_columns == []
    shards = psc.make_shards()
    assert len(shards) == 1
    # all columns are present
    assert set(tuple(shard.columns) for shard in shards) == set(  # type: ignore
        [("cat", "name", "score")]
    )
    assert set(tuple(shard.partition_columns) for shard in shards) == set([()])  # type: ignore

    first_shard = shards[0]

    # working with full batch
    batches = list(
        first_shard.to_batches(
            batch_size=None, columns=psc.columns, batch_format=psc.batch_format
        )
    )
    assert len(batches) == 1
    assert [len(bb) for bb in batches] == [245]
    assert [list(bb.column_names) for bb in batches] == [["cat", "name", "score"]]  # type: ignore

    one_batch = batches[0]
    assert one_batch["score"].to_numpy().max() < 0  # type: ignore
    assert one_batch["cat"].to_numpy().min() >= 5  # type: ignore

    batches = list(
        first_shard.to_batches(
            batch_size=psc.batch_size,
            columns=psc.columns,
            batch_format=psc.batch_format,
        )
    )
    assert len(batches) == 23
    assert [len(bb) for bb in batches] == [11] * 22 + [3]
    assert [list(bb.column_names) for bb in batches] == 23 * [["cat", "name", "score"]]  # type: ignore

    one_batch = batches[3]
    assert one_batch["score"].to_numpy().max() < 0  # type: ignore
    assert one_batch["cat"].to_numpy().min() >= 5  # type: ignore


def test_parquet_sharding_config(multi_partition_file_dataset):
    psc = ParquetShardingConfig(
        input_file=str(multi_partition_file_dataset),
        batch_size=10,
        columns=["score", "cat"],
        batch_format=BatchFormat.PANDAS,
        filters_expr="(ds.field('score') < 0) & (ds.field('cat') >= 5)",
    )

    assert psc.partition_columns == ["cat"]
    shards = psc.make_shards()
    assert len(shards) == 5
    # all columns are present
    assert set(tuple(shard.columns) for shard in shards) == set(  # type: ignore
        [("name", "score", "cat")]
    )
    assert set(tuple(shard.partition_columns) for shard in shards) == set([("cat",)])  # type: ignore
    first_shard = shards[0]
    batches = list(
        first_shard.to_batches(
            batch_size=psc.batch_size,
            columns=psc.columns,
            batch_format=psc.batch_format,
        )
    )
    assert len(batches) == 5
    assert [len(bb) for bb in batches] == [10, 10, 10, 10, 8]
    assert [list(bb.columns) for bb in batches] == 5 * [psc.columns]  # type: ignore

    one_batch = batches[3]
    assert one_batch["score"].max() < 0
    assert one_batch["cat"].astype("int").min() >= 5

    # working with full batch
    batches = list(
        first_shard.to_batches(
            batch_size=None, columns=psc.columns, batch_format=psc.batch_format
        )
    )
    assert len(batches) == 1
    assert [len(bb) for bb in batches] == [48]
    assert [list(bb.columns) for bb in batches] == [psc.columns]  # type: ignore

    one_batch = batches[0]
    assert one_batch["score"].max() < 0
    assert one_batch["cat"].astype("int").min() >= 5


def test_parquet_output_config():
    tmpdir = tempfile.mkdtemp()

    poc = ParquetOutputConfig(
        dataset_path=tmpdir,
        validate_schema=True,
        compression="gzip",
        row_group_size=300,
        keep_same_partitioning=False,
        partition_columns=["cat"],
    )

    table = get_random_table(100)
    poc.expected_schema = table.schema

    single_path_ = poc.write_batch(
        table.filter(ds.field("cat") == 5), iteration_index=(111,)
    )
    assert len(single_path_) == 1
    single_path = single_path_[0]
    assert str(single_path).endswith("_111.parquet")
    assert str(single_path.parent.name) == "cat=5"
    assert permutationally_equal_dataframes(
        pq.read_table(poc.dataset_path).to_pandas(),
        table.filter(ds.field("cat") == 5).to_pandas(),
    )

    multi_path_ = poc.write_batch(
        table.filter(ds.field("cat").isin([3, 2])), iteration_index=(222,)
    )
    assert len(multi_path_) == 2
    assert all(str(single_path).endswith("_222.parquet") for single_path in multi_path_)
    assert str(multi_path_[0].parent.name) == "cat=2"
    assert str(multi_path_[1].parent.name) == "cat=3"
    expected_table = pa.concat_tables(
        [table.filter(ds.field("cat") == 5), table.filter(ds.field("cat").isin([3, 2]))]
    )
    assert permutationally_equal_dataframes(
        pq.read_table(poc.dataset_path).to_pandas(), expected_table.to_pandas()
    )

    try:  # changing schema
        poc.write_batch(table.select(["cat"]), iteration_index=(2,))
        raise Exception("should not get there")
    except Exception as ex:
        assert "Item has schema" in str(ex)

    shutil.rmtree(tmpdir)


def test_parquet_sharding_config_single_file_many_fragements(
    multi_partition_file_dataset,
):
    psc = ParquetShardingConfig(
        input_file=str(multi_partition_file_dataset),
        batch_size=11,
        fragment_group_size=2,
        batch_format=BatchFormat.ARROW,
        filters_expr="(ds.field('score') < 0) & (ds.field('cat') >= 2)",
    )

    assert psc.partition_columns == ["cat"]
    shards = psc.make_shards()
    assert len(shards) == 4

    # all columns are present
    assert set(tuple(shard.columns) for shard in shards) == set(  # type: ignore
        [("name", "score", "cat")]
    )
    assert set(tuple(shard.partition_columns) for shard in shards) == set([("cat",)])  # type: ignore

    ## full dataset
    full_ds = pa.concat_tables(
        [
            bb
            for shard in shards
            for bb in shard.to_batches(
                None, columns=psc.columns, batch_format=psc.batch_format
            )
        ]
    )
    reload_full_ds = pq.read_table(str(multi_partition_file_dataset)).filter(psc.filter)
    assert permutationally_equal_dataframes(
        full_ds.to_pandas(), reload_full_ds.to_pandas()
    )

    first_shard = shards[0]
    assert isinstance(first_shard, ParquetShard)
    assert first_shard.nb_rows == 195  # unfiltered numbers

    # working with full batch
    batches = list(
        first_shard.to_batches(
            batch_size=None, columns=psc.columns, batch_format=psc.batch_format
        )
    )
    assert len(batches) == 1
    assert [len(bb) for bb in batches] == [94]
    assert [list(bb.column_names) for bb in batches] == [["name", "score", "cat"]]  # type: ignore

    one_batch = batches[0]
    assert one_batch["score"].to_numpy().max() < 0  # type: ignore
    assert one_batch["cat"].to_numpy().min() >= 2  # type: ignore

    batches = list(
        first_shard.to_batches(
            batch_size=psc.batch_size,
            columns=psc.columns,
            batch_format=psc.batch_format,
        )
    )
    assert len(batches) == 9
    assert [len(bb) for bb in batches] == [11] * 8 + [6]


def test_parquet_sharding_config_single_file_many_fragements_and_nb_samples_per_group(
    multi_partition_file_dataset,
):
    psc = ParquetShardingConfig(
        input_file=str(multi_partition_file_dataset),
        batch_size=11,
        nb_samples_per_group=330,
        batch_format=BatchFormat.ARROW,
        filters_expr="ds.field('cat') >= 1",
    )

    assert psc.partition_columns == ["cat"]
    shards = psc.make_shards()
    assert len(shards) == 3

    # all columns are present
    assert set(tuple(shard.columns) for shard in shards) == set(  # type: ignore
        [("name", "score", "cat")]
    )
    assert set(tuple(shard.partition_columns) for shard in shards) == set([("cat",)])  # type: ignore

    ## full dataset
    tables = [
        bb
        for shard in shards
        for bb in shard.to_batches(
            None, columns=psc.columns, batch_format=psc.batch_format
        )
    ]
    assert [len(bb) for bb in tables] == [401, 400, 96]
    full_ds = pa.concat_tables(tables)
    reload_full_ds = pq.read_table(str(multi_partition_file_dataset)).filter(psc.filter)
    assert permutationally_equal_dataframes(
        full_ds.to_pandas(), reload_full_ds.to_pandas()
    )
