# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from stopes.modules.partitioned_data_mapper import (
    PartitionedDataMapper,
    PartitionedDataMapperConfig,
)
from stopes.utils.sharding.parquet_shards import (
    ParquetOutputConfig,
    ParquetShardingConfig,
)
from stopes.utils.sharding.text_shards import TextOutputConfig, TextShardingConfig
from stopes.utils.tests.conftest import permutationally_equal_dataframes

NUM_ROW_GROUPS = 15
FULL_DS_SIZE = 3 * 10**5


class IdentityPartitionedDataMapper(PartitionedDataMapper):
    def get_batch_mapper(self):
        return lambda batch: batch

    def requirements(self):
        ...

    def get_custom_metadata(self, *args, **kwargs) -> Dict[str, Any]:
        return {}


def generated_partitioned_parquet_dataset(
    path: str, size: int, n_partitions: int = 5, seed: int = 123
) -> None:
    np_rs = np.random.RandomState(seed)
    df = {
        "int_col": np_rs.randint(0, 200, size),
        "float_col": np.round(np_rs.randn(size), 10),
        "bool_col": np_rs.randn(size) > 0,
        "part_key": np.arange(size) % (n_partitions + 1),
    }

    table = pa.Table.from_pydict(df)

    pq.write_to_dataset(
        table,
        path,
        partition_cols=["part_key"] if n_partitions > 0 else None,
        **{"max_rows_per_group": 2000, "min_rows_per_group": 1000},
    )
    df_pd = table.to_pandas()
    df_pd.to_csv(os.path.join(path, ".tsv"), sep="\t", index=False)


class TestPartitionedDataMapper(unittest.TestCase):
    _tmpdir: str
    _tmp_parquet_ds_path: str
    _tmp_parquet_single_path: str

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.mkdtemp()
        cls._tmp_parquet_ds_path = os.path.join(cls._tmpdir, "test")
        generated_partitioned_parquet_dataset(
            cls._tmp_parquet_ds_path, size=FULL_DS_SIZE
        )

        cls._tmp_parquet_single_path = os.path.join(cls._tmpdir, "single_test.parquet")
        generated_partitioned_parquet_dataset(
            cls._tmp_parquet_single_path, size=FULL_DS_SIZE, n_partitions=0
        )

    @classmethod
    def tearDownClass(cls) -> None:
        # Cleanup temp working dir.
        if cls._tmpdir is not None:
            shutil.rmtree(cls._tmpdir)  # type: ignore

    def test_basic_parquet_to_parquet_walkthrough(self):
        input_config = ParquetShardingConfig(
            input_file=self._tmp_parquet_ds_path,
            batch_size=100,
            split_row_groups=True,
            fragment_group_size=2,
            filters_expr="pa.compute.and_kleene(pa.compute.greater(ds.field('part_key'), 2), pa.compute.greater(ds.field('float_col'), 0))",
            columns=["part_key", "float_col", "bool_col"],
        )
        output_config = ParquetOutputConfig(
            os.path.join(self._tmpdir, "output"),
            batch_size=10**3,
            max_rows_per_file=600,
        )
        mapper_config = PartitionedDataMapperConfig(input_config, output_config)

        dpdm = IdentityPartitionedDataMapper(mapper_config)
        self.assertEqual(len(dpdm.array()), 15)  # so this's clearly split by row groups

        output_paths = [
            y for i, frag in enumerate(dpdm.array()) for y in dpdm.run(frag, i)  # type: ignore
        ]
        reloaded_dataset = pa.parquet.ParquetDataset(output_paths)

        reloaded_table = reloaded_dataset.read()
        input_table = pa.parquet.ParquetDataset(
            self._tmp_parquet_ds_path,
            filters=pa.compute.and_kleene(
                pa.compute.greater(ds.field("part_key"), 2),
                pa.compute.greater(ds.field("float_col"), 0),
            ),
        ).read(columns=["part_key", "float_col", "bool_col"])

        assert permutationally_equal_dataframes(
            reloaded_table.to_pandas(), input_table.to_pandas()
        )
        # check metadata
        metadata = reloaded_table.schema.metadata
        assert list(metadata.keys()) == [
            b"config",
            b"batch_mapper_class",
            b"batch_mapper_code",
            b"username",
            b"save_time",
            b"previous_metadata",
        ]
        assert metadata[b"batch_mapper_class"] == b'{"<class \'function\'>": ""}'
        assert (
            metadata[b"batch_mapper_code"] == b'"        return lambda batch: batch\\n"'
        )

        assert len(output_paths) == 153

    def test_basic_tsv_to_parquet_walkthrough(self):
        input_config = TextShardingConfig(
            input_file=os.path.join(self._tmp_parquet_ds_path, ".tsv"),
            columns=["part_key", "float_col"],
            filters_expr="pa.compute.greater(ds.field('float_col'), 0)",
            sep="\t",
            batch_size=211,
            nb_shards=11,
            header=True,
        )

        output_config = ParquetOutputConfig(
            os.path.join(self._tmpdir, "output_from_tsv"),
            keep_same_partitioning=False,
            partition_columns=["part_key"],
        )
        mapper_config = PartitionedDataMapperConfig(input_config, output_config)

        dpdm = IdentityPartitionedDataMapper(mapper_config)
        self.assertEqual(len(dpdm.array()), 11)

        output_paths = [
            y for i, frag in enumerate(dpdm.array()) for y in dpdm.run(frag, i)  # type: ignore
        ]
        self.assertEqual(len(output_paths), 11 * (5 + 1))

        reloaded_table = (
            pa.parquet.ParquetDataset(output_paths).read_pandas().to_pandas()
        )

        input_table = (
            pa.parquet.ParquetDataset(
                self._tmp_parquet_ds_path,
                filters=pa.compute.greater(ds.field("float_col"), 0),
            )
            .read_pandas(columns=["part_key", "float_col"])
            .to_pandas()
        )

        assert permutationally_equal_dataframes(reloaded_table, input_table)

    def test_basic_tsv_to_tsv_walkthrough(self):
        input_config = TextShardingConfig(
            input_file=os.path.join(self._tmp_parquet_ds_path, ".tsv"),
            columns=["part_key", "float_col"],
            filters_expr="pa.compute.greater(ds.field('float_col'), -2.)",
            sep="\t",
            batch_size=1112,
            nb_shards=110,
            header=True,
        )
        output_config = TextOutputConfig(
            os.path.join(self._tmpdir, "output_text"),
            sep="\t",
        )

        mapper_config = PartitionedDataMapperConfig(input_config, output_config)

        dpdm = IdentityPartitionedDataMapper(mapper_config)
        self.assertEqual(len(dpdm.array()), 110)

        output_paths = [
            y for i, frag in enumerate(dpdm.array()) for y in dpdm.run(frag, i)  # type: ignore
        ]
        self.assertEqual(len(output_paths), 110)
        self.assertTrue(all(str(path).endswith(".tsv") for path in output_paths))

        reloaded_table = pd.concat(
            [pd.read_csv(path, sep="\t", compression=None) for path in output_paths],
            axis=0,
        )

        input_table = pd.read_csv(
            os.path.join(self._tmp_parquet_ds_path, ".tsv"),
            sep="\t",
            usecols=["part_key", "float_col"],
        )
        input_table = input_table[input_table["float_col"] > -2.0]

        assert permutationally_equal_dataframes(reloaded_table, input_table)

    def test_basic_parquet_to_tsv_walkthrough(self):

        input_config = ParquetShardingConfig(
            input_file=self._tmp_parquet_ds_path,
            split_row_groups=False,
            columns=["part_key", "float_col", "bool_col"],
        )
        out_dir = os.path.join(self._tmpdir, "output_text")
        output_config = TextOutputConfig(out_dir, sep="\t", compression="gzip")
        mapper_config = PartitionedDataMapperConfig(input_config, output_config)

        dpdm = IdentityPartitionedDataMapper(mapper_config)
        shards = dpdm.array()
        self.assertEqual(len(shards), 6)  # so this's clearly split by row groups

        output_paths = [
            y for i, frag in enumerate(shards) for y in dpdm.run(frag, i)  # type: ignore
        ]
        self.assertEqual(len(output_paths), 6)
        self.assertTrue(all(str(path).endswith(".tsv.gzip") for path in output_paths))

        reloaded_table = pd.concat(
            [pd.read_csv(path, sep="\t", compression="gzip") for path in output_paths],
            axis=0,
        )

        input_table = (
            pa.parquet.ParquetDataset(
                self._tmp_parquet_ds_path,
            )
            .read_pandas(columns=["part_key", "float_col", "bool_col"])
            .to_pandas()
        )

        assert len(list(Path(out_dir).glob(".text_output.*.state"))) == len(shards)
        final_states = [output_config.reload_state(shard) for shard in shards]
        assert all([s is not None for s in final_states])
        total_states_row = sum([s.input_rows_written for s in final_states])  # type: ignore
        assert total_states_row == len(input_table)

        assert permutationally_equal_dataframes(reloaded_table, input_table)

    def test_limits_sharding_number(self):

        input_config_parq = ParquetShardingConfig(
            input_file=self._tmp_parquet_ds_path,
            split_row_groups=True,
            take=3,
            columns=["part_key", "float_col", "bool_col"],
        )

        self.assertEqual(
            len(input_config_parq.make_shards()), 3
        )  # so this's clearly split by row groups

        input_config_text = TextShardingConfig(
            input_file=os.path.join(self._tmp_parquet_ds_path, ".tsv"),
            take=2,
            filters_expr="pa.compute.greater(ds.field('float_col'), -2.)",
            sep="\t",
            batch_size=112,
            nb_shards=110,
            header=True,
        )

        self.assertEqual(
            len(input_config_text.make_shards()), 2
        )  # so this's clearly split by row groups

    def test_parquet_skipping(self):
        input_table = pa.parquet.ParquetDataset(
            self._tmp_parquet_ds_path,
            filters=pa.compute.and_kleene(
                pa.compute.greater(ds.field("part_key"), 2),
                pa.compute.greater(ds.field("float_col"), 0),
            ),
        ).read(columns=["part_key", "float_col", "bool_col"])

        original_len = len(input_table)
        to_skip = math.floor(
            (original_len / NUM_ROW_GROUPS) / 3
        )  # 1/3 of a shard, we have 15 shards

        input_config = ParquetShardingConfig(
            input_file=self._tmp_parquet_ds_path,
            batch_size=100,
            split_row_groups=True,
            fragment_group_size=2,
            filters_expr="pa.compute.and_kleene(pa.compute.greater(ds.field('part_key'), 2), pa.compute.greater(ds.field('float_col'), 0))",
            columns=["part_key", "float_col", "bool_col"],
            skip_n_rows_per_shard={0: to_skip},
        )
        out_dir = os.path.join(self._tmpdir, "output_parquet")
        output_config = ParquetOutputConfig(
            out_dir,
            batch_size=10**3,
            max_rows_per_file=600,
        )
        mapper_config = PartitionedDataMapperConfig(input_config, output_config)

        dpdm = IdentityPartitionedDataMapper(mapper_config)

        shards = dpdm.array()
        output_paths = [
            y for i, frag in enumerate(shards) for y in dpdm.run(frag, i)  # type: ignore
        ]

        reloaded_dataset = pa.parquet.ParquetDataset(output_paths)

        reloaded_table = reloaded_dataset.read()

        expected = original_len - to_skip
        actual = len(reloaded_table)
        assert expected == actual

        assert len(list(Path(out_dir).glob(".parquet_output.*.state"))) == len(shards)
        final_states = [output_config.reload_state(shard) for shard in shards]
        assert all([s is not None for s in final_states])
        total_states_row = sum([s.input_rows_written for s in final_states])  # type: ignore
        assert total_states_row == len(input_table)
