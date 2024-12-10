# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import tempfile
import unittest
from collections import Counter

from conftest import generated_partitioned_parquet_file

from stopes.utils.parquet_dataloader import ParquetBasicDataLoader, np, pa, pd


class TestParquetDataloader(unittest.TestCase):
    _tmpdir: str
    _tmp_parquet_ds_path: str
    _tmp_parquet_single_path: str

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.mkdtemp()
        cls._tmp_parquet_ds_path = os.path.join(cls._tmpdir, "test")
        generated_partitioned_parquet_file(cls._tmp_parquet_ds_path, size=2 * 10**3)

        cls._tmp_parquet_single_path = os.path.join(cls._tmpdir, "single_test.parquet")
        generated_partitioned_parquet_file(
            cls._tmp_parquet_single_path, size=10**3, n_partitions=0
        )

    @classmethod
    def tearDownClass(cls) -> None:
        # Cleanup temp working dir.
        if cls._tmpdir is not None:
            shutil.rmtree(cls._tmpdir)  # type: ignore

    def test_basic_dataload(self):
        pbdl = ParquetBasicDataLoader(self._tmp_parquet_ds_path, to_pandas=False)
        ei_batch = pbdl.epoch_iterator(11, nb_producers=2, nb_cpu=4, seed=333)
        res = [tt for tt in ei_batch]

        for x in res:
            self.assertIsInstance(x, pa.Table)

        self.assertEqual(
            list(res[0].to_pandas().columns),
            [
                "int_col",
                "float_col",
                "string_col1",
                "string_col2",
                "list_int_col",
                "list_float_col",
                "list_float_fixed_size_col",
                "part_key",
            ],
        )
        self.assertEqual(Counter(map(len, res)), Counter({11: 180, 2: 10}))  # 180 * 11
        self.assertEqual(sum(map(len, res)), 2000)

        # determinism check
        pbdl_new = ParquetBasicDataLoader(self._tmp_parquet_ds_path, to_pandas=True)
        ei_batch_bis = pbdl_new.epoch_iterator(11, nb_producers=2, nb_cpu=4, seed=333)
        res_bis = [tt for tt in ei_batch_bis]

        for x in res_bis:
            self.assertIsInstance(x, pd.DataFrame)

        self.assertTrue(
            all(
                (x["float_col"].to_pandas() == y["float_col"]).all()
                for x, y in zip(res, res_bis)
            )
        )
        # another seed
        ei_batch_ter = pbdl_new.epoch_iterator(11, nb_producers=2, nb_cpu=4, seed=222)
        res_ter = [tt for tt in ei_batch_ter]
        self.assertTrue(
            any((x["float_col"] != y["float_col"]).any() for x, y in zip(res, res_ter))
        )

    def test_filtered_with_columns_dataload(self):
        pbdl = ParquetBasicDataLoader(
            self._tmp_parquet_ds_path,
            columns=["string_col2", "list_int_col", "float_col"],
            filters=[("float_col", ">", 0)],
            to_pandas=True,
        )
        ei_batch = pbdl.epoch_iterator(3, nb_producers=2, nb_cpu=4, seed=111)
        res = [tt for tt in ei_batch]

        self.assertEqual(
            list(res[0].columns), ["string_col2", "list_int_col", "float_col"]
        )
        self.assertEqual(Counter(map(len, res)), Counter({3: 338, 1: 4, 2: 2}))

        ei_batch_trunc = pbdl.epoch_iterator(3, min_sample_size=3, seed=111)
        res = [tt for tt in ei_batch_trunc]
        self.assertEqual(Counter(map(len, res)), Counter({3: 340}))

    def test_ordered_dataload(self):
        pbdl = ParquetBasicDataLoader(self._tmp_parquet_ds_path, to_pandas=True)
        ei_ordered = pbdl.epoch_iterator(
            batch_size=20,
            order_by_length="list_int_col",
            nb_producers=20,
            nb_cpu=2,
            seed=123,
        )
        length_by_batches = [tt["list_int_col"].apply(len) for tt in ei_ordered]
        length_by_batches_diff = max(tt.max() - tt.min() for tt in length_by_batches)
        total_length = sum(map(len, length_by_batches))

        self.assertLess(length_by_batches_diff, 4)
        self.assertEqual(total_length, 2000)
        self.assertTrue(all(len(tt) == 20 for tt in length_by_batches))

    def test_ordered_max_token_dataload(self):
        pbdl = ParquetBasicDataLoader(self._tmp_parquet_ds_path, to_pandas=True)

        ei_max_token = pbdl.epoch_iterator(
            max_tokens=3000,
            order_by_length="list_int_col",
            nb_producers=20,
            nb_cpu=2,
            seed=123,
        )

        length_by_batches = [tt["list_int_col"].apply(len) for tt in ei_max_token]
        length_by_batches_diff = max(tt.max() - tt.min() for tt in length_by_batches)
        max_padded_total_length = max(tt.max() * len(tt) for tt in length_by_batches)
        mean_padded_total_length = np.mean(
            [tt.max() * len(tt) for tt in length_by_batches]
        )
        total_length = sum(map(len, length_by_batches))

        self.assertLessEqual(length_by_batches_diff, 12)
        self.assertEqual(total_length, 2000)
        self.assertLessEqual(max_padded_total_length, 3000)
        self.assertGreater(mean_padded_total_length, 2900)

    def test_ordered_max_token_simple_dataload(self):
        pbdl_single = ParquetBasicDataLoader(
            self._tmp_parquet_single_path, to_pandas=False
        )
        ei_batch = pbdl_single.epoch_iterator(10, nb_producers=2, nb_cpu=4, seed=333)
        res = [tt for tt in ei_batch]

        self.assertEqual(Counter(map(len, res)), Counter({10: 100}))
        self.assertEqual(
            res[0].column_names,
            [
                "int_col",
                "float_col",
                "string_col1",
                "string_col2",
                "list_int_col",
                "list_float_col",
                "list_float_fixed_size_col",
            ],
        )
