# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import random
import shutil
import string
import tempfile
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def sanitize_categorical(df: pd.DataFrame) -> pd.DataFrame:
    for col, type_ in df.dtypes.items():
        if type_ == "category":
            df[col] = df[col].astype("object")
    return df


def permutationally_equal_dataframes(one: pd.DataFrame, other: pd.DataFrame) -> bool:
    assert len(one) == len(other)
    assert sorted(one.columns) == sorted(other.columns)
    one = sanitize_categorical(one)[sorted(one.columns)]
    other = sanitize_categorical(other)[sorted(other.columns)]

    one = one.sort_values(list(one.columns)).set_index(np.arange(len(one)))
    other = other.sort_values(list(other.columns)).set_index(np.arange(len(other)))

    assert (one == other).all(axis=None)
    return True


def get_random_table(size: int, seed: int = 123) -> pa.Table:
    rds = np.random.RandomState(seed)
    data = {
        "cat": rds.randint(0, 10, size),
        "name": ["name_" + str(i) for i in range(size)],
        "score": np.round(rds.randn(size), 7),
    }
    return pa.Table.from_pydict(data)


@pytest.fixture()
def single_file_dataset() -> tp.Generator[Path, None, None]:
    tmpdir = tempfile.mkdtemp()
    tmp_parquet_ds_path = Path(tmpdir) / "test1"
    table = get_random_table(10**3)
    pq.write_table(table, tmp_parquet_ds_path)
    yield tmp_parquet_ds_path
    shutil.rmtree(tmpdir)


@pytest.fixture()
def multi_partition_file_dataset() -> tp.Generator[Path, None, None]:
    tmpdir = tempfile.mkdtemp()
    tmp_parquet_ds_path = Path(tmpdir) / "test2"

    table = get_random_table(10**3)
    pq.write_to_dataset(table, tmp_parquet_ds_path, partition_cols=["cat"])

    yield tmp_parquet_ds_path
    shutil.rmtree(tmpdir)


def gen_random_string(length: int) -> str:
    return "".join(
        random.choice(string.ascii_letters + string.digits) for n in range(length)
    )


def generate_random_pandas_df(size: int, seed: int = 123) -> pd.DataFrame:
    np_rs = np.random.RandomState(seed)
    df: tp.Dict[str, tp.Union[np.ndarray, list]] = {}
    df["int_col"] = np_rs.randint(0, 200, size)
    df["float_col"] = np_rs.randn(size)

    df["string_col1"] = [gen_random_string(10) for _ in range(size)]
    df["string_col2"] = [gen_random_string(2) for _ in range(size)]

    df["list_int_col"] = [
        np_rs.randint(-10, 10, np_rs.randint(0, 100)) for _ in range(size)
    ]
    df["list_float_col"] = [
        np_rs.rand(np_rs.randint(0, 10)).astype(np.float32) for _ in range(size)
    ]
    df["list_float_fixed_size_col"] = [
        np_rs.rand(7).astype(np.float32) for _ in range(size)
    ]
    return pd.DataFrame(df)


def generated_partitioned_parquet_file(
    path: str, size: int, n_partitions: int = 20, seed: int = 123
) -> None:
    df = generate_random_pandas_df(size, seed)

    if n_partitions > 0:
        df["part_key"] = np.arange(size) % n_partitions

    table = pa.Table.from_pandas(df)

    pq.write_to_dataset(
        table,
        path,
        partition_cols=["part_key"] if n_partitions > 0 else None,
        existing_data_behavior="delete_matching",
    )
