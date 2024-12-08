# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import itertools
import logging
import random
import re
from pathlib import Path
from typing import Iterator, List

import pandas as pd
import pytest

from stopes.core.utils import open_write
from stopes.utils.sharding import text_shards as sh
from stopes.utils.sharding.abstract_shards import BatchFormat, concat_batches
from stopes.utils.sharding.text_shards import (
    ChunkShard,
    RoundRobinShard,
    TextOutputConfig,
    TextShard,
    TextShardingConfig,
)
from stopes.utils.tests.test_parquet_shards import (
    get_random_table,
    permutationally_equal_dataframes,
)


def test_shard_context(tmp_path: Path):
    tmp_file = tmp_path / "tmp.tsv.gz"

    # Test that when a shard enters, the underlying input file must exist
    with pytest.raises(FileNotFoundError):
        with TextShard(input_file=tmp_file, filter=None) as _:
            pass


def test_shard_internal_update(tmp_path: Path):
    tmp_file = tmp_path / "tmp.tsv.gz"
    # Test that the shard.lines is properly updated during shard iteration
    line_cnt = 10
    with open_write(tmp_file, "w") as o:
        [o.write(f"line{i}\n") for i in range(line_cnt)]

    shard = TextShard(input_file=tmp_file, filter=None)
    break_point = 4
    with shard as first_pass:
        for i, line in enumerate(iter(first_pass)):
            assert line.rstrip() == f"line{i}"
            if i == break_point:
                break

    assert shard.has_started()
    with shard as second_pass:
        for i, line in enumerate(iter(second_pass)):
            assert line.rstrip() == f"line{i + break_point + 1}"
    assert shard._lines_cnt == line_cnt


@pytest.mark.parametrize(
    "input_file", ["tmp.tsv", "tmp.tsv.gz", "tmp.tsv_expanded.txt"]
)
@pytest.mark.parametrize("output_file", [None, "output.txt", "outdir", "outfile"])
@pytest.mark.parametrize("suffix", ["", ".tsv.gz"])
def test_resolve_output(tmp_path: Path, input_file, output_file, suffix):
    tmp_file = tmp_path / input_file
    if output_file:
        (tmp_path / "outdir").mkdir()
        if output_file == "outdir":
            output_file = tmp_path / "outdir"
        else:
            output_file = tmp_path / "outdir" / output_file
            output_file.touch()

    shard1 = TextShard(input_file=tmp_file, filter=None)
    resolved_output_file = sh.resolve_output(shard1, output_file, suffix)

    if output_file is None:
        if suffix == "" and input_file == "tmp.tsv":
            assert resolved_output_file is None
        elif suffix == ".tsv.gz" and input_file == "tmp.tsv.gz":
            assert resolved_output_file is None
        else:
            assert resolved_output_file is not None
            assert resolved_output_file.parent == tmp_path
            output_name = "tmp.tsv" if suffix == "" else "tmp.tsv.gz"
            assert resolved_output_file.name == output_name
    elif suffix != "":
        assert resolved_output_file is not None
        assert resolved_output_file.parent == tmp_path / "outdir"
        assert resolved_output_file.suffix == ".gz"
    elif output_file.is_file():
        assert resolved_output_file == output_file
    else:
        assert resolved_output_file is not None
        assert resolved_output_file.parent == tmp_path / "outdir"
        assert resolved_output_file.name == "tmp.tsv"

    shard2 = TextShard(input_file=tmp_file, index=5, filter=None)
    resolved_output_shard = sh.resolve_output(shard2, output_file, suffix)
    assert resolved_output_shard is not None
    if suffix == ".tsv.gz":
        assert resolved_output_shard.suffixes == [".5", ".tsv", ".gz"]
    elif output_file is None or output_file.name == "outdir":
        assert resolved_output_shard.suffixes == [".5", ".tsv"]
    elif output_file.name == "outfile":
        assert resolved_output_shard.suffixes == [".5"]
    else:
        assert resolved_output_shard.suffixes == [".5", ".txt"]


@pytest.mark.parametrize("header", [False, True])
def test_shard_values(header, tmp_path: Path):
    tmp_file = tmp_path / "tmp.tsv"
    with open(tmp_file, "w") as o:
        if header:
            o.write("header1\theader2\n")
        o.write("val1\tval2\n")
        o.write("val3\tval4\n")

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    cols = ["header1", "header2"] if header else None
    col_name = "header2" if header else 1

    shard = TextShard(input_file=tmp_file, columns=cols, filter=None)
    out_file = sh.resolve_output(shard, output_dir, suffix=".txt")
    assert out_file is not None
    assert out_file.resolve() == output_dir / "tmp.txt"

    with open(out_file, "w") as o, shard as progress:
        for line in progress:
            val = progress.value(col_name)  # type: ignore
            o.write(val)

    vals = open(out_file).readline()
    assert vals == "val2val4"


def test_shard_headers():
    for i in range(5):
        shard = TextShard(
            input_file=Path("input.tsv"), columns=["header"], index=i, filter=None
        )

        # The first shard should contains the header if the `cols` is passed
        if i == 0:
            assert shard.contains_header()
        else:
            assert not shard.contains_header()


def test_make_one_shard(tmp_path: Path):
    with open(tmp_path / "tmp.tsv", "w") as o:
        o.write("header1;heade2;header3\n")
        o.write("1;2;3\n")

    shards = sh.make_one_text_shard(tmp_path / "tmp.tsv", header=True, sep=";")
    assert len(shards) == 1
    with shards[0] as progress:
        next(iter(progress))
        col = progress.value("header3")
        assert int(col) == 3


def test_one_shard_with_nrows(tmp_path: Path):
    with open(tmp_path / "tmp.tsv", "w") as o:
        o.write("header1|heade2|header3\n")
        [o.write("col1|col2|col3\n") for _ in range(10)]

    shards = sh.make_one_text_shard(tmp_path / "tmp.tsv", header=True, sep="|", nrows=5)
    with shards[0] as shard:
        col = shard.resolve_column_index("header3")
        assert col == 2
        lines = list(iter(shard))
        assert len(lines) == 5
        assert lines[0].rstrip().split("|")[col] == "col3"


def test_make_shards_from_file_list(tmp_path: Path):
    input_files = []
    for i in range(5):
        input_file = tmp_path / f"file{i}.tsv"
        input_file.touch()
        input_files.append(input_file)

    # `nshards`` should have no impact here
    shards = sh.make_text_file_shards(input_files, nshards=50)
    assert len(shards) == len(input_files)
    for i, p in enumerate(shards):
        assert Path(p.input_file).name == f"file{i}.tsv"


@pytest.mark.parametrize("zip", [False, True])
@pytest.mark.parametrize("header", [False, True])
def test_make_chunk_shards_from_single_file(tmp_path: Path, zip: bool, header: bool):
    if zip:
        tmp_file = tmp_path / "tmp.tsv.gz"
    else:
        tmp_file = tmp_path / "tmp.tsv"
    with open_write(tmp_file) as o:
        if header:
            o.write("header1|header2\n")
        [o.write(f"line{i}|val{i}\n") for i in range(23)]
    out_dir = tmp_path / "fakedir"
    out_dir.mkdir()
    shards = sh.make_text_file_shards(
        tmp_file, nshards=5, header=header, sep="|", cache_dir=out_dir
    )

    # Change filename to reflect fairseq internal decompression
    if zip:
        tmp_file = out_dir / (tmp_file.name + "_expanded.txt")

    if header:
        cols = ["header1", "header2"]
        col_name = "header2"
        expected_shards = [
            ChunkShard(
                input_file=tmp_file,
                columns=cols,
                sep="|",
                index=0,
                start=0,
                end=60,
                filter=None,
            ),
            ChunkShard(
                input_file=tmp_file,
                columns=cols,
                sep="|",
                index=1,
                start=60,
                end=126,
                filter=None,
            ),
            ChunkShard(
                input_file=tmp_file,
                columns=cols,
                sep="|",
                index=2,
                start=126,
                end=178,
                filter=None,
            ),
            ChunkShard(
                input_file=tmp_file,
                columns=cols,
                sep="|",
                index=3,
                start=178,
                end=243,
                filter=None,
            ),
            ChunkShard(
                input_file=tmp_file,
                columns=cols,
                sep="|",
                index=4,
                start=243,
                end=295,
                filter=None,
            ),
        ]
    else:
        col_name = 1  # type: ignore
        expected_shards = [
            ChunkShard(
                input_file=tmp_file, sep="|", index=0, start=0, end=66, filter=None
            ),
            ChunkShard(
                input_file=tmp_file, sep="|", index=1, start=66, end=123, filter=None
            ),
            ChunkShard(
                input_file=tmp_file, sep="|", index=2, start=123, end=175, filter=None
            ),
            ChunkShard(
                input_file=tmp_file, sep="|", index=3, start=175, end=227, filter=None
            ),
            ChunkShard(
                input_file=tmp_file, sep="|", index=4, start=227, end=279, filter=None
            ),
        ]
    assert len(shards) == len(expected_shards)

    line_idx = 0
    for i in range(5):
        assert shards[i] == expected_shards[i]
        with shards[i] as progress:
            for _ in progress:
                assert progress.value(col_name) == f"val{line_idx}"
                line_idx += 1


@pytest.mark.parametrize("zip", [False, True])
@pytest.mark.parametrize("header", [False, True])
def test_make_chunk_shards_with_nrows(tmp_path: Path, zip: bool, header: bool):
    if zip:
        tmp_file = tmp_path / "tmp.tsv.gz"
    else:
        tmp_file = tmp_path / "tmp.tsv"
    with open_write(tmp_file) as o:
        if header:
            o.write("header1|header2\n")
        [o.write(f"line{i}|{i}\n") for i in range(50)]
    # Change filename to reflect fairseq internal decompression
    shards = sh.make_text_file_shards(
        tmp_file, nshards=5, header=header, sep="|", nrows=23
    )

    if zip:
        tmp_filename = tmp_file.name + "_expanded.txt"
    else:
        tmp_filename = tmp_file.name
    real_input = tmp_file.parent / tmp_filename
    if header:
        cols = ["header1", "header2"]
        col_name = "header2"
        expected_shards = [
            ChunkShard(
                input_file=real_input,
                columns=cols,
                sep="|",
                index=0,
                start=0,
                end=48,
                filter=None,
            ),
            ChunkShard(
                input_file=real_input,
                columns=cols,
                sep="|",
                index=1,
                start=48,
                end=88,
                filter=None,
            ),
            ChunkShard(
                input_file=real_input,
                columns=cols,
                sep="|",
                index=2,
                start=88,
                end=136,
                filter=None,
            ),
            ChunkShard(
                input_file=real_input,
                columns=cols,
                sep="|",
                index=3,
                start=136,
                end=176,
                filter=None,
            ),
            ChunkShard(
                input_file=real_input,
                columns=cols,
                sep="|",
                index=4,
                start=176,
                end=216,
                filter=None,
            ),
        ]
    else:
        col_name = 1  # type: ignore[assignment]
        expected_shards = [
            ChunkShard(
                input_file=real_input, sep="|", index=0, start=0, end=40, filter=None
            ),
            ChunkShard(
                input_file=real_input, sep="|", index=1, start=40, end=80, filter=None
            ),
            ChunkShard(
                input_file=real_input, sep="|", index=2, start=80, end=130, filter=None
            ),
            ChunkShard(
                input_file=real_input, sep="|", index=3, start=130, end=170, filter=None
            ),
            ChunkShard(
                input_file=real_input, sep="|", index=4, start=170, end=210, filter=None
            ),
        ]

    assert len(shards) == len(expected_shards)
    line_idx = 0
    for i in range(5):
        assert shards[i] == expected_shards[i]
        with shards[i] as shard:
            for _ in iter(shard):
                assert int(shard.value(col_name)) == line_idx
                line_idx += 1
    assert line_idx == (22 if header else 23)


@pytest.mark.parametrize("header", [False, True])
def test_make_robin_shards(tmp_path: Path, header: bool):
    nshards = 5
    tmp_file = tmp_path / "tmp.tsv.gz"
    with open_write(tmp_file) as o:
        if header:
            o.write("header1\theader2\n")
        [o.write(f"line\t{i}\n") for i in range(23)]
    sep = "\t" if header else None
    shards = sh.make_text_file_shards(
        tmp_file, nshards=nshards, algo="robin", header=header, sep=sep
    )
    if header:
        cols = ["header1", "header2"]
        col_name = "header2"
        expected_shards = [
            RoundRobinShard(
                input_file=tmp_file,
                columns=cols,
                sep=sep,
                index=i,
                nshards=nshards,
                filter=None,
            )
            for i in range(nshards)
        ]
        expected_vals = [
            [4, 9, 14, 19],
            [0, 5, 10, 15, 20],
            [1, 6, 11, 16, 21],
            [2, 7, 12, 17, 22],
            [3, 8, 13, 18],
        ]
    else:
        col_name = 1  # type: ignore
        expected_shards = [
            RoundRobinShard(input_file=tmp_file, index=i, nshards=nshards, filter=None)
            for i in range(nshards)
        ]
        expected_vals = [
            [0, 5, 10, 15, 20],
            [1, 6, 11, 16, 21],
            [2, 7, 12, 17, 22],
            [3, 8, 13, 18],
            [4, 9, 14, 19],
        ]
    assert shards == expected_shards
    if header:
        assert shards[0].contains_header()
    assert all([not x.contains_header() for x in shards[1:]])

    for i in range(5):
        with shards[i] as progress:
            vals = [int(progress.value(col_name)) for _ in progress]
            assert list(vals) == expected_vals[i]


@pytest.mark.parametrize("header", [False, True])
def test_make_sorted_shards(tmp_path: Path, header: bool):
    import string

    tmp_file = tmp_path / "tmp.tsv"
    with open_write(tmp_file) as o:
        if header:
            o.write("header1\theader2\n")
        chs = list(string.ascii_lowercase)
        random.shuffle(chs)
        [o.write(f"a really long line\t{i}\n") for i in chs]

    col = "header2" if header else 1
    shards = sh.make_text_file_shards(
        tmp_file, nshards=5, algo="sort", header=header, sep="\t", col=col, filter=None
    )
    sorted_file = str(tmp_file) + ".merge_sort"
    assert Path(sorted_file).exists()
    with open(sorted_file) as f:
        lines = iter(f)
        if header:
            lines = itertools.islice(iter(f), 1, None)
        for line, ch in zip(lines, string.ascii_lowercase):
            assert line == f"a really long line\t{ch}\n"
    assert len(shards) == 5
    if header:
        assert shards[0] == ChunkShard(
            input_file=sorted_file,
            sep="\t",
            columns=["header1", "header2"],
            index=0,
            start=0,
            end=121,
            filter=None,
        )
    else:
        assert shards[0] == ChunkShard(
            input_file=sorted_file, sep="\t", index=0, start=0, end=126, filter=None
        )


def test_make_shards_from_glob(tmp_path: Path):
    (tmp_path / "file1.tsv").touch()
    (tmp_path / "file2.tsv").touch()

    shards = list(
        sh.make_text_file_shards(tmp_path / "file*.tsv", cache_dir=Path("fakedir"))
    )
    assert len(shards) == 2
    shards.sort(key=lambda x: Path(x.input_file).name)
    assert Path(shards[0].input_file).name == "file1.tsv"
    assert Path(shards[1].input_file).name == "file2.tsv"


@contextlib.contextmanager
def assert_warns(caplog, *, match: str) -> Iterator[None]:
    caplog.clear()
    sh.warn_once.cache_clear()

    with caplog.at_level(logging.WARN):
        yield
        assert len(caplog.messages) == 1
        assert re.match(match, caplog.messages[0])
        caplog.clear()


def test_text_sharding_config(tmp_path: Path):
    df1 = get_random_table(10**3, seed=111).to_pandas()
    df2 = get_random_table(10**2, seed=222).to_pandas()

    df1.to_csv(tmp_path / "file1.csv", index=False)
    df2.to_csv(tmp_path / "file2.csv", index=False)

    tsc = TextShardingConfig(
        input_file=str(tmp_path / "file*.csv"),
        batch_size=11,
        columns=["score", "cat"],
        batch_format=BatchFormat.PANDAS,
        filters_expr="(ds.field('score') < 0) & (ds.field('cat') >= 5)",
        sep=",",
    )
    shards: List[pd.DataFrame] = tsc.make_shards()
    assert len(shards) == 2
    full_batches: pd.DataFrame = concat_batches(
        [bb for shard in shards for bb in shard.to_batches(batch_size=None)]
    )
    assert len(full_batches) == 280
    expected_df: pd.DataFrame = concat_batches([df1, df2])
    expected_df = expected_df[(expected_df["score"] < 0) & (expected_df["cat"] >= 5)]
    assert list(full_batches.columns) == ["cat", "name", "score"]
    assert permutationally_equal_dataframes(full_batches, expected_df)

    tsc.path_column = "my_custom_path_name"
    another_full_batches: pd.DataFrame = concat_batches(
        [
            bb
            for shard in tsc.make_shards()
            for bb in shard.to_batches(batch_size=tsc.batch_size)
        ]
    )

    assert list(another_full_batches.columns) == [
        "cat",
        "name",
        "score",
        "my_custom_path_name",
    ]
    assert list(another_full_batches["my_custom_path_name"].unique()) == [
        str(tmp_path / "file1.csv"),
        str(tmp_path / "file2.csv"),
    ]

    another_full_batches = another_full_batches[["cat", "name", "score"]]
    assert permutationally_equal_dataframes(full_batches, another_full_batches)

    assert list(map(len, shards[0].to_batches(batch_size=tsc.batch_size))) == [
        11
    ] * 22 + [10]
    assert list(map(len, shards[1].to_batches(batch_size=tsc.batch_size))) == [
        11
    ] * 2 + [6]

    single_df_few_columns = next(
        shards[0].to_batches(
            batch_size=tsc.batch_size,
            columns=tsc.columns,
            batch_format=tsc.batch_format,
        )
    )
    assert list(single_df_few_columns.columns) == tsc.columns
    single_df_all_columns = next(
        shards[0].to_batches(batch_size=tsc.batch_size, batch_format=tsc.batch_format)
    )
    assert permutationally_equal_dataframes(
        single_df_few_columns, single_df_all_columns[tsc.columns]
    )


def test_output_text_config(tmp_path: Path):
    toc = TextOutputConfig(str(tmp_path), compression="tar", sep="\t")
    table1 = get_random_table(10**3, seed=211)

    paths_ = toc.write_batch(table1, iteration_index=(555,))
    assert len(paths_) == 1
    assert str(paths_[0]).endswith("555.tsv.tar")
    reload_df = pd.read_csv(paths_[0], sep=toc.sep)
    assert permutationally_equal_dataframes(reload_df, table1.to_pandas())

    table2 = get_random_table(10**2, seed=231)

    paths_ = toc.write_batch(table2, iteration_index=(777,))
    assert len(paths_) == 1
    assert str(paths_[0]).endswith("777.tsv.tar")
    reload_df = pd.read_csv(paths_[0], sep=toc.sep)
    assert permutationally_equal_dataframes(reload_df, table2.to_pandas())


def test_list_header_in_text_sharding_config(tmp_path: Path):
    df1 = get_random_table(10**3, seed=111).to_pandas()
    df2 = get_random_table(10**2, seed=222).to_pandas()

    df1.to_csv(tmp_path / "file1.csv", index=False)
    df2.to_csv(tmp_path / "file2.csv", index=False)

    tsc = TextShardingConfig(
        input_file=str(tmp_path / "file*.csv"),
        batch_size=11,
        header=["CAT", "NAME", "SCORE"],  # upcase for tests
        columns=["SCORE", "CAT"],
        filters_expr="(ds.field('SCORE') < 0) & (ds.field('CAT') >= 5)",
        batch_format=BatchFormat.PANDAS,
        sep=",",
    )
    shards: List[pd.DataFrame] = tsc.make_shards()
    assert len(shards) == 2
    full_batches: pd.DataFrame = concat_batches(
        [bb for shard in shards for bb in shard.to_batches(batch_size=None)]
    )
    assert len(full_batches) == 280
    expected_df: pd.DataFrame = concat_batches([df1, df2])
    expected_df = expected_df[(expected_df["score"] < 0) & (expected_df["cat"] >= 5)]
    assert list(full_batches.columns) == ["CAT", "NAME", "SCORE"]
    assert permutationally_equal_dataframes(
        full_batches, expected_df.rename(columns=str.upper)
    )
