# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import itertools
import logging
import os
import random
import re
from pathlib import Path
from typing import Iterator

import pytest

from stopes.core.utils import open_write
from stopes.utils import shards as sh
from stopes.utils.shards import ChunkShard, RoundRobinShard, Shard


def test_shard_context(tmp_path: Path):
    tmp_file = tmp_path / "tmp.tsv.gz"

    # Test that when a shard enters, the underlying input file must exist
    with pytest.raises(FileNotFoundError):
        with Shard(tmp_file) as _:
            pass


def test_shard_internal_update(tmp_path: Path):
    tmp_file = tmp_path / "tmp.tsv.gz"
    # Test that the shard.lines is properly updated during shard iteration
    line_cnt = 10
    with open_write(tmp_file, "w") as o:
        [o.write(f"line{i}\n") for i in range(line_cnt)]

    shard = Shard(tmp_file)
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

    shard1 = Shard(tmp_file)
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

    shard2 = Shard(tmp_file, index=5)
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

    shard = Shard(tmp_file, cols=cols)
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
        shard = Shard(Path("input.tsv"), cols=["header"], index=i)

        # The first shard should contains the header if the `cols` is passed
        if i == 0:
            assert shard.contains_header()
        else:
            assert not shard.contains_header()


def test_make_one_shard(tmp_path: Path):
    with open(tmp_path / "tmp.tsv", "w") as o:
        o.write("header1;heade2;header3\n")
        o.write("1;2;3\n")

    shards = sh.make_one_shard(tmp_path / "tmp.tsv", header=True, sep=";")
    assert len(shards) == 1
    with shards[0] as progress:
        next(iter(progress))
        col = progress.value("header3")
        assert int(col) == 3


def test_one_shard_with_nrows(tmp_path: Path):
    with open(tmp_path / "tmp.tsv", "w") as o:
        o.write("header1|heade2|header3\n")
        [o.write("col1|col2|col3\n") for _ in range(10)]

    shards = sh.make_one_shard(tmp_path / "tmp.tsv", header=True, sep="|", nrows=5)
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
    shards = sh.make_shards(input_files, nshards=50)
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
    shards = sh.make_shards(
        tmp_file, nshards=5, header=header, sep="|", cache_dir=out_dir
    )

    # Change filename to reflect fairseq internal decompression
    if zip:
        tmp_file = out_dir / (tmp_file.name + "_expanded.txt")

    if header:
        cols = ["header1", "header2"]
        col_name = "header2"
        expected_shards = [
            ChunkShard(tmp_file, cols=cols, sep="|", index=0, start=0, end=60),
            ChunkShard(tmp_file, cols=cols, sep="|", index=1, start=60, end=126),
            ChunkShard(tmp_file, cols=cols, sep="|", index=2, start=126, end=178),
            ChunkShard(tmp_file, cols=cols, sep="|", index=3, start=178, end=243),
            ChunkShard(tmp_file, cols=cols, sep="|", index=4, start=243, end=295),
        ]
    else:
        col_name = 1  # type: ignore
        expected_shards = [
            ChunkShard(tmp_file, sep="|", index=0, start=0, end=66),
            ChunkShard(tmp_file, sep="|", index=1, start=66, end=123),
            ChunkShard(tmp_file, sep="|", index=2, start=123, end=175),
            ChunkShard(tmp_file, sep="|", index=3, start=175, end=227),
            ChunkShard(tmp_file, sep="|", index=4, start=227, end=279),
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
    shards = sh.make_shards(tmp_file, nshards=5, header=header, sep="|", nrows=23)

    if zip:
        tmp_filename = tmp_file.name + "_expanded.txt"
    else:
        tmp_filename = tmp_file.name
    real_input = tmp_file.parent / tmp_filename
    if header:
        cols = ["header1", "header2"]
        col_name = "header2"
        expected_shards = [
            ChunkShard(real_input, cols=cols, sep="|", index=0, start=0, end=48),
            ChunkShard(real_input, cols=cols, sep="|", index=1, start=48, end=88),
            ChunkShard(real_input, cols=cols, sep="|", index=2, start=88, end=136),
            ChunkShard(real_input, cols=cols, sep="|", index=3, start=136, end=176),
            ChunkShard(real_input, cols=cols, sep="|", index=4, start=176, end=216),
        ]
    else:
        col_name = 1  # type: ignore[assignment]
        expected_shards = [
            ChunkShard(real_input, sep="|", index=0, start=0, end=40),
            ChunkShard(real_input, sep="|", index=1, start=40, end=80),
            ChunkShard(real_input, sep="|", index=2, start=80, end=130),
            ChunkShard(real_input, sep="|", index=3, start=130, end=170),
            ChunkShard(real_input, sep="|", index=4, start=170, end=210),
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
    shards = sh.make_shards(
        tmp_file, nshards=nshards, algo="robin", header=header, sep=sep
    )
    if header:
        cols = ["header1", "header2"]
        col_name = "header2"
        expected_shards = [
            RoundRobinShard(tmp_file, cols=cols, sep=sep, index=i, nshards=nshards)
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
            RoundRobinShard(tmp_file, index=i, nshards=nshards) for i in range(nshards)
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
    shards = sh.make_shards(
        tmp_file, nshards=5, algo="sort", header=header, sep="\t", col=col
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
            sorted_file,
            sep="\t",
            cols=["header1", "header2"],
            index=0,
            start=0,
            end=121,
        )
    else:
        assert shards[0] == ChunkShard(sorted_file, sep="\t", index=0, start=0, end=126)


def test_make_shards_from_glob(tmp_path: Path):
    (tmp_path / "file1.tsv").touch()
    (tmp_path / "file2.tsv").touch()

    shards = list(sh.make_shards(tmp_path / "file*.tsv", cache_dir=Path("fakedir")))
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
