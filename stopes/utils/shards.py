# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
# Different methods to support sharding for audio files

import functools
import glob
import itertools
import logging
import typing as tp
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Self

from stopes.core.utils import expand_if_compressed
from stopes.core.utils import open as stopes_open
from stopes.core.utils import sort_file


@functools.lru_cache(10)
def warn_once(msg: str) -> None:
    """Prevents flooding stderr with the same repeated error message."""
    log.warning(msg)


log = logging.getLogger("stopes.speech.shards")


@dataclass
class Shard:
    """
    input to  one worker procesing a file shard for an array module. Default behaviour
    is that the worker will process an entire file.

    A shard is a contextmanager object: When you enter a shard in a local job, it gives
    access from the input file resource (by default via `stopes.core.utils.open()`).

    A shard is also an iterator: You lazily reads each line after entering the shard. It
    will update the internal states silently, to ensure the reading can be recovered if
    the job needs to be re-run.
    Note that this recovery is only guaranteed within one (slurm) job or machine, and not
    if the whole pipeline is re-run, because a Shard object - once created - will be sent
    and kept locally to each job only.

    Args:
        input_file (Path): The input file.
        cols (list or bool, optional): a list of header columns. None if there is no header
        sep (optional): the separator of lines. Only applicable when `cols` is not None
        index: index of the shard. None if there is only one shard for the file
    """

    input_file: tp.Union[str, Path]
    cols: tp.Optional[tp.List[str]] = None
    sep: tp.Optional[str] = None
    index: tp.Optional[int] = None

    def __post_init__(self):
        """Prepare internal properties"""

        # Keep how many lines already processed. Use to re-run the job
        self._lines_cnt: int = 0

        # handle the input resource
        self._input_handler: tp.Optional[tp.ContextManager] = None
        self._reader: tp.Optional[tp.Iterator[str]] = None

    def __enter__(self) -> Self:
        if not Path(self.input_file).exists():
            raise FileNotFoundError(self.input_file)
        self._reader = self.input_handler.__enter__()
        return self

    @property
    def input_handler(self) -> tp.ContextManager:
        if self._input_handler is None:
            self._input_handler = stopes_open(self.input_file)
        return self._input_handler

    def resolve_column_index(self, column_name: tp.Union[int, str]) -> int:
        if isinstance(column_name, int) or column_name.isdecimal():
            return int(column_name)
        assert (
            isinstance(self.cols, tp.List) and len(self.cols) > 0
        ), f"{self.input_file} has no header"
        try:
            return self.cols.index(column_name)
        except ValueError:
            raise ValueError(
                f"Column {column_name} not found in header of {self.input_file}: {self.cols}"
            )

    def value(self, column_name: tp.Union[int, str]) -> str:
        """Get value from a given column in the current line"""

        column_offset = self.resolve_column_index(column_name)
        lines = self.line.rstrip().split(self.sep)
        return lines[column_offset]

    def __iter__(self) -> tp.Iterator[str]:
        """start or resume the input file consumption from the last attempt."""
        lines = iter(self._reader)  # type: ignore
        if self.has_started():
            log.info(
                f"Resuming from previous attempt, already processed {self._lines_cnt} lines"
            )
        # Skip processed lines
        skipped_lines = int(self.contains_header()) + self._lines_cnt
        for line in itertools.islice(lines, skipped_lines, None):
            # Keep track of current line and processed lines so far
            self.line = line
            self._lines_cnt += 1
            yield line

    def has_started(self):
        """whether the shard is already (partially) processed"""
        return self._lines_cnt > 0

    def contains_header(self) -> bool:
        """whether the corresponding shard contains header"""
        return bool(self.cols) and not bool(self.index)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.input_handler.__exit__(exc_type, exc_val, exc_tb)
        self._input_handler = None
        self._reader = None


@dataclass
class TopNShard(Shard):
    """
    progress of one worker processing a file up to top-N lines
    """

    nrows: tp.Optional[int] = None

    def __iter__(self) -> tp.Iterator[str]:
        lines = super().__iter__()
        lines = itertools.islice(lines, 0, self.nrows)
        for line in lines:
            yield line


@dataclass
class ChunkShard(Shard):
    """
    A shard that corresponds to a file contiguous chunk.

    Args:
        start (int): start byte offset of the shard
        end (int): end byte offset of the shard. None if the shard is to be processed till EOF
    """

    from stopes.utils.file_chunker_utils import Chunker

    start: int = 0
    end: tp.Optional[int] = None

    @property
    def input_handler(self) -> tp.ContextManager:
        if self._input_handler is None:
            self._input_handler = self.Chunker(
                str(self.input_file), self.start, self.end
            )
        return self._input_handler


@dataclass
class RoundRobinShard(Shard):
    """
    A shard that corresponds to a subset of lines read from the file in the round robin fashion

    Args:
        nshards: Number of the shards
    """

    nshards: int = 1

    def __iter__(self) -> tp.Iterator[str]:
        if self.has_started():
            log.info(
                f"Resuming from previous attempt, already processed {self._lines_cnt} lines"
            )
        skipped_lines = int(self.contains_header()) + self._lines_cnt
        for i, line in enumerate(iter(self._reader)):  # type: ignore
            if i % self.nshards == self.index:
                if skipped_lines == 0:
                    self.line = line
                    self._lines_cnt += 1
                    yield line
                else:
                    skipped_lines -= 1


def resolve_output(
    shard: Shard, output_file: tp.Optional[Path] = None, suffix: str = ""
) -> tp.Optional[Path]:
    """
    A convenience function to help users make a standard output filename for the shard.
    Recommended if user wants a more consistent sharding output naming to be used in stopes pipeline

    The output suffix calibration logic:

    First, find the proper suffix (in order of priority):

    - If the input `suffix` is given, use it
    - If the `output_file` if a file with suffixes, use it
    - If the `output_file` is a directory, use input_file suffix.
    - If neither `output_file` nor `suffix` is given, use the input_file suffix

    In all cases, make sure the output is not compressed (even if input_file is compressed)
    except the user explicitly wants it, either via `output_file` or `suffix`

    After that, prepend shard index to the output suffix

    Example:

    - ouput_file = out.txt , suffix = ".tsv.gz" , no shard --> output = out.tsv.gz
    - ouput_file = out.txt , suffix = ".tsv.gz" , 2 shards --> outputs = out.0.tsv.gz , out.1.tsv.gz
    - ouput_file = out.txt , suffix = "" , no shard --> output = out.txt
    - ouput_file = out.tsv.gz , suffix = "" , 2 shards --> outputs = out.0.tsv.gz , out.1.tsv.gz
    - ouput_file = out_dir , suffix = ".tsv.gz" , no shard --> output = out_dir / input.tsv.gz
    - output_file = None, suffix = "", input = in.tsv.gz", no shard --> output = in.tsv
    - output_file = None, suffix = "", input = in.tsv.gz", 2 shards --> output = in.0.tsv, in.1.tsv
    - output_file = file_without_ext, suffix = "" , input = in.tsv.gz, 2 shards -> ouput = file_without_ext.0, file_without_ext.1

    """
    # stoptes.utils.file_chunker_utils adds "_expanded.txt" to a compressed file
    input_name = Path(shard.input_file).name.replace("_expanded.txt", "")

    # an intermediate file from stopes.utils.sort_files has a ".merge_sort" suffix
    input_name = input_name.replace(".merge_sort", "")

    # Unless user specifies suffix with .gz or .xz, we do not compress output
    input_name = input_name.replace(".gz", "").replace(".xz", "")

    in_suffix = Path(input_name).suffix  # .tsv or .txt
    input_stem = Path(input_name).stem

    if suffix:
        out_suffix = suffix
    elif output_file is None or output_file.is_dir():
        out_suffix = in_suffix
    else:
        out_suffix = "".join(output_file.suffixes)

    # If there are more than one shard for the file, add shard index to each output name
    if shard.index is not None:
        out_suffix = f".{shard.index}{out_suffix}"

    if output_file is None:
        resolved_output = (Path(shard.input_file).parent / input_stem).with_suffix(
            out_suffix
        )
    elif output_file.is_dir():
        resolved_output = (output_file / input_stem).with_suffix(out_suffix)
    elif len(output_file.suffixes) == 0:
        resolved_output = output_file.with_suffix(out_suffix)
    else:
        # file.ext1.ext2.ext3 --> file
        output_stem = output_file.name[: -len("".join(output_file.suffixes))]
        resolved_output = output_file.parent / (output_stem + out_suffix)

    # Happens when suffix = "" and output_file = None and input_file is not compressed
    if resolved_output.resolve() == Path(shard.input_file).resolve():
        log.warning(
            f"Output file is the same as input file ({shard.input_file}). Writing is disabled"
        )
        return None
    return resolved_output.resolve()


def parse_header(input_file: Path, sep: str):
    with stopes_open(input_file) as reader:
        return next(reader).rstrip("\n").split(sep)


def find_offsets_of_lines(filename: str, num_chunks: int, nrows: int) -> tp.List[int]:
    """
    Find the offsets of a text file that makes a total of `num_chunks` roughly equal-size chunks.
    Here only the first `nrows` lines are read. This function should be used when `nrows` is
    relatively small compared to the size of `filename`.
    To find offsets of the entire file, please use `stopes.utils.file_chunker_utils.find_offsets()`
    """
    offsets = []
    r = nrows % num_chunks
    chunk_size = nrows // num_chunks
    with open(filename, "r", encoding="utf-8") as f:

        # Each of the r first chunks has one more line than the rest num_chunks - r
        size = chunk_size + 1
        for _ in range(r):
            offsets.append(f.tell())
            [f.readline() for _ in range(size)]

        for _ in range(0, num_chunks - r):
            offsets.append(f.tell())
            [f.readline() for _ in range(chunk_size)]

        offsets.append(f.tell())

    return offsets


def make_one_shard(
    input_file: Path,
    header: bool = False,
    sep: tp.Optional[str] = None,
    nrows: tp.Optional[int] = None,
) -> tp.Sequence[Shard]:
    if header:
        assert (
            sep is not None
        ), "Please provide separator input. Sharder will not guess the file format cannot be guessed at this point"
        cols = parse_header(input_file, sep=sep)
    else:
        cols = None
    if nrows:
        return [TopNShard(input_file, cols, sep, 0, nrows)]
    else:
        return [Shard(input_file, cols, sep, 0)]


def make_chunk_shards(
    input_file: Path,
    nshards: int = 1,
    header: bool = False,
    sep: tp.Optional[str] = None,
    cache_dir: tp.Optional[Path] = None,
    nrows: tp.Optional[int] = None,
) -> tp.Sequence[ChunkShard]:
    """
    Create multiple shards from a single file, where each share corresponds to a continuous
    chunk of lines in the file. If the file is compressed, it will be decompressed into
    the `cache_dir` under the new name: `input_file_expanded.txt`
    """

    from stopes.utils.file_chunker_utils import find_offsets

    if input_file.suffix in {".gz", ".xz"}:
        warn_once(
            "Multiple shards for compressed file is asked. Chunking the compressed file results "
            "in a slow scheduler warm-up. Please give the decompressed file if possible."
        )
        _cache = cache_dir if cache_dir else input_file.parent
        assert Path(
            _cache
        ).exists(), (
            f"cache directory {_cache} not found, cannot write intermediate files"
        )
        input_file = expand_if_compressed(input_file, _cache)  # type: ignore

    if nrows:
        offsets = find_offsets_of_lines(str(input_file), nshards, nrows)
    else:
        offsets = find_offsets(str(input_file), nshards)

    # Convert [pos1, pos2, pos3,...] to [(pos1, pos2), (pos2, pos3),..]
    file_chunks = zip(offsets, offsets[1:])
    if header:
        assert (
            sep is not None
        ), "Please provide separator input. Sharder will not guess the file format cannot be guessed at this point"
        cols = parse_header(input_file, sep=sep)
    else:
        cols = None
    return [
        ChunkShard(input_file, cols, sep, i, start, end)
        for i, (start, end) in enumerate(file_chunks)
    ]


def make_roundrobin_shards(
    input_file: Path,
    nshards: int = 1,
    header: bool = False,
    sep: tp.Optional[str] = None,
) -> tp.Sequence[RoundRobinShard]:
    """
    Make multiple shards from a single file where each shard correspond to all lines in the file
    with certain index. For example, if there are 8 shards, shard_0 corresponds to lines 0, 8, 16,..
    """
    if header:
        assert (
            sep is not None
        ), "Please provide separator input. Sharder will not guess the file format cannot be guessed at this point"
        cols = parse_header(input_file, sep=sep)
    else:
        cols = None
    return [RoundRobinShard(input_file, cols, sep, i, nshards) for i in range(nshards)]


def make_sorted_shards(
    input_file: Path,
    nshards: int = 1,
    header: bool = False,
    sep: tp.Optional[str] = None,
    cache_dir: tp.Optional[Path] = None,
    col: tp.Optional[tp.Union[str, int]] = None,
    no_duplicate: bool = False,
) -> tp.Sequence[Shard]:
    """
    Create shards from one file by sorting the lines after values in column `col` and divide the
    sorted lines into different chunks. This algorithm requires input file to be uncompressed
    before. If `no_duplicate` is True, the shard will not have duplicates
    """

    from stopes.utils.file_chunker_utils import find_offsets

    if input_file.suffix in {".gz", ".xz"}:
        warn_once(
            "Multiple shards for compressed file is asked. Chunking the compressed file results "
            "in a slow scheduler warm-up. Please give the decompressed file if possible."
        )
        _cache = cache_dir if cache_dir else input_file.parent
        assert Path(
            _cache
        ).exists(), (
            f"cache directory {_cache} not found, cannot write intermediate files"
        )
        input_file = expand_if_compressed(input_file, _cache)  # type: ignore

    sorted_file = str(input_file) + ".merge_sort"
    sort_file(input_file, sorted_file, col=col, sep=sep, no_duplicate=no_duplicate)
    offsets = find_offsets(str(sorted_file), nshards)

    # Convert [pos1, pos2, pos3,...] to [(pos1, pos2), (pos2, pos3),..]
    file_chunks = zip(offsets, offsets[1:])
    if header:
        assert (
            sep is not None
        ), "Please provide separator input. Sharder will not guess the file format cannot be guessed at this point"
        cols = parse_header(input_file, sep=sep)
    else:
        cols = None
    return [
        ChunkShard(sorted_file, cols, sep, i, start, end)
        for i, (start, end) in enumerate(file_chunks)
    ]


def make_shards(
    input: tp.Union[str, tp.List, Path],
    nshards: int = 1,
    algo: str = "chunk",
    header: bool = False,
    sep: tp.Optional[str] = None,
    cache_dir: tp.Optional[Path] = None,
    **kwargs,
) -> tp.Sequence[Shard]:
    """
    Make shards from an `input`.

    Args:
        input (str, list or Path): Input to gnerate the shards. It could be:
        nshards: The number of shards to generate from the input. Only applicable if `input`
            is a single file.
        header (bool): Whether or not the input files have headers. Default False
        sep: separator for columns in the line (all files in `input` must have the same format and separator)
        cache_dir (str, Path, optional): directory to cache the intermedia files (such as uncompressed input file)


    """
    assert nshards > 0, f"invalid number of shards ({nshards})"
    if isinstance(input, tp.List):
        return [
            s
            for f in input
            for s in make_shards(f, 1, algo, header, sep, cache_dir, **kwargs)
        ]
    elif (p := Path(input)).is_dir():
        return [
            s
            for f in p.iterdir()
            for s in make_shards(f, 1, algo, header, sep, cache_dir, **kwargs)
        ]
    elif not p.is_file():
        return [
            s
            for f in glob.glob(str(input))
            for s in make_shards(f, 1, algo, header, sep, cache_dir, **kwargs)
        ]
    elif nshards == 1:
        return make_one_shard(p, header, sep)
    elif algo == "chunk":
        return make_chunk_shards(
            p, nshards, header, sep, cache_dir, kwargs.get("nrows")
        )
    elif algo == "robin":
        return make_roundrobin_shards(p, nshards, header, sep)
    elif algo == "sort":
        return make_sorted_shards(
            p,
            nshards=nshards,
            header=header,
            sep=sep,
            cache_dir=cache_dir,
            col=kwargs.get("col"),
            no_duplicate=bool(kwargs.get("no_duplicate")),
        )

    raise ValueError(
        f"invalid input: input={str(input)}, nshards={nshards}, algo={algo}"
    )
