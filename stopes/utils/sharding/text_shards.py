# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
# Different methods to support sharding for audio files

import functools
import io
import itertools
import logging
import shutil
import typing as tp
import uuid
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
import pyarrow as pa
import xxhash
from omegaconf.listconfig import ListConfig
from pyarrow import csv as csv_pa

from stopes.core.utils import expand_if_compressed
from stopes.core.utils import open as stopes_open
from stopes.core.utils import sort_file
from stopes.utils.arrow_utils import hash_table_with_schema
from stopes.utils.file_chunker_utils import find_offsets_of_lines
from stopes.utils.sharding.abstract_shards import (
    BatchFormat,
    BatchType,
    InputShardingConfig,
    OutputDatasetConfig,
    PartitionedDataMapperState,
    Shard,
    arrow_table_to_batch,
    batch_length,
    batch_to_pandas,
    batch_to_table,
)


@functools.lru_cache(10)
def warn_once(msg: str) -> None:
    """Prevents flooding stderr with the same repeated error message."""
    log.warning(msg)


log = logging.getLogger("stopes.speech.shards")


@dataclass
class TextShard(Shard):
    """
    input to one worker processing a file shard for an array module. Default behaviour
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
        columns (list or bool, optional): a list of header columns. None if there is no header
        sep (optional): the separator of lines. Only applicable when `cols` is not None
        index: index of the shard. None if there is only one shard for the file
        path_column : when not None, means the column's name (returned with to_batches())
            containing the file path from which the corresponding data is read.
            If None, no extra column is added.

    """

    input_file: tp.Union[str, Path]
    columns: tp.Optional[tp.List[str]] = None
    sep: tp.Optional[str] = None
    index: tp.Optional[int] = None
    path_column: tp.Optional[str] = None

    def __post_init__(self):
        """Prepare internal properties"""
        super().__post_init__()
        # Keep how many lines already processed. Use to re-run the job
        self._lines_cnt: int = 0

        # handle the input resource
        self._input_handler: tp.Optional[tp.ContextManager] = None
        self._reader: tp.Optional[tp.Iterator[str]] = None

    def __enter__(self) -> "TextShard":
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
            isinstance(self.columns, tp.List) and len(self.columns) > 0
        ), f"{self.input_file} has no header"
        try:
            return self.columns.index(column_name)
        except ValueError:
            raise ValueError(
                f"Column {column_name} not found in header of {self.input_file}: {self.columns}"
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
        return bool(self.columns) and not bool(self.index)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.input_handler.__exit__(exc_type, exc_val, exc_tb)
        self._input_handler = None
        self._reader = None

    def to_batches(
        self,
        batch_size: tp.Optional[int],
        columns: tp.Optional[tp.List[str]] = None,
        batch_format: BatchFormat = BatchFormat.PANDAS,
    ) -> tp.Iterator[BatchType]:
        assert batch_size is None or batch_size > 0

        if columns is not None and self.columns is not None:
            assert set(columns).issubset(set(self.columns))

        if columns is None:
            columns = self.columns

        read_options = csv_pa.ReadOptions(
            use_threads=True, column_names=self.columns, encoding="utf8"
        )
        parse_options = csv_pa.ParseOptions(delimiter=self.sep, ignore_empty_lines=True)
        convert_options = csv_pa.ConvertOptions(include_columns=columns)

        old_lines_cnt, self._lines_cnt = self._lines_cnt, 0
        with self as reading_context:
            stream = io.BytesIO("".join(reading_context).encode())
            table = csv_pa.read_csv(
                stream,
                read_options=read_options,
                parse_options=parse_options,
                convert_options=convert_options,
            )
            stream.close()
        self._lines_cnt = old_lines_cnt

        if self.path_column:
            table = table.append_column(
                self.path_column,
                pa.DictionaryArray.from_arrays(
                    np.zeros(len(table), dtype=np.int32), [str(self.input_file)]
                ),
            )

        if self.filter is not None:
            table = table.filter(self.filter)
        if len(table) > 0:
            if batch_size is None:
                yield arrow_table_to_batch(table, batch_format)
            else:
                for tt in table.to_batches(max_chunksize=batch_size):
                    min_table = pa.Table.from_batches([tt])
                    yield arrow_table_to_batch(min_table, batch_format)


@dataclass
class TextShardingConfig(InputShardingConfig):
    nb_shards: int = 1
    sharding_strategy: str = "chunk"
    header: tp.Any = True  # should be tp.Optional[tp.Union[bool, tp.List[str]]]
    sep: tp.Optional[str] = None
    cache_dir: tp.Optional[Path] = None
    path_column: tp.Optional[str] = None
    # TODO: restrict only on supported sharding strategies
    # TODO: split by given number of rows
    """
        - header: either bool (True meaning the presence of header in a file) or explicit list of resulting column names
        - path_column : when not None, means the column's name (returned with shard.to_batches())
                      containing the file path from which the corresponding data is read. If None, no extra column is added.
    """

    def __post_init__(self):
        super().__post_init__()
        assert self.nb_shards > 0, f"invalid number of shards ({self.nb_shards})"
        assert (
            len(self.skip_n_rows_per_shard) == 0
        ), "skipping not supported for this shard type"

    def validate(self) -> None:
        # TODO: verify that files exists and are readable with provided parameters
        pass

    def make_shards(self, **kwargs) -> tp.List[Shard]:
        shards: tp.List[Shard] = list(
            make_text_file_shards(
                input=(
                    self.input_dataset[0]
                    if len(self.input_dataset) == 1
                    else self.input_dataset
                ),
                nshards=self.nb_shards,
                algo=self.sharding_strategy,
                header=self.header,  # type: ignore
                sep=self.sep,
                cache_dir=self.cache_dir,
                filter=self.filter,
                **kwargs,
            )
        )
        shards = shards[: self.take]
        if self.path_column:
            for shard in shards:
                shard.path_column = self.path_column  # type: ignore
        return shards


@dataclass
class TextOutputConfig(OutputDatasetConfig):
    header: bool = True
    sep: str = "\t"
    storage_options: tp.Optional[tp.Dict[str, str]] = None
    quoting: tp.Optional[int] = None

    """
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        assert self.sep in [
            ",",
            "\t",
        ], f"only comma and tab are supported as separators, got {self.sep}"
        if self.compression == "default":
            self.compression = None
        if self.validate_schema:
            raise NotImplementedError("not supported yet for text files")

        assert self.compression in [
            None,
            "zip",
            "gzip",
            "bz2",
            "zstd",
            "xz",
            "tar",
        ], f"unsupported compression {self.compression}"
        Path(self.dataset_path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compression_to_extension(compression: tp.Optional[str]) -> str:
        if compression is None:
            return ""
        return f".{compression}"

    @staticmethod
    def separator_to_extension(sep: str) -> str:
        mapping = {",": ".csv", "\t": ".tsv"}
        return mapping.get(sep, ".txt")

    def write_batch(
        self,
        batch: BatchType,
        iteration_index: tp.Sequence[int],
        metadata: tp.Optional[tp.Dict[str, tp.Any]] = None,
        state_checkpoint: tp.Optional[PartitionedDataMapperState] = None,
    ) -> tp.List[Path]:
        if batch is None or batch_length(batch) == 0:
            # TODO: logger empty batch
            return []

        # TODO: reuse resolve_output logic here
        try:
            guid = hash_table_with_schema(batch_to_table(batch))[:20]
        except Exception as e:
            print(f"`hash_table_with_schema` failed : {e}")
            guid = f"{uuid.uuid4()}"[:20]

        file_name = f"{guid}"
        iteration_index = (
            (iteration_index,) if isinstance(iteration_index, int) else iteration_index
        )
        for idx in iteration_index:
            file_name += f"_{idx}"
        file_name += f"{self.separator_to_extension(self.sep)}{self.compression_to_extension(self.compression)}"

        path = Path(self.dataset_path).joinpath(file_name)

        df_pd: pd.DataFrame = batch_to_pandas(batch)

        df_pd.to_csv(
            path,
            sep=self.sep,
            header=self.header,
            quoting=self.quoting,
            compression=self.compression,
            storage_options=self.storage_options,
            index=False,
        )

        if state_checkpoint:
            shard_hash = xxhash.xxh3_64_intdigest(
                cloudpickle.dumps(state_checkpoint.iteration_value)
            )
            with (Path(self.dataset_path) / f".text_output.{shard_hash}.state").open(
                "wb"
            ) as f:  # filename is wrong
                cloudpickle.dump(state_checkpoint, f)

        # this could be interesing
        # https://arrow.apache.org/docs/python/csv.html#incremental-writing
        # it'll be about x3 - x4 faster for writing but we need to handle the compression and remote storage adhoc
        # TODO : Write metadata
        return [path]

    def reload_state(
        self,
        shard: Shard,
    ) -> tp.Optional[PartitionedDataMapperState]:
        try:
            shard_hash = xxhash.xxh3_64_intdigest(cloudpickle.dumps(shard))
            with (Path(self.dataset_path) / f".text_output.{shard_hash}.state").open(
                "rb"
            ) as f:  # filename is wrong
                return cloudpickle.load(f)
        except:
            return None


@dataclass
class TopNShard(TextShard):
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
class ChunkShard(TextShard):
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
class RoundRobinShard(TextShard):
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
    shard: TextShard, output_file: tp.Optional[Path] = None, suffix: str = ""
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


def parse_header(
    input_file: Path,
    header: tp.Optional[tp.Union[bool, tp.List[str]]],
    sep: tp.Optional[str],
) -> tp.Optional[tp.List[str]]:
    if header is False or header is None:
        return None

    assert (
        sep is not None
    ), "Please provide separator input. Sharder will not guess the file format cannot be guessed at this point"

    with stopes_open(input_file) as reader:
        parsed_cols = next(reader).rstrip("\n").split(sep)

    if header is True:
        return parsed_cols
    if isinstance(header, (list, tuple)):
        return list(header)


def make_one_text_shard(
    input_file: Path,
    header: tp.Optional[tp.Union[bool, tp.List[str]]] = False,
    sep: tp.Optional[str] = None,
    nrows: tp.Optional[int] = None,
    filter: tp.Optional[pa.dataset.Expression] = None,
) -> tp.Sequence[TextShard]:
    cols = parse_header(input_file, header, sep)
    if nrows is not None:
        return [
            TopNShard(
                input_file=input_file,
                columns=cols,
                index=0,
                sep=sep,
                nrows=nrows,
                filter=filter,
            )
        ]
    else:
        return [
            TextShard(
                input_file=input_file, columns=cols, index=0, sep=sep, filter=filter
            )
        ]


def make_chunk_text_shards(
    input_file: Path,
    nshards: int = 1,
    header: tp.Optional[tp.Union[bool, tp.List[str]]] = False,
    sep: tp.Optional[str] = None,
    cache_dir: tp.Optional[Path] = None,
    nrows: tp.Optional[int] = None,
    filter: tp.Optional[pa.dataset.Expression] = None,
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

    return [
        ChunkShard(
            input_file=input_file,
            columns=parse_header(input_file, header, sep),
            index=i,
            sep=sep,
            start=start,
            end=end,
            filter=filter,
        )
        for i, (start, end) in enumerate(file_chunks)
    ]


def make_roundrobin_text_shards(
    input_file: Path,
    nshards: int = 1,
    header: tp.Optional[tp.Union[bool, tp.List[str]]] = False,
    sep: tp.Optional[str] = None,
    filter: tp.Optional[pa.dataset.Expression] = None,
) -> tp.Sequence[RoundRobinShard]:
    """
    Make multiple shards from a single file where each shard correspond to all lines in the file
    with certain index. For example, if there are 8 shards, shard_0 corresponds to lines 0, 8, 16,..
    """
    return [
        RoundRobinShard(
            input_file=input_file,
            columns=parse_header(input_file, header, sep),
            index=i,
            sep=sep,
            nshards=nshards,
            filter=filter,
        )
        for i in range(nshards)
    ]


def make_sorted_text_shards(
    input_file: Path,
    nshards: int = 1,
    header: tp.Optional[tp.Union[bool, tp.List[str]]] = False,
    sep: tp.Optional[str] = None,
    cache_dir: tp.Optional[Path] = None,
    col: tp.Optional[tp.Union[str, int]] = None,
    no_duplicate: bool = False,
    filter: tp.Optional[pa.dataset.Expression] = None,
) -> tp.Sequence[TextShard]:
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
    return [
        ChunkShard(
            input_file=sorted_file,
            sep=sep,
            columns=parse_header(input_file, header, sep),
            index=i,
            start=start,
            end=end,
            filter=filter,
        )
        for i, (start, end) in enumerate(file_chunks)
    ]


def make_text_file_shards(
    input: tp.Union[str, tp.Sequence, Path],
    nshards: int = 1,
    algo: str = "chunk",
    header: tp.Optional[tp.Union[bool, tp.List[str]]] = False,
    sep: tp.Optional[str] = None,
    cache_dir: tp.Optional[Path] = None,
    filter: tp.Optional[pa.dataset.Expression] = None,
    **kwargs,
) -> tp.Sequence[TextShard]:
    """
    Make shards from an `input`.

    Args:
        input (str, list or Path): Input to gnerate the shards. It could be:
        nshards: The number of shards to generate from the input. Only applicable if `input`
            is a single file.
        header (bool): Whether or not the input files have headers. Default False
        sep: separator for columns in the line (all files in `input` must have the same format and separator)
        cache_dir (str, Path, optional): directory to cache the intermedia files (such as uncompressed input file)
        filter : to apply when batching  # TODO: apply them for simple row iterations for consistency

    """
    assert nshards > 0, f"invalid number of shards ({nshards})"
    if isinstance(input, tp.List) or isinstance(input, ListConfig):
        return [
            s
            for f in input
            for s in make_text_file_shards(
                input=f,
                nshards=1,
                algo=algo,
                header=header,
                sep=sep,
                cache_dir=cache_dir,
                filter=filter,
                **kwargs,
            )
        ]
    elif (p := Path(input)).is_dir():  # type: ignore
        return [
            s
            for f in p.iterdir()
            for s in make_text_file_shards(
                input=f,
                nshards=1,
                algo=algo,
                header=header,
                sep=sep,
                cache_dir=cache_dir,
                filter=filter,
                **kwargs,
            )
        ]
    elif not p.is_file():
        return [
            s
            for f in sorted(glob(str(input)))
            for s in make_text_file_shards(
                input=f,
                nshards=1,
                algo=algo,
                header=header,
                sep=sep,
                cache_dir=cache_dir,
                filter=filter,
                **kwargs,
            )
        ]
    elif nshards == 1:
        return make_one_text_shard(
            p,
            header,
            sep,
            filter=filter,
        )
    elif algo == "chunk":
        return make_chunk_text_shards(
            p,
            nshards,
            header,
            sep,
            cache_dir,
            kwargs.get("nrows"),
            filter=filter,
        )
    elif algo == "robin":
        return make_roundrobin_text_shards(
            p,
            nshards,
            header,
            sep,
            filter=filter,
        )
    elif algo == "sort":
        return make_sorted_text_shards(
            input_file=p,
            nshards=nshards,
            header=header,
            sep=sep,
            cache_dir=cache_dir,
            col=kwargs.get("col"),
            no_duplicate=bool(kwargs.get("no_duplicate")),
            filter=filter,
        )

    raise ValueError(
        f"invalid input: input={str(input)}, nshards={nshards}, algo={algo}"
    )


def merge_shards(shards: tp.List[tp.Union[Path]], outdir: Path, suffix: str = ""):
    """Merge the shard outputs in the order of the shard indices"""

    def get_name_no_ext(fname: str) -> str:
        if len(suffix) == 0:
            return fname
        return fname[: -len(suffix)]

    def get_shard_idx(fname: Path) -> int:
        fname_no_ext = get_name_no_ext(str(fname))
        shard_idx = fname_no_ext[fname_no_ext.rfind(".") + 1 :]
        return int(shard_idx)

    if len(shards) == 1:
        shutil.copy2(shards[0], outdir)
    else:
        fname = get_name_no_ext(shards[0].name)
        outputfile = str(outdir) + "/" + (fname[: fname.rfind(".")] + suffix)
        ordered_shards = sorted(shards, key=get_shard_idx)
        log.info(f"Writing {len(ordered_shards)} shard outputs to {outputfile}")

        with stopes_open(outputfile, "wt") as o:
            for shard_output in ordered_shards:
                try:
                    with stopes_open(shard_output) as so:
                        for line in so:
                            o.write(line)
                except Exception:
                    log.error(f"Error in processing {shard_output}")
