# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import functools
import importlib.util
import logging
import typing as tp
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import cloudpickle
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import xxhash
from pyarrow.dataset import get_partition_keys
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from stopes.core.utils import batch as batched
from stopes.utils.arrow_utils import add_metadata_to_table, hash_table_with_schema
from stopes.utils.sharding.abstract_shards import (
    BatchFormat,
    BatchType,
    InputShardingConfig,
    OutputDatasetConfig,
    PartitionedDataMapperState,
    Shard,
    ShardWithSkip,
    arrow_table_to_batch,
    batch_length,
    batch_to_table,
)

logger = logging.getLogger("stopes.launcher")


import signal
from functools import wraps


class TimeoutException(Exception):
    """Exception to raise on a timeout"""

    ...


def timeout(seconds=60, error_message="Function call timed out"):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutException(error_message)

        @wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)  # Cancel the alarm
            return result

        return wrapper

    return decorator


def get_filesystem_from_path(
    uri: Union[Union[str, Path], Sequence[Union[str, Path]]],
    **kwargs,
) -> Tuple[Union[Union[str, Path], Sequence[Union[str, Path]]], Any]:
    return uri, None


@dataclass
class ParquetShardBase(ShardWithSkip, ABC):
    @abstractmethod
    def mini_batches(
        self, max_chunk_size: int, columns: tp.Optional[tp.List[str]] = None
    ) -> tp.Iterator[pa.Table]:
        """
        Returns an iterator over the mini-batches of this shard. Mini batches
        might be aggregated to form batches of the requested size if they are too
        small.

        Args:
            max_chunk_size (int): Maximum size of the mini-batches. Mini batches might
            be smaller.
        """

    def __iter__(self):
        raise NotImplementedError(
            "single-item iteration not yet implemented in parquetShard. "
            "Use to_batches() with batch_size = 1 instead"
        )

    def to_batches(
        self,
        batch_size: tp.Optional[int],
        columns: tp.Optional[tp.List[str]] = None,
        batch_format: BatchFormat = BatchFormat.PANDAS,
    ) -> tp.Iterator[BatchType]:
        assert batch_size, "you need to specify a batch_size."

        table: pa.Table = None

        for new_table in self.mini_batches(batch_size, columns):
            if self.filter is not None:
                new_table = new_table.filter(self.filter)
            # Note that the filters can reduce the number of rows,
            # so we combine the results from several batches filtered mini-batches
            if len(new_table) > 0:
                if table is not None:
                    table = pa.concat_tables([table, new_table])
                else:
                    table = new_table

                if len(table) >= batch_size:
                    table_to_return = table.slice(0, batch_size).combine_chunks()
                    yield arrow_table_to_batch(table_to_return, batch_format)

                    if len(table) == batch_size:
                        table = None
                    else:
                        table = table.slice(batch_size, len(table) - batch_size)

        # if we have a table left, it means that the last batch was smaller than `batch_size`
        # yield the rest
        if table is not None and len(table) > 0:
            yield arrow_table_to_batch(table, batch_format)


@dataclass
class ParquetShard(ParquetShardBase):
    fragment: List[pa.dataset.Fragment]

    def __post_init__(self) -> None:
        self._first_fragment: pa.dataset.Fragment = self.fragment[0]

    def __enter__(self) -> "ParquetShard":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    @property
    def nb_rows(self) -> int:
        return sum(frag.metadata.num_rows for frag in self.fragment)

    @functools.cached_property
    def partition_columns(self):
        return list(
            get_partition_keys(self._first_fragment.partition_expression).keys()
        )

    @functools.cached_property
    def columns(self) -> tp.List[str]:
        return list(self._first_fragment.physical_schema.names) + self.partition_columns

    def fragment_columns(
        self, columns: tp.Optional[tp.List[str]] = None
    ) -> tp.Optional[tp.List[str]]:
        if columns is None:
            return None

        assert set(columns).issubset(set(self.columns)), (
            sorted(set(columns) - set(self.columns)),
            self.columns,
        )
        return sorted(set(columns) - set(self.partition_columns))

    def _add_partitioning_columns(
        self,
        frag: pa.dataset.Fragment,
        table: pa.Table,
        columns: tp.Optional[tp.List[str]] = None,
    ) -> pa.Table:
        """
        When loading a single fragment, pyarrow does not add the partitioning columns,
        so we need to do it manually.
        """
        for key, val in get_partition_keys(frag.partition_expression).items():
            if columns is None or key in columns:
                values = pa.DictionaryArray.from_arrays(
                    np.zeros(len(table), dtype=np.int32), [val]
                )
                table = table.append_column(key, values)

        return table

    def mini_batches(
        self, max_chunk_size: int, columns: tp.Optional[tp.List[str]] = None
    ) -> tp.Iterator[pa.Table]:
        frag_columns = self.fragment_columns(columns)
        for frag in self.fragment:
            for record in frag.to_batches(
                batch_size=max_chunk_size, columns=frag_columns
            ):
                table = pa.Table.from_batches([record])
                table = self._add_partitioning_columns(frag, table, columns)
                yield table
                del table

    def to_batches(
        self,
        batch_size: tp.Optional[int],
        columns: tp.Optional[tp.List[str]] = None,
        batch_format: BatchFormat = BatchFormat.PANDAS,
    ) -> tp.Iterator[BatchType]:
        # TODO : if filter does not contain partitioned columns we can propagate to fragment.to_table(...)
        yield from super().to_batches(batch_size or self.nb_rows, columns, batch_format)


FragmentSorting = Enum(
    "FragmentSorting", ["DEFAULT", "RANDOM", "INCREASING_LENGTH", "DESCREASING_LENGTH"]
)


@functools.lru_cache(maxsize=10)
def get_dataset(input_dataset, filters_expr, filesystem) -> pq.ParquetDataset:
    filter_: tp.Optional[pa.compute.Expression] = (
        eval(filters_expr, {}, {"pa": pa, "ds": ds, "pc": pc}) if filters_expr else None
    )

    return pq.ParquetDataset(
        (input_dataset[0] if len(input_dataset) == 1 else list(input_dataset)),
        filters=filter_,
        filesystem=filesystem,
    )


@dataclass
class ParquetShardingConfig(InputShardingConfig):
    """
    Config defining how one parquet shard outputs its content.

    Extra args:
    - split_row_groups: whether to split the parquet fragments further by the row group
    - sorting_strategy: fragment sorting strategy
    - filesystem_expr: filesystem expression, permissible values:
      * None -> to guess the filesystem from input_file ("s3://bucket/key") format (recommended)
      * `s3fs` (using s3fs.core.S3FileSystem)
      * `pyarrow_s3fs` (using pyarrow.fs.S3FileSystemt)
      * Evaluable Python code (e.g `fs.S3FileSystem(region="us-west-2", role_arn=...)`)


    Note that for manually provided `filesystem_expr`, one should use "bucket/key" as input file (without "s3://") !!
    """

    split_row_groups: bool = False
    fragment_group_size: int = 1
    """
    This determines how many parquet fragments will be grouped to form a single shard.
    Defaults to 1
    """
    nb_samples_per_group: tp.Optional[int] = None
    """
    Allows to group several parquet fragments together so that the resulting shard will get ~ `nb_samples_per_group`.
    Only partition filters are taking into account.
    Defaults to None (not applied)
    """

    # aggregate_files: tp.Optional[tp.List[str]] = None
    sorting_strategy: FragmentSorting = FragmentSorting.DEFAULT
    filesystem_expr: tp.Optional[str] = None

    def validate(self) -> None:
        if self.nb_samples_per_group is not None:
            assert self.nb_samples_per_group > 0, "only positive values are accepted"
            assert (
                self.fragment_group_size == 1
            ), "cannot use `fragment_group_size` with `group_to_nb_samples`"
        _ = self.input_dataset  # to init input dataset

    @functools.cached_property
    def input_dataset(self) -> tp.Sequence[Path]:
        if not isinstance(self.input_file, (list, tuple)):
            # we expect single Path here
            self.input_file = [str(self.input_file)]

        self.input_file, self.filesystem = get_filesystem_from_path(
            self.input_file, filter=self.filesystem_expr
        )
        self.input_file = tuple(self.input_file)
        if self.filesystem is not None:
            if hasattr(self.filesystem, "glob"):
                return tuple(
                    [
                        y
                        for f in self.input_file
                        for y in sorted(self.filesystem.glob(str(f)))
                    ]
                )
            else:
                return tuple(sorted(self.input_file))
        else:
            return tuple(super().input_dataset)

    @functools.cached_property
    def partition_columns(self) -> tp.List[str]:
        dataset = get_dataset(self.input_dataset, self.filters_expr, self.filesystem)
        partitioning = dataset.partitioning
        if partitioning is None:
            return []
        return [
            name
            for name, dd in zip(partitioning.schema.names, partitioning.dictionaries)
            if dd is not None
        ]

    def make_shards(self, **kwargs) -> tp.List[Shard]:
        dataset = get_dataset(self.input_dataset, self.filters_expr, self.filesystem)
        fragments = list(dataset._dataset.get_fragments(self.filter))[: self.take]

        # TODO: making thread parallel when using S3 datasets
        if self.split_row_groups:
            fragments = [
                y for fragment in fragments for y in fragment.split_by_row_group()
            ][: self.take]

        logger.info(f"Finding {len(fragments)} fragments")

        sorting_strategy = self.sorting_strategy
        if sorting_strategy == FragmentSorting.RANDOM:
            fragments = list(np.random.RandomState(None).permutation(fragments))

        if self.nb_samples_per_group or sorting_strategy in [
            FragmentSorting.DESCREASING_LENGTH,
            FragmentSorting.INCREASING_LENGTH,
        ]:
            nb_rows_per_fragment_ = []
            logger.info("Computing fragments rows count!")

            with logging_redirect_tqdm():
                for frag in tqdm(fragments):
                    nb_rows_per_fragment_.append(frag.count_rows())
            nb_rows_per_fragment = np.array(nb_rows_per_fragment_)

        if sorting_strategy in [
            FragmentSorting.DESCREASING_LENGTH,
            FragmentSorting.INCREASING_LENGTH,
        ]:
            if sorting_strategy == FragmentSorting.DESCREASING_LENGTH:
                permutation = np.argsort(-nb_rows_per_fragment, kind="stable")
            else:
                permutation = np.argsort(nb_rows_per_fragment, kind="stable")
            nb_rows_per_fragment = nb_rows_per_fragment[permutation]
            fragments = [fragments[i] for i in permutation]

        if self.nb_samples_per_group:
            shards_list: tp.List[tp.List[pa.dataset.Fragment]] = []
            current_nb_samples = 0
            current_list = []
            for size, frag in zip(
                nb_rows_per_fragment[: self.take], fragments[: self.take]
            ):
                current_list.append(frag)
                current_nb_samples += size
                if current_nb_samples >= self.nb_samples_per_group:
                    shards_list.append(current_list)
                    current_list = []
                    current_nb_samples = 0

            if current_list:  # remainder
                shards_list.append(current_list)
            return [
                ParquetShard(
                    fragment=frags,
                    filter=self.filter,
                    skip_n_rows=self.skip_n_rows_per_shard.get(i, 0),
                )
                for i, frags in enumerate(shards_list)
            ]

        return [
            ParquetShard(
                fragment=list(frags),
                filter=self.filter,
                skip_n_rows=self.skip_n_rows_per_shard.get(i, 0),
            )
            for i, frags in enumerate(
                batched(fragments[: self.take], self.fragment_group_size)
            )
        ]


@dataclass
class ParquetOutputConfig(OutputDatasetConfig):
    """
    For s3 files, there're two possible options :
    * dataset_path = "s3://bucket/key/" and filesystem_expr = None  (automatically getting client)
    * dataset_path = "bucket/key/" and filesystem_expr = "s3fs"

    """

    row_group_size: tp.Optional[int] = None
    max_rows_per_file: tp.Optional[int] = None
    keep_same_partitioning: bool = True
    partition_columns: tp.Optional[tp.List[str]] = None
    filesystem_expr: tp.Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if self.keep_same_partitioning and self.partition_columns is not None:
            raise ValueError(
                "cannot provide `partition_cols` when `keep_same_partining` is True"
            )
        if self.compression == "default":
            self.compression = "snappy"
        assert self.compression in [
            None,
            "none",
            "snappy",
            "gzip",
            "brotli",
            "lz4",
            "zstd",
        ]
        self.dataset_path, self.filesystem = get_filesystem_from_path(  # type: ignore
            str(self.dataset_path),
            filter=self.filesystem_expr,
        )
        if self.filesystem is None:
            Path(self.dataset_path).mkdir(parents=True, exist_ok=True)
        else:
            try:
                self.filesystem.create_dir(self.dataset_path, recursive=True)
            except Exception:  # noqa
                try:
                    self.filesystem.mkdir(self.dataset_path, create_parents=True)
                except Exception:
                    pass

    def write_batch(
        self,
        batch: tp.Optional[BatchType],
        iteration_index: tp.Sequence[int],
        metadata: tp.Optional[tp.Dict[str, tp.Any]] = None,
        state_checkpoint: tp.Optional[PartitionedDataMapperState] = None,
    ) -> tp.List[Path]:
        if batch is None or batch_length(batch) == 0:
            # TODO: logger empty batch
            return []
        table = batch_to_table(batch)
        partition_cols: tp.Optional[tp.List[str]] = self.partition_columns

        try:
            guid = hash_table_with_schema(table)[:20]
        except Exception as e:
            logger.warn(f"`hash_table_with_schema` failed : {e}")
            guid = f"{uuid.uuid4()}"[:20]

        basename_template = "{i}_" + f"{guid}"
        iteration_index = (
            (iteration_index,) if isinstance(iteration_index, int) else iteration_index
        )
        for idx in iteration_index:
            basename_template += f"_{idx}"
        basename_template += ".parquet"
        # Thus final template will look as follows:
        # `{file_number}_{guid}_{batch_number}_{shard_number}.parquet`

        if metadata is not None:
            try:
                table = add_metadata_to_table(table, metadata)
            except Exception as e:
                logger.warn(f"`add_metadata_to_table` failed : {e}")

        written_files = []

        def collect_files(f: pa._dataset.WrittenFile):
            written_files.append(Path(f.path))

        pq.write_to_dataset(
            table,
            self.dataset_path,
            partition_cols=partition_cols,
            max_rows_per_file=self.max_rows_per_file,
            filesystem=self.filesystem,
            schema=self.expected_schema if self.validate_schema else None,
            basename_template=basename_template,
            use_threads=True,
            file_visitor=collect_files,
            **{
                "row_group_size": self.row_group_size or self.max_rows_per_file,
                "compression": self.compression,
            },
        )

        if state_checkpoint:
            with self._open(state_checkpoint.iteration_value, "wb") as f:
                cloudpickle.dump(state_checkpoint, f)

        return sorted(written_files)

    def reload_state(
        self,
        shard: Shard,
    ) -> tp.Optional[PartitionedDataMapperState]:
        try:
            with self._open(shard, "rb") as f:  # filename is wrong
                return cloudpickle.load(f)
        except:
            return None

    @contextmanager
    def _open(self, shard: Shard, mode: str = "r"):
        shard_hash = xxhash.xxh3_64_intdigest(cloudpickle.dumps(shard))
        fname = Path(self.dataset_path) / f".parquet_output.{shard_hash}.state"
        if self.filesystem is None:
            with fname.open(mode) as f:
                yield f
        else:
            with self.filesystem.open(str(fname), mode) as f:
                yield f
