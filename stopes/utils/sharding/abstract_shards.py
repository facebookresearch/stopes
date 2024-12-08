# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import dataclasses
import functools
import typing as tp
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum
from glob import iglob
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

BatchFormat = Enum("BatchFormat", ["PANDAS", "NUMPY", "ARROW"])
BatchType = tp.Union[pd.DataFrame, tp.Dict[str, np.ndarray], pa.Table]


def batch_length(batch: tp.Optional[BatchType]) -> int:
    if batch is None:
        return 0

    if isinstance(batch, dict):
        if len(batch) == 0:
            return 0
        return len(next(batch.values()))  # type: ignore

    return len(batch)


def batch_tail(batch: BatchType, nb_samples: int) -> BatchType:
    """
    take nb_samples from the tail of the batch
    """
    if isinstance(batch, pd.DataFrame):
        return batch.tail(nb_samples)
    elif isinstance(batch, dict):
        raise ValueError("batch_tail cannot be implemented for dict")
    elif isinstance(batch, pa.Table):
        return batch.slice(len(batch) - nb_samples)
    else:
        raise ValueError("data type is not understood :", type(batch))


def batch_to_table(batch: BatchType) -> pa.Table:
    if isinstance(batch, pd.DataFrame):
        return pa.Table.from_pandas(batch, preserve_index=False)
    elif isinstance(batch, dict):
        return pa.Table.from_pydict(batch)
    elif isinstance(batch, pa.Table):
        return batch.combine_chunks()
    else:
        raise ValueError("data type is not understood :", type(batch))


def batch_to_pandas(batch: BatchType) -> pd.DataFrame:
    if isinstance(batch, pd.DataFrame):
        return batch
    elif isinstance(batch, dict):
        return pd.DataFrame(batch)
    elif isinstance(batch, pa.Table):
        return batch.to_pandas()
    else:
        raise ValueError("data type is not understood :", type(batch))


def arrow_table_to_batch(table: pa.Table, batch_format: BatchFormat) -> BatchType:
    if batch_format == BatchFormat.ARROW:
        return table.combine_chunks()
    elif batch_format == BatchFormat.PANDAS:
        return table.to_pandas(split_blocks=True, self_destruct=True)
    elif batch_format == BatchFormat.NUMPY:
        return table.to_pydict()
    else:
        raise ValueError(f"Unknown batch format {batch_format}")


def concat_batches(list_of_batches: tp.List[BatchType]) -> tp.Optional[BatchType]:
    if len(list_of_batches) == 0:
        return None

    types_ = list(set(map(type, list_of_batches)))
    assert len(types_) == 1
    common_type = list_of_batches[0]

    if isinstance(common_type, pd.DataFrame):
        return pd.concat(list_of_batches, axis=0)
    elif isinstance(common_type, pa.Table):
        return pa.concat_tables(list_of_batches).combine_chunks()
    elif isinstance(common_type, dict):
        assert (
            len(set(tuple(bb.keys()) for bb in list_of_batches)) == 1
        ), "all batches should share the same keys"
        return {
            key: np.concatenate([batch[key] for batch in list_of_batches], axis=0)
            for key in list_of_batches[0].keys()
        }
    else:
        raise ValueError("data type is not understood", common_type)


@dataclass
class Shard(AbstractContextManager):
    """
    An abstract dataclass class holding configuration that represents a piece of tabular data.
    It also exposes methods to read this data (possibly in smaller batches) in various data formats.

    It optionally supports the reading only a part of the columns present in the data schema.
    It uses `filter` for row level filtering with the implementation based on `pyarrow.dataset.Expression`:
        - see https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Expression.html
        - example for "col1" and "col2"  (present in data schema):
        >>> import pyarrow.compute as pc
        >>> (pc.field("col1") < pc.scalar(3)) | (pc.field("col2") > 7)
    """

    filter: tp.Optional[pa.dataset.Expression]

    def __post_init__(self) -> None:
        ...

    @abstractmethod
    def to_batches(
        self,
        batch_size: tp.Optional[int],
        columns: tp.Optional[tp.List[str]] = None,
        batch_format: BatchFormat = BatchFormat.PANDAS,
    ) -> tp.Iterator[BatchType]:
        """
        Return a sequential mini-batch iterator of given `batch-size` over the underlying data.
        If `batch_size` is None, it will read the whole data as a single batch.

        Args:
            batch_size (tp.Optional[int]): batch size to use
            columns (tp.Optional[tp.List[str]], optional): columns to read. Defaults to None.
            batch_format (BatchFormat, optional): type of batch container. Defaults to BatchFormat.PANDAS.

        Returns:
            tp.Iterator[BatchType]:
        """
        ...

    @abstractmethod
    def __iter__(self) -> tp.Iterator[tp.Any]:
        ...


@dataclass
class ShardWithSkip(Shard, ABC):
    """
    A Shard, but with extra info to help skip rows at the beginning of the shard.
    This is useful when doing checkpointing or resuming failed jobs.
    """

    skip_n_rows: int

    def __post_init__(self) -> None:
        if not self.skip_n_rows:
            self.skip_n_rows = 0


@dataclass
class PartitionedDataMapperState:
    iteration_index: int
    iteration_value: Shard
    written_batches_index: int
    intermediate_index: int
    input_rows_written: int = 0


@dataclass
class InputShardingConfig(ABC):
    """
    Define how to handle the input data into different shards.

    Args:
        - input_file: input to the sharding (a file or data identifier). expected to be
            tp.Union[str, tp.List[tp.Union[str, Path]], Path]
        - filters_expr, str is python evaluated string corresponding to the row level filtering
            implemented as `pyarrow.dataset.Expression`:
            * see https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Expression.html
            * for instance,
                filters_expr= '(pc.field("col1") < 3) | (pc.field("col2") > 7)'
            will result in valid filter if "col1" and "col2" are numeric columns present in data schema.
        - batch_size (int): size of the batch within one shard
        - columns: List of column headers to construct the shards.
        - take: tp.Optional[int], if not None, can be used in subclasses to use only
            `take` number of shards in `make_shards(...)` method and ignoring others shards.
            This option can be used for debugging or sampling.
        - skip_n_rows_per_shard: tp.Optional[int], if not None, can be used to skip a number of rows at the beginning
        of this shard. Useful when doing checkpointing or resuming failed jobs. The shard will not skip automatically, this info is used in the mapper.
    """

    input_file: (
        tp.Any
    )  # we expect it to be tp.Union[str, tp.List[tp.Union[str, Path]], Path]
    batch_size: tp.Optional[int] = None
    columns: tp.Optional[tp.List[str]] = None
    batch_format: BatchFormat = BatchFormat.PANDAS
    filters_expr: tp.Optional[str] = None  # for passing e.g. through Hydra config
    take: tp.Optional[int] = None
    skip_n_rows_per_shard: tp.Dict[int, int] = dataclasses.field(
        default_factory=lambda: defaultdict(int)
    )

    def __post_init__(self):
        self.filter: tp.Optional[pa.compute.Expression] = (
            eval(self.filters_expr, {}, {"pa": pa, "ds": ds, "pc": pc})
            if self.filters_expr
            else None
        )
        self.validate()

    @functools.cached_property
    def input_dataset(self) -> tp.Sequence[Path]:
        """
        Parse the input_file and construct the input_dataset as a list.
        Default behaviour (can be overriden in concrete subclass of InputConfig):
        1) If the input_file is a str, assume this is glob pattern and construct
            input_dataset as a list of matching paths
        2) If input_file is a Path, make input_dataset to be a single-item out of it
        3) If input_file is a a list of string, treat each item as a glob pattern
            and repeat 1)
        """
        if isinstance(self.input_file, str):
            return sorted(map(Path, iglob(self.input_file)))
        elif isinstance(self.input_file, Path):
            return [self.input_file]
        elif isinstance(self.input_file, (list, tuple)):
            return [
                y for f in self.input_file for y in sorted(map(Path, iglob(str(f))))
            ]
        else:
            raise ValueError(
                f"Unsupported input_file format {self.input_file} of type {type(self.input_file)}"
            )

    def validate(self) -> None:
        """Method that validates the files existence/readability + other params compatibility (filters, columns, ...)"""
        pass

    @abstractmethod
    def make_shards(self, **kwargs) -> tp.List[Shard]:
        """
        returns a list shards corresponding to the given configuration

        """
        ...

    def head(
        self,
        nb_top: int = 5,
        columns: tp.Optional[tp.List[str]] = None,
        batch_format: tp.Optional[BatchFormat] = None,
    ) -> BatchType:
        shard = self.make_shards()[0]
        with shard:
            return next(
                shard.to_batches(
                    nb_top,
                    columns=columns or self.columns,
                    batch_format=batch_format or self.batch_format,
                )
            )


@dataclass
class OutputDatasetConfig(ABC):
    """
    Config defining how one shard outputs its content.

    Args:
    - dataset_path: str, a folder where the output dataset will be written
    - validate_schema: bool, if True, it makes sure that the all written batches follows the same schema
    - batch_size, optional int, default=None. If provided, `write_batch` should be called as soon as the size of processed batch > `write_each_nb_samples`
        This should allow to write intermediate results (without loosing them in case of errors/preemtions) and to free some memory.
    - compression: str, the format specific compression that is applied on output files
        * use `compression=None` to deactivate any compression
        * use `compression="default"` to overwrite this value in subclasses
            - for parquet, default compression is "snappy"
            - for text, default compression is None (deactivated)"""

    dataset_path: str
    validate_schema: bool = False  # it's not yet completely supported
    batch_size: tp.Optional[int] = None
    compression: tp.Optional[str] = "default"

    def __post_init__(self) -> None:
        self.expected_schema: tp.Optional[
            pa.Schema
        ] = None  # TODO: how to pass it through Hydra serialization ?

        if self.compression is not None:
            self.compression = self.compression.lower()

    @abstractmethod
    def write_batch(
        self,
        batch: BatchType,
        iteration_index: tp.Sequence[int],
        metadata: tp.Dict[str, tp.Any] = {},
        state_checkpoint: tp.Optional[PartitionedDataMapperState] = None,
    ) -> tp.List[Path]:
        ...

    @abstractmethod
    def reload_state(
        self,
        shard: Shard,
    ) -> tp.Optional[PartitionedDataMapperState]:
        ...
