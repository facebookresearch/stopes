# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import pyarrow as pa
from typing_extensions import Self

from stopes.core.utils import batch
from stopes.utils.file_chunker_utils import Chunker, find_offsets, find_offsets_of_lines
from stopes.utils.sharding.abstract_shards import (
    BatchFormat,
    BatchType,
    InputShardingConfig,
    Shard,
    arrow_table_to_batch,
)


@dataclass
class JSONShard(Shard):
    input_file: Union[str, Path]
    start_offset: int = 0
    end_offset: Optional[int] = None

    def __enter__(self) -> Self:
        self.file_handler = Chunker(
            str(self.input_file),
            start_offset=self.start_offset,
            end_offset=self.end_offset,
        )
        self.reader = self.file_handler.__enter__()  # type: ignore
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if hasattr(self, "file_handler"):
            self.file_handler.__exit__(exc_type, exc_val, exc_tb)
            del self.file_handler
        if hasattr(self, "reader"):
            self.reader = None
            del self.reader

    def _select_columns(
        self,
        line: str,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        data = json.loads(line)
        if columns is None:
            return data
        return {k: v for k, v in data.items() if k in columns}

    def __iter__(self):
        if self.reader is None:
            raise ValueError("shard is not entered yet")
        lines = iter(self.reader)
        mapper_func = partial(self._select_columns, columns=None)
        yield from map(mapper_func, lines)  # type: ignore

    def to_batches(
        self,
        batch_size: Optional[int],
        columns: Optional[List[str]] = None,
        batch_format: BatchFormat = BatchFormat.PANDAS,
    ) -> Iterator[BatchType]:
        mapper_func = partial(self._select_columns, columns=columns)

        with self as reading_context:
            lines = iter(reading_context.reader)  # type: ignore
            lines = map(mapper_func, lines)  # type: ignore
            if batch_size is None:
                # Read the whole file as a single batch
                batched = [list(lines)]
            else:
                assert batch_size > 0, f"Invalid batch size: {batch_size}"
                batched = batch(lines, batch_size=batch_size)  # type: ignore
            for _batch in batched:
                table = pa.Table.from_pylist(_batch)
                yield arrow_table_to_batch(table, batch_format)


@dataclass
class JSONShardConfig(InputShardingConfig):
    input_file: Union[str, Path]
    num_shards: int = 1
    nrows: Optional[int] = None
    partition_columns: Optional[List[str]] = None

    def __post_init__(self):
        super().__post_init__()
        assert self.num_shards > 0, f"invalid number of shards ({self.num_shards})"
        assert (
            len(self.skip_n_rows_per_shard) == 0
        ), "skipping not supported for this shard type"

    def validate(self) -> None:
        assert Path(self.input_file).exists()
        pass

    def make_shards(self, **kwargs) -> List[Shard]:
        if self.nrows:
            offsets = find_offsets_of_lines(
                self.input_file, self.num_shards, self.nrows
            )
        else:
            offsets = find_offsets(self.input_file, self.num_shards)
        return [
            JSONShard(
                filter=None,
                input_file=self.input_file,
                start_offset=start,
                end_offset=end,
            )
            for start, end in zip(offsets, offsets[1:])
        ]
