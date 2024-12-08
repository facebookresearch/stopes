# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import functools
import os
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
from datasets import Dataset, DownloadMode, concatenate_datasets, load_dataset

from stopes.utils.sharding.abstract_shards import (
    BatchFormat,
    BatchType,
    InputShardingConfig,
    Shard,
    arrow_table_to_batch,
)


@dataclass
class HFShard(Shard):
    """
    A wrapper over HuggingFace datatsets's Dataset to make it
    compatible with stopes Shard and Mapper API.

    Args:
        path_or_name (str or Path): Path to a local dataset, or name of the dataset
            in HuggingFace Hubg
        data_dir: HuggingFace-specific data dir (kind of subset of the dataset)
        split: (str) Split of the data. If None, all splits will be downloaded. Can
            accept "train", "test", "validation", or a HF split syntax (see
            https://huggingface.co/docs/datasets/v1.11.0/splits.html)
        use_cache (bool): If we should reuse the cached dataset, or download from
            HF Hub. This param has no impact if `path_or_name` is a local directory.
            Default True
        index (int): Index of the shard
        num_shards: Total number of shards in which the current one is one member
    """

    path_or_name: tp.Union[str, Path]
    data_dir: tp.Optional[str] = None
    split: tp.Optional[str] = None
    cached: bool = True
    index: tp.Optional[int] = None
    num_shards: int = 1
    trust_remote_code: bool = False

    def __post_init__(self):
        if self.filter:
            raise NotImplementedError(
                f"Arrow-syntax filter is not supported in HF shard. Get {self.filter}"
            )
        if self.index is None:
            assert self.num_shards == 1, f"Unknown shard index {self.index}"
        else:
            assert (
                self.index < self.num_shards
            ), f"Cannot make shard {self.index} from {self.num_shards} shards"

        self._data: tp.Optional[Dataset] = None

        # We could only iterate the underlying dataset in "nornmal" mode (via __iter__) or
        # in "converted" mode (via to_batches())
        # mode = 0 --> not started
        # mode = 1 --> _data is being consumed via __iter__()
        # mode = 2 --> _data is being consumed via to_batches()
        self._mode = 0

    def __enter__(self):
        """
        Create the underlying dataset, without loading them into main memory

        Note: When we enter the first shard, if `path_or_name` is not a local directory,
        the underlying dataset will be downloaded to a _central_ local data dir that is
        shared by all workers. This local data can be customized via os environment
        STOPES_HF_CACHE
        """

        _download_mode = (
            DownloadMode.REUSE_DATASET_IF_EXISTS
            if self.cached
            else DownloadMode.FORCE_REDOWNLOAD
        )

        _cache_dir = None
        if not Path(self.path_or_name).is_dir() and self.cached:
            # TODO: We trust that HF DownloadManager will perform proper locks to avoid
            # concurrent downloads from multiple workers. If error occurs, consider using
            # stopes FileLock explicity
            _cache_dir = os.getenv(
                "STOPES_HF_CACHE", Path.home() / ".cache" / "huggingface" / "datasets"
            )

        _data = load_dataset(
            path=self.path_or_name,
            data_dir=self.data_dir,
            cache_dir=_cache_dir,
            download_mode=_download_mode,
            split=self.split,
            trust_remote_code=self.trust_remote_code,
        )
        if self.split is None:  # _data is a DatasetDict, convert to Dataset
            _data = concatenate_datasets(
                [_data["train"], _data["test"], _data["validation"]]
            )

        if self.num_shards > 0:
            _data = _data.shard(num_shards=self.num_shards, index=self.index)
        self._data = _data
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._mode = 0
        self._data = None

    def __iter__(self) -> tp.Iterator[tp.Union[tp.Dict, tp.List]]:
        if self._data is None:
            raise ValueError("shard is not entered yet")
        assert self._mode != 2, "Consumption mode changed during iterating"
        if self._mode == 0:
            self._mode = 1
        yield from self._data
        self._mode = 0

    def to_batches(
        self,
        batch_size: tp.Optional[int],
        columns: tp.Optional[tp.List[str]] = None,
        batch_format: BatchFormat = BatchFormat.ARROW,
    ) -> tp.Iterator[BatchType]:
        if self._data is None:
            raise ValueError("shard is not entered yet")
        assert self._mode != 1, "Consumption mode changed during iterating"
        if self._mode == 0:
            self._mode = 2
        for item in self._data.iter(batch_size=batch_size):
            _table = pa.Table.from_pydict(item)
            _table = arrow_table_to_batch(_table, batch_format=batch_format)
            # TODO: Move the column projection in early step (before mini batch begins)
            if columns:
                _table = _table.select(columns)
            yield _table
        self._mode = 1


@dataclass
class HFInputConfig(InputShardingConfig):
    data_dir: tp.Optional[str] = None
    split: tp.Optional[str] = None
    cached: bool = True
    num_shards: int = 1
    trust_remote_code: bool = False

    def __post_init__(self):
        super().__post_init__()
        assert (
            len(self.skip_n_rows_per_shard) == 0
        ), "skipping not supported for this shard type"

    @functools.cached_property
    def input_dataset(self):
        return self.input_file

    def make_shards(self, **kwargs):
        assert isinstance(
            self.input_dataset, (Path, str)
        ), f"Expect input dataset to be Path or str, get {type(self.input_dataset)}"
        if self.filters_expr:
            raise NotImplementedError("Not implemented yet for HF Shards")

        return [
            HFShard(
                filter=None,
                path_or_name=self.input_dataset,
                data_dir=self.data_dir,
                split=self.split,
                cached=self.cached,
                index=i,
                num_shards=self.num_shards,
                trust_remote_code=self.trust_remote_code,
            )
            for i in range(min(self.num_shards, self.take or self.num_shards))
        ]
