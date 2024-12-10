# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import typing as tp
from contextlib import contextmanager
from itertools import islice

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from joblib import Parallel, delayed
from pyarrow.dataset import get_partition_keys  # requires pyarrow >= 13


def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


@contextmanager
def pyarrow_cpu(nb_cpu: int):
    """
    Context manager to control threads number used by pyarrow

    Args:
        nb_cpu (int):
    """
    nb_cpu_old = pa.cpu_count()
    nb_io_cpu_old = pa.io_thread_count()
    pa.set_cpu_count(nb_cpu)
    pa.set_io_thread_count(nb_cpu)
    try:
        yield
    finally:
        pa.set_cpu_count(nb_cpu_old)
        pa.set_io_thread_count(nb_io_cpu_old)


class ParquetBasicDataLoader:
    """
    args:
        - dataset_path str: path to parquet dataset (possibly partitioned)
        - columns list[str]: list of columns to load (default=None will load all available columns)
        - filters : pa.dataset.Expression (https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Expression.html#pyarrow.dataset.Expression)
        - use_threads: bool, option in parquet reading (gives some tradeoff memory vs speed), default=True
        - to_pandas: bool, to transform to pandas DataFrame (default=True) otherwise to keep as pa.Table (more memory efficient)
        - split_to_row_groups: bool, option to split individual parquet files to row groups level (default=True)

     Example of usage :

        >>> from stopes.utils.parquet_dataloader import ParquetBasicDataLoader
        >>> from tqdm.auto import tqdm
        >>> source_path = "path/to/parquet/dataset"
        >>> pq_dl = ParquetBasicDataLoader(source_path,
        ...                                columns=["lang1_audio_base_wav", "lang2_audio_base_wav",
        ...                                         "lang1", "lang2", "lang1_text", "lang2_text"],
        ...                                filters=[("split", "=", "train"),
        ...                                        ("lang1", "in", ["eng","spa"]),
        ...                                        ("lang2", "in", ["eng", "spa"])])
        >>> ei_batch = pq_dl.epoch_iterator(100)
        >>> res = []
        >>> for i, batch in tqdm(enumerate(ei_batch)): res.append(len(batch))

    Note that more complicated filters can be applied:
        >>> from pyarrow import dataset as ds
        >>> filters = pa.compute.greater(pa.compute.utf8_length(ds.field("lang1_text")), 4)
    """

    def __init__(
        self,
        dataset_path: str,
        columns: tp.Optional[tp.List[str]] = None,
        filters: tp.Optional[pa.dataset.Expression] = None,
        use_threads: bool = False,
        to_pandas: bool = True,
        split_to_row_groups: bool = True,
    ):
        self.dataset_path = dataset_path
        self.use_threads = use_threads
        self.to_pandas = to_pandas
        self._filter_expression: tp.Optional[pa.dataset.Expression] = (
            pq.filters_to_expression(filters) if filters else None
        )
        # split_row_groups=True is not supported yet
        self.source_ds = pq.ParquetDataset(self.dataset_path, filters=filters)

        self.columns = columns or self.source_ds.schema.names
        assert set(self.columns).issubset(set(self.source_ds.schema.names))

        if self.source_ds.partitioning:
            self.partitioning_keys = [
                name
                for (name, dd) in zip(
                    self.source_ds.partitioning.schema.names,
                    self.source_ds.partitioning.dictionaries,
                )
                if dd is not None
            ]
        else:
            self.partitioning_keys = []

        self._columns_wo_partition_keys = [
            col for col in self.columns if col not in self.partitioning_keys
        ]

        if split_to_row_groups:
            self._all_fragments = [
                piece
                for fragment in self.source_ds._dataset.get_fragments(
                    self._filter_expression
                )
                for piece in fragment.split_by_row_group()
            ]
        else:
            self._all_fragments = list(
                self.source_ds._dataset.get_fragments(self._filter_expression)
            )

    @staticmethod
    def _add_partitioning_values(table, fragment):
        """
        When dealing with partitioned dataset,
        we need to add the corresponding partitioned keys to the each individual fragment Table.
        """
        for key, val in get_partition_keys(fragment.partition_expression).items():
            values = pd.Series([val] * len(table), dtype="category")
            table = table.append_column(key, pa.Array.from_pandas(values))
        return table

    @staticmethod
    def _compute_length_splits(
        length_col: np.ndarray, max_tokens: int
    ) -> tp.List[np.ndarray]:
        """split sequence of length_col in the chunks such that total length is ~ max_tokens
           countint the padding to max length of elements in a chunk

        Args:
            length_col (np.ndarray):
            max_tokens (int):

        Returns:
            tp.List[np.ndarray]: splits that contain indices over the original length_col
        """
        argsort_ind = np.argsort(length_col)
        # TODO: remove 0 lengths
        sorted_length_col = length_col[argsort_ind]

        splits = []
        ptr = 0
        for i, length in enumerate(sorted_length_col):
            if length * (i - ptr) > max_tokens:
                splits.append(argsort_ind[ptr : (i - 1)])
                ptr = i - 1
        if (
            length <= max_tokens
        ):  # we drop the last iteration if it results in a batch greater than max_tokens
            splits.append(argsort_ind[ptr:])
        return splits

    @staticmethod
    def _compute_length(pa_array: pa.Array) -> np.ndarray:
        """
        Computes the length of each element of pa.Array.
        For strings and array types, it relies on pa.compute methods,
        otherwise, fallbacks to pandas.
        Currently, it raises ArrowInvalid on Null values.

        >>> _compute_length(pa.array([[3, 4], [], [1, 3, 4]]))
        ... array([2, 0, 3], dtype=int32)
        """
        type_ = pa_array.type
        if pa.types.is_list(type_) or pa.types.is_large_list(type_):
            length_col = pa.compute.list_value_length(pa_array).to_numpy()
        elif pa.types.is_string(type_):
            length_col = pa.compute.utf8_length(pa_array).to_numpy()
        else:
            length_col = np.asarray(pa_array.to_pandas().apply(len))

        length_col = length_col.copy()
        length_col[np.isnan(length_col)] = 0
        return length_col.astype(np.int32)

    def epoch_iterator(
        self,
        batch_size: tp.Optional[int] = None,
        max_tokens: tp.Optional[int] = None,
        order_by_length: tp.Optional[str] = None,
        min_sample_size: int = 0,
        rank: int = 0,
        world_size: int = 1,
        nb_producers: int = 10,
        nb_prefetch: int = 2,
        seed: tp.Optional[int] = None,
        nb_cpu: int = 10,
    ):
        """
        One full epoch iteration pass over the full dataset that returns randomized batches ordered by length of order_by column length
        args:
            - batch_size: int, iterations with fixed batch size (or sometimes less when there are some remaining elements).
            - max_tokens: int, iterations with batches whose total sum of padded lengths (for column `order_by_length`) ~ `max_tokens`
            - order_by_length: string optional, allows to create batches with uniform lengths of that column (to get a more efficient padding later on)
            - min_sample_size, int, default=0, discards the batches whose length is smaller than `min_sample_size`
            - nb_producers: int, default=10, number of parquet fragments being read and concatenated in parallel to produce batches from
            - nb_prefetch: int, default=2, number of producers to prepare ahead of time
            - seed: tp.Optional[int], to make iteration deterministic
            - nb_cpu: int, number of cpu being used during the iterations

        """
        if not ((batch_size is None) ^ (max_tokens is None)):
            raise ValueError("need to provide either `batch_size` either `max_tokens`")
        if max_tokens is not None and order_by_length is None:
            raise ValueError(
                "`order_by_length` should be given to deal with `max_tokens`"
            )

        np_rs = np.random.RandomState(seed)
        with Parallel(
            n_jobs=max(nb_cpu // 2, 1), backend="threading", return_as="generator"
        ) as parallel_outer, Parallel(
            n_jobs=nb_cpu, backend="threading", return_as="generator"
        ) as parallel_inner, pyarrow_cpu(
            nb_cpu
        ):
            if order_by_length is not None:
                columns = sorted(
                    set(self._columns_wo_partition_keys) | set([order_by_length])
                )
            else:
                columns = self._columns_wo_partition_keys

            def load_one_fragement(fragment):
                fragment_table = fragment.to_table(
                    columns=columns, use_threads=self.use_threads
                )
                fragment_table = self._add_partitioning_values(fragment_table, fragment)
                if self._filter_expression is not None:
                    fragment_table = fragment_table.filter(self._filter_expression)
                return fragment_table

            def get_table_producer(all_fragments, nb_producers, nb_prefetch):
                def unit_table_producer():
                    all_shuffled_fragments = np_rs.permutation(all_fragments)
                    for block in batched(
                        all_shuffled_fragments, nb_producers * nb_prefetch
                    ):
                        yield from parallel_inner(
                            delayed(load_one_fragement)(frag) for frag in block
                        )

                for tables in batched(unit_table_producer(), nb_producers):
                    yield pa.concat_tables(list(tables)).combine_chunks()

            def table_iterator(table):
                if order_by_length is not None:
                    length_col = self._compute_length(table[order_by_length])
                    # add small perturbation to avoid same sample appear together during different epochs
                    length_col += np_rs.randint(
                        0, max(np.quantile(length_col, 0.001), 2), len(length_col)
                    )
                else:
                    length_col = np_rs.randint(0, 2**23, len(table))

                table = table.select(self.columns)

                if batch_size is not None:
                    # faster equivalent to
                    # yield from table.take(np.argsort(length_col)).to_batches(batch_size)
                    order_tt = pa.Table.from_arrays(
                        [pa.array(np.argsort(length_col))], ["order"]
                    )
                    batches = order_tt.to_batches(batch_size)
                    batches = [
                        batches[i]["order"] for i in np_rs.permutation(len(batches))
                    ]
                    yield from parallel_outer(delayed(table.take)(bb) for bb in batches)
                elif max_tokens is not None:
                    splits = self._compute_length_splits(length_col, max_tokens)
                    splits = [splits[i] for i in np_rs.permutation(len(splits))]
                    yield from parallel_outer(delayed(table.take)(bb) for bb in splits)
                else:
                    raise ValueError("unknown batching method")

            local_fragments = np.split(
                np.array(self._all_fragments, dtype="O"), world_size
            )[rank]
            table_producer = get_table_producer(
                local_fragments, nb_producers, nb_prefetch
            )
            for table in table_producer:
                for out in table_iterator(table):
                    if len(out) < min_sample_size:
                        continue
                    if self.to_pandas:
                        out = out.to_pandas()
                    yield out
