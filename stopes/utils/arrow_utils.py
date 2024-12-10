# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import typing as tp

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


class DataClassEncoder(json.JSONEncoder):
    def default(self, obj):
        import dataclasses

        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        import omegaconf

        if isinstance(obj, omegaconf.dictconfig.DictConfig):
            resolved = omegaconf.OmegaConf.to_container(obj, resolve=True)
            return resolved
        import enum

        if isinstance(obj, enum.Enum):
            return obj.name  # or return obj.name if you want the name of the enum
        return super().default(obj)


def hash_table_with_schema(table, seed: int = 0) -> str:
    """
    Computes a hash for a pyarrow.Table including its schema using xxHash.
    This function serializes the schema of the table and updates the hash with it,
    ensuring that any changes in the schema affect the hash result. It then iterates
    over each column and chunk in the table, updating the hash with the data buffers.
    This approach provides a comprehensive hash that reflects both the structure
    and content of the table.

    Parameters:
    - table (pyarrow.Table): The PyArrow table to hash.
    - seed (int, optional): An optional seed for the xxHash function. Default is 0.
    Returns:
    - str: The hexadecimal string representing the hash of the table including its schema.
    Example:
    >>> data = {
        'column1': [1, 2, 3, 4],
        'column2': ['foo', 'bar', 'baz', 'qux']
    }
    >>> table = pa.Table.from_pydict(data)
    >>> hash_table_with_schema(table)
    394e32679db7eced

    """
    import xxhash

    hash_obj = xxhash.xxh64(seed=seed)
    # Serialize the schema to a string and update the hash
    schema_str = table.schema.serialize().to_pybytes()
    hash_obj.update(schema_str)

    for column in table.itercolumns():
        for buffer in pyarrow_column_to_array(column).buffers():
            if buffer is not None:
                hash_obj.update(buffer)

    return hash_obj.hexdigest()


def add_metadata_to_table(table: pa.Table, meta: dict) -> pa.Table:
    existing_metadata = table.schema.metadata or {}
    encoded_meta = {
        key: json.dumps(val, cls=DataClassEncoder) for key, val in meta.items()
    }
    if existing_metadata:
        encoded_meta["previous_metadata"] = json.dumps(
            {
                key.decode("utf-8"): val.decode("utf-8")
                for key, val in existing_metadata.items()
            },
            cls=DataClassEncoder,
        )

    return table.replace_schema_metadata(encoded_meta)


def is_list_like(arr):
    return pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type)


def _fix_list_offset(arr: pa.Array) -> pa.Array:
    """
    Recursively fixes list offset to 0, so that arr.offsets are always starts from 0
    and can be used easily downstream.
    """
    if not is_list_like(arr):
        return arr
    if arr.offset == 0:
        return arr

    new_values = _fix_list_offset(pc.list_flatten(arr))
    new_offsets = pc.subtract(arr.offsets, arr.offsets[0])

    return (
        pa.LargeListArray.from_arrays(new_offsets, new_values)
        if pa.types.is_large_list(arr.type)
        else pa.ListArray.from_arrays(new_offsets, new_values)
    )


def pyarrow_column_to_array(arg: tp.Union[pa.ChunkedArray, pa.Array]) -> pa.Array:
    # see https://github.com/apache/arrow/issues/37318
    if isinstance(arg, pa.Array):
        return _fix_list_offset(arg)

    return _fix_list_offset(
        arg.chunk(0) if arg.num_chunks == 1 else arg.combine_chunks()
    )


def numpy_to_fixed_size_pyarrow_array(array: np.ndarray) -> pa.Array:
    assert array.ndim == 2
    buffer = array.ravel(order="C")
    return pa.FixedSizeListArray.from_arrays(pa.array(buffer), array.shape[1])


def apply_on_nested_array(
    fn: tp.Callable[[pa.Array], pa.Array], arr: tp.Union[pa.ChunkedArray, pa.Array]
) -> tp.Union[pa.ChunkedArray, pa.Array]:
    if is_list_like(arr):
        arr = pyarrow_column_to_array(arr)
        res = apply_on_nested_array(fn, pc.list_flatten(arr))

        assert arr.offset == 0
        cls = pa.LargeListArray if pa.types.is_large_list(arr.type) else pa.ListArray
        output = cls.from_arrays(arr.offsets, res)
        if arr.null_count > 0:
            output = pc.if_else(pc.is_null(arr), None, output)
        return output

    return fn(arr)


def pyarrow_fixed_size_array_to_numpy(
    cc: tp.Union[pa.ChunkedArray, pa.Array],
) -> np.ndarray:
    cc = pyarrow_column_to_array(cc)
    assert cc.null_count == 0
    assert cc.type.list_size is not None
    return np.reshape(np.asarray(pc.list_flatten(cc)), (-1, cc.type.list_size))


def nested_pyarrow_to_torch(arr: pa.Array):
    """
    Transforms are List[List[ListOfFixedSize]] to Nested Torch Tensors of shape :
        - batch_size x SeqLen* x Dim
    The Tensor representation of Seq of Vectors batch.

    One can use
    >>> normal_torch_tensor = nested_pyarrow_to_torch(arr).to_padded_tensor(0.)

    Args:
        arr (pa.Array):

    Returns:
        torch.Tensor:
    """
    import torch

    return torch.nested.as_nested_tensor(
        arr.to_pandas().map(np.vstack).map(torch.from_numpy).tolist()
    )


def explode_table_include_null(
    table: pa.Table, columns: tp.Union[str, tp.Sequence[str]]
) -> pa.Table:
    """
    Similar to pandas.DataFrame.explode method for pyarrow.Table
    >>> table = pa.table({'a': range(3), 'b': [[1, 2], None, [3, 4, 5]]})
    >>> explode_table_include_null(table, 'b').to_pandas()
       a  b
    0  0  1
    1  0  2
    2  2  3
    3  2  4
    4  2  5


    Args:
        table (pa.Table):
        columns (str): list type columns in table

    Returns:
        pa.Table
    """
    if isinstance(columns, str):
        columns = [columns]

    assert len(columns) > 0

    other_columns = list(table.schema.names)
    for column in columns:
        other_columns.remove(column)

    # checking compatibility
    new_cols = []
    lengths = pc.list_value_length(pc.fill_null(table[columns[0]], [None])).to_numpy()

    for name in columns:
        col = pc.fill_null(table[name], [None])
        # checking that all columns list structures are parallel
        assert (lengths == pc.list_value_length(col).to_numpy()).all()
        new_cols.append(pc.list_flatten(col))

    if len(other_columns) > 0:
        indices = pc.list_parent_indices(pc.fill_null(table[columns[0]], [None]))
        result = table.select(other_columns).take(indices)

        for name, new_col in zip(columns, new_cols):
            result = result.append_column(
                pa.field(name, table.schema.field(name).type.value_type), new_col
            )
    else:
        result = pa.Table.from_arrays(new_cols, columns)

    return result


# numba njit
def _get_indices_and_offsets(lengths, max_seq_len):
    new_lengths, res = [], []
    for i, ll in enumerate(lengths):
        nb_full, remaining = ll // max_seq_len, ll % max_seq_len

        if remaining != 0:
            res.append(np.full((nb_full + 1), i, dtype=np.int32))
            new_lengths.append(
                np.array([max_seq_len] * nb_full + [remaining], dtype=np.int32)
            )
        else:
            res.append(np.full(nb_full, i, dtype=np.int32))
            new_lengths.append(np.array([max_seq_len] * nb_full, dtype=np.int32))

    return (
        np.concatenate(res),
        np.concatenate([np.array([0], dtype=np.int32)] + new_lengths).cumsum(),
    )


def _cast_fs16_to_int16(table: pa.Table) -> pa.Table:
    # polars does not work with fs16 data type, but works with int16
    def _view_as_fs16(col):
        if pa.types.is_fixed_size_list(col.type) and pa.types.is_float16(
            col.type.value_type
        ):
            return col.view(pa.list_(pa.int16(), col.type.list_size))
        elif pa.types.is_float16(col.type):
            return col.view(pa.int16())
        else:
            return col

    out = {}
    for col in table.column_names:
        out[col] = apply_on_nested_array(_view_as_fs16, table[col])

    return pa.Table.from_pydict(out)


def _cast_back_int16_to_fs16(table: pa.Table, reference_table: pa.Table) -> pa.Table:
    # for compatibility with polars we cast int16 back to fs16
    # large_list to simple list
    for col in table.column_names:
        if pa.types.is_large_list(table[col].type) and pa.types.is_list(
            reference_table[col]
        ):
            table = table.drop(col).append_column(
                col, table[col].cast(pa.list_(table[col].type.value_type))
            )
        if table[col].type != reference_table[col].type:
            casted_columns = pyarrow_column_to_array(table[col]).view(
                reference_table[col].type
            )
            table = table.drop(col).append_column(col, casted_columns)
    return table


def explode_table_with_fixed_length(
    table: pa.Table, columns: tp.Union[str, tp.Sequence[str]], max_seq_len: int
) -> pa.Table:
    """
    This function takes an Apache Arrow Table, explodes it based on the specified columns,
    and then rechunks the exploded table based on a specified sequence length

    ## Parameters:
    - `table` (`pa.Table`): The input Apache Arrow Table that needs to be exploded and rechunked.
    - `columns` (`tp.Union[str, tp.Sequence[str]]`): The column or columns on which the table should be exploded.
    - `max_seq_len` (`int`): The sequence length for rechunking the exploded table. This should be a positive integer.
    ## Returns:
    - `pa.Table`: The rechunked Table after exploding on the specified columns.

    ## Example:

    >>> table = pa.Table.from_pydict({"col1": [[1, 2], [3, 4, 5, 6, 7], [8, 10], [11]],
    ...                  "col2": [[-1, -2], [-3, -4, -5, -6, -7], [-8, -10], [-11]],
    ...                  "col3": ["a", "b", "c", "d"]})
    >>> exploded_table = explode_table_with_fixed_length(table, ["col1", "col2"], 3)
    >>> exploded_table.to_pandas()
            col3        col1           col2 __doc_segments __doc_lengths
    0  [a, a, b]   [1, 2, 3]   [-1, -2, -3]      [0, 0, 1]        [2, 1]
    1  [b, b, b]   [4, 5, 6]   [-4, -5, -6]      [0, 0, 0]           [3]
    2  [b, c, c]  [7, 8, 10]  [-7, -8, -10]      [0, 1, 1]        [1, 2]
    3        [d]        [11]          [-11]            [0]           [1]
    """
    assert max_seq_len > 0
    table = table.append_column("__doc_index", pa.array(np.arange(len(table))))
    flatten_table = explode_table_include_null(table, columns)
    offsets = np.arange(0, len(flatten_table), max_seq_len, dtype=np.int64)
    offsets = pa.array(np.concatenate([offsets, [len(flatten_table)]]))

    if len(flatten_table) < 2 * 32 - 1:
        cls = pa.ListArray.from_arrays
    else:
        cls = pa.LargeListArray.from_arrays

    out = {}
    for col in flatten_table.column_names:
        out[col] = cls(offsets, pyarrow_column_to_array(flatten_table[col]))
    out_table = pa.Table.from_pydict(out)

    # transforming "__doc_index" to ordered segments indices
    # [1, 1, 1, 2, 2, 4, 5, 5] -> [0, 0, 0, 1, 1, 2, 3, 3]
    arr = out_table["__doc_index"]
    rows = [
        pc.value_counts(pc.list_flatten(arr.slice(i, 1))).field(1).to_numpy()
        for i in range(len(arr))
    ]
    doc_lengths = pa.array(rows)
    doc_segments = pa.array(
        [np.repeat(np.arange(len(xx), dtype=np.int32), xx) for xx in rows]
    )

    return (
        out_table.drop(["__doc_index"])
        .append_column("__doc_segments", doc_segments)
        .append_column("__doc_lengths", doc_lengths)
    )


def explode_table_with_max_length(
    table: pa.Table, columns: tp.Union[str, tp.Sequence[str]], max_seq_len: int
) -> pa.Table:
    """
    Unrolling list array into smaller list with fixed max length.
    If provided several `columns`, all columns are supposed to the parallel list structure.

    >>> table = pa.table({'a': range(5), 'b': [[2, 2], None, [3,3,3], [4,4,4,4], [5,5,5,5,5]]})
    >>> explode_table_with_max_length(table, "b", 2).to_pandas()
       a           b
    0  0  [2.0, 2.0]
    1  1       [nan]
    2  2  [3.0, 3.0]
    3  2       [3.0]
    4  3  [4.0, 4.0]
    5  3  [4.0, 4.0]
    6  4  [5.0, 5.0]
    7  4  [5.0, 5.0]
    8  4       [5.0]
    >>> explode_table_with_max_length(table, "b", 3).to_pandas()
       a                b
    0  0       [2.0, 2.0]
    1  1            [nan]
    2  2  [3.0, 3.0, 3.0]
    3  3  [4.0, 4.0, 4.0]
    4  3            [4.0]
    5  4  [5.0, 5.0, 5.0]
    6  4       [5.0, 5.0]
    """
    if isinstance(columns, str):
        columns = [columns]

    assert len(columns) > 0

    cols = [pc.fill_null(table[columns[0]], [None])]
    lengths = pc.list_value_length(cols[0]).to_numpy()

    for name in columns[1:]:
        col = pc.fill_null(table[name], [None])
        # checking that all columns list structures are parallel
        assert (lengths == pc.list_value_length(col).to_numpy()).all()
        cols.append(col)

    # next unroll with max_seq_len
    indices, new_offests = _get_indices_and_offsets(lengths, max_seq_len)

    other_columns = list(table.schema.names)
    for name in columns:
        other_columns.remove(name)

    remaining_table = table.select(other_columns).take(indices)

    result_dict = {}
    for name in other_columns:
        result_dict[name] = remaining_table[name]

    for name, col in zip(columns, cols):
        rolled_array = pa.ListArray.from_arrays(
            offsets=new_offests,
            values=pyarrow_column_to_array(pc.list_flatten(col)),
        )
        result_dict[name] = rolled_array

    return pa.Table.from_pydict(result_dict, schema=table.schema)


def nested_numpy_to_pyarrow(series: tp.Union[list, np.ndarray]) -> pa.Array:
    """
    >>> a = [np.random.rand(i + 1, 2) for i in range(10)]
    >>> nested_array = nested_numpy_to_pyarrow(a)
    >>> nested_array.type
    ListType(list<item: fixed_size_list<item: double>[2]>)
    """
    offsets = np.array([0] + list(map(len, series)), dtype=np.int32).cumsum()
    values = numpy_to_fixed_size_pyarrow_array(np.vstack(list(map(np.asarray, series))))
    return pa.ListArray.from_arrays(offsets, values)


def simple_array_to_nested(arr: tp.Union[pa.ChunkedArray, pa.Array]) -> pa.Array:
    """
    >>> a = pa.array([1,2,3])
    >>> simple_array_to_nested(a).to_pylist()
    [[1], [2], [3]]
    """
    return pa.ListArray.from_arrays(
        pa.array(np.arange(len(arr) + 1, dtype=np.int32)), pyarrow_column_to_array(arr)
    )


def hstack_pyarray_list(*arrays: tp.Union[pa.ChunkedArray, pa.Array]) -> pa.Array:
    """
    Example with simple list:
    >>> a = pa.array([[1], [2,3], [5], []])
    >>> b = pa.array([[-1, -3], [-11], [], [22]])
    >>> hstack_pyarray_list(a, b).to_pylist()
    [[1, -1, -3], [2, 3, -11], [5], [22]]

    Example with nested lists :
    >>> data = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]), np.array([[9, 10]])]
    >>> list_array = nested_numpy_to_pyarrow(data)
    >>> list_array.type
    ListType(list<item: fixed_size_list<item: int64>[2]>)
    >>> truncated_list_array = pc.list_slice(list_array, 1, 2)
    [[[3, 4]], [[7, 8]], []]
    >>> hstack_pyarray_list(list_array, truncated_list_array)
    [[[1, 2], [3, 4], [3, 4]],
     [[5, 6], [7, 8], [7, 8]],
     [[9, 10]]]
    """
    assert all(map(is_list_like, arrays))

    lens = list(set(map(len, arrays)))
    assert len(lens) == 1

    list_off_views = [
        pyarrow_column_to_array(pc.list_flatten(arr.slice(i, 1)))
        for i in range(lens[0])
        for arr in arrays
    ]

    is_large = any(pa.types.is_large_list(arr.type) for arr in arrays)

    offsets = np.concatenate(
        [np.array([0]), np.sum([pc.list_value_length(arr) for arr in arrays], axis=0)],
        dtype=np.int64 if is_large else np.int32,
    ).cumsum()

    cls = pa.LargeListArray if is_large else pa.ListArray
    return cls.from_arrays(offsets, pa.concat_arrays(list_off_views))


def apply_over_groups(
    table: pa.Table,
    grp_columns: tp.Optional[tp.List[tp.Optional[str]]],
    table_mapper: tp.Callable[[pa.Table], pa.Table],
) -> pa.Table:
    """
    Apply a mapping function to each group of a PyArrow table.

    Parameters:
    - table: The input PyArrow table to be grouped and mapped.
    - grp_columns: A list of column names to group the table by.
        if `grp_columns=[None]` or `grp_columns=[]` or `grp_columns=None`,
        `table_mapper` is applied on the full table.
        Note also that None values in `grp_columns` will be filtered,
        so one can you use grp_columns=[col1, col2] where each of col1 and col2 can be None.
    - table_mapper: A callable function that takes a PyArrow table as input and returns a new PyArrow table.

    Returns:
    - A new PyArrow table resulting from applying the `table_mapper` function to each group of the input table.

    Notes:
    - The function adds a temporary column "__uuu_index" to the input table to facilitate grouping and sorting.
    - The `table_mapper` function is applied to each group of the table, and the resulting tables are concatenated,
        Therefore, the resulting sub-tables should have the same schema
    - The function adds a temporary column "__uuu_index" to keep track of the original order of the rows.
        So, it should be kept inchanged by `table_mapper`.
        This column is removed in the final result.
    """

    # shortcut for no group case
    if grp_columns is None:
        return table_mapper(table)

    grp_columns = [x for x in grp_columns if x is not None]
    if len(grp_columns) == 0:
        return table_mapper(table)

    table = table.append_column(
        "__uuu_index", pa.array(np.arange(len(table), dtype=np.int32))
    )
    split_grps = (
        table.select(grp_columns + ["__uuu_index"])
        .group_by(grp_columns)
        .aggregate([("__uuu_index", "list")])
    )
    # shortcut for single group case
    if len(split_grps) == 1:
        return table_mapper(table.drop("__uuu_index"))

    # to iterate per rows we convert to pandas
    # TODO : this could be called in parallel
    results = [
        table_mapper(table.take(pa.array(ind)))
        for ind in split_grps["__uuu_index_list"].to_pandas()
    ]

    result = pa.concat_tables(results, promote_options="permissive")
    del results

    if "__uuu_index" in result.column_names:
        return result.sort_by("__uuu_index").drop("__uuu_index")

    return result.combine_chunks()


def first_element_in_nested_list(arr: tp.Union[pa.ChunkedArray, pa.Array]) -> pa.Array:
    """
    >>> arr = pa.array([[[1, 2], [-1]], [[3, 2, 1], [], [4]] ])
    >>> first_element_in_nested_list(arr).to_pylist()
    [[1, -1], [3, None, 4]]
    """
    arr = pyarrow_column_to_array(arr)
    return pa.ListArray.from_arrays(
        arr.offsets,
        pc.list_flatten(
            pc.list_slice(arr.flatten(), start=0, stop=1, return_fixed_size_list=True)
        ),
    )
