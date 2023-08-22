# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List

DictRow = Dict[str, Any]


def read_tsv(filename: Path, named_columns: bool = True) -> List[DictRow]:
    """Read a named or unnamed tsv file and return a list of row dicts"""
    result = []
    with open(filename, "r", encoding="utf-8") as csvfile:
        if named_columns:
            for dict_row in csv.DictReader(csvfile, delimiter="\t"):
                result.append(dict_row)
        else:
            for row in csv.reader(csvfile, delimiter="\t"):
                result.append(dict(enumerate(row)))
    return result


def write_tsv(filename: Path, data: List[DictRow]) -> None:
    """Write a named tsv file"""
    with open(filename, "w", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=list(data[0].keys()), delimiter="\t"
        )
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def select_columns(
    data: Iterable[DictRow], column_ids: List[str]
) -> List[List[DictRow]]:
    """Select list of "columns" from a list of dicts"""
    columns: List[List[DictRow]] = [[] for _ in column_ids]
    for item in data:
        for column, column_id in zip(columns, column_ids):
            column.append(item[column_id])
    return columns


def join_lists_of_dicts(*inputs: Iterable[DictRow]) -> List[DictRow]:
    """Loop through several lists of dicts and join the corresponding elements"""
    results = []
    for items in zip(*inputs):
        results.append({k: v for item in items for k, v in item.items()})
    return results
