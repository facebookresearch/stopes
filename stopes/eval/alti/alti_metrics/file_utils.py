# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import typing as tp
from pathlib import Path


def read_tsv(filename: Path, named_columns: bool = True) -> tp.List[tp.Dict]:
    """Read a named or unnamed tsv file and return a list of row dicts"""
    result = []
    with open(filename, "r", encoding="utf-8") as csvfile:
        if named_columns:
            reader = csv.DictReader(csvfile, delimiter="\t")
            for row in reader:
                result.append(row)
        else:
            reader = csv.reader(csvfile, delimiter="\t")
            for row in reader:
                result.append(dict(enumerate(row)))
    return result


def write_tsv(filename: Path, data: tp.List[tp.Dict]) -> None:
    """Write a named tsv file"""
    with open(filename, "w", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=list(data[0].keys()), delimiter="\t"
        )
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def select_columns(data: tp.List[tp.Dict], column_ids: tp.List) -> tp.List[tp.List]:
    """Select list of "columns" from a list of dicts"""
    columns = [[] for _ in column_ids]
    for item in data:
        for column, column_id in zip(columns, column_ids):
            column.append(item[column_id])
    return columns


def join_lists_of_dicts(*inputs: tp.List[tp.List[tp.Dict]]) -> tp.List[tp.Dict]:
    """Loop through several lists of dicts and join the corresponding ellements"""
    results = []
    for items in zip(*inputs):
        results.append({k: v for item in items for k, v in item.items()})
    return results
