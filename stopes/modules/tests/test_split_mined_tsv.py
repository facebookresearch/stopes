# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import filecmp
import os
import tempfile
import unittest
from pathlib import Path

from stopes.modules.nmt_bitext_eval_utils import split_mined_tsv

# Tsv file with 5 scored bitext (don't need to be sorted)
SAMPLE_TSV = """1.9999999\taaa\tAAA
1.2495081\tbbb\tBBB
1.0100101\tccc\tCCC
1.2491002\tddd\tDDD
1.0000000\teee\tEEE
"""


def test_split_mined_tsv_files_filters_low_score(tmp_path: Path):
    tsv_file = tmp_path / "sample.tsv"
    tsv_file.write_text(SAMPLE_TSV)
    low_file = tmp_path / "sample.low"
    upp_file = tmp_path / "sample.upp"
    accepted_lines = split_mined_tsv(
        bitext_tsv=tsv_file,
        output_left=low_file,
        output_right=upp_file,
        max_number_of_lines=1000,
        threshold=1.06,
    )
    # only 3 bitext out of 5 have score >= 1.06
    assert accepted_lines == 3

    expected_low = "aaa\nbbb\nddd\n"
    assert expected_low == low_file.read_text()

    expected_upp = "AAA\nBBB\nDDD\n"
    assert expected_upp == upp_file.read_text()


def test_split_mined_tsv_files_no_filtering(tmp_path: Path):
    tsv_file = tmp_path / "sample.tsv"
    tsv_file.write_text(SAMPLE_TSV)
    low_file = tmp_path / "sample.low"
    upp_file = tmp_path / "sample.upp"
    accepted_lines = split_mined_tsv(
        bitext_tsv=tsv_file,
        output_left=low_file,
        output_right=upp_file,
        max_number_of_lines=1000,
        # Threshold is set to 0, it should take all the lines
        threshold=0,
    )
    # We should have all the lines
    assert accepted_lines == 5
    assert accepted_lines == low_file.read_text().count("\n")
    assert accepted_lines == upp_file.read_text().count("\n")
    assert accepted_lines == SAMPLE_TSV.count("\n")


def test_split_mined_tsv_files_max_number_of_lines(tmp_path: Path):
    tsv_file = tmp_path / "sample.tsv"
    tsv_file.write_text(SAMPLE_TSV)
    low_file = tmp_path / "sample.low"
    upp_file = tmp_path / "sample.upp"
    accepted_lines = split_mined_tsv(
        bitext_tsv=tsv_file,
        output_left=low_file,
        output_right=upp_file,
        # Will only take the first one
        max_number_of_lines=1,
        threshold=0,
    )
    # We should have all the lines
    assert accepted_lines == 1
    expected_low = "aaa\n"
    assert expected_low == low_file.read_text()

    expected_upp = "AAA\n"
    assert expected_upp == upp_file.read_text()
