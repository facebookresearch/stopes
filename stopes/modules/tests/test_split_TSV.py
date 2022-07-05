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

from stopes.modules.nmt_bitext_eval_utils.preproc_binarized_mined_utils import split_TSV

# Defining variables for test
test_src_lang = "ben"
test_tgt_lang = "hin"
test_size = 5
test_threshold = 1.06

# Defining input & desired output for the test - to be written to tmp file
test_input_str = """1.9999999\tওয়েল, যেমন আপনি ইতিমধ্যে জানেন, আমরা নিজেদেরকে একজন আমেরিকান বিড়াল আছে!\tठीक है, जैसा कि आप पहले से ही पता, हम अपने आप को एक अमेरिकी बिल्ली है!
1.2495081\tআমরা একটি অবিশ্বাস্য সঙ্গে একটি সাক্ষাতের জন্য প্রস্তুত?\tहम एक अविश्वसनीय के साथ एक बैठक के लिए तैयार हैं?
1.2491002\t1) কিভাবে ইতালি মধ্যে কিনতে উপর আরও তথ্য পান\t1) कैसे इटली में खरीदने के लिए पर अधिक जानकारी प्राप्त करें
1.0100101\tএটা এখনও পাগল যে আবহাওয়া ফ্রান্স এই তথ্য দেয় না!\tयह अभी भी पागल है कि मौसम फ्रांस इन जानकारी नहीं देता है!
1.0000000\tআঘাতের পর টম কথা বলা\tचोट के बाद टॉम बात कर रहे"""
test_desired_output_lang1_ben_str = "ওয়েল, যেমন আপনি ইতিমধ্যে জানেন, আমরা নিজেদেরকে একজন আমেরিকান বিড়াল আছে!\nআমরা একটি অবিশ্বাস্য সঙ্গে একটি সাক্ষাতের জন্য প্রস্তুত?\n1) কিভাবে ইতালি মধ্যে কিনতে উপর আরও তথ্য পান"
test_desired_output_lang2_hin_str = "ठीक है, जैसा कि आप पहले से ही पता, हम अपने आप को एक अमेरिकी बिल्ली है!\nहम एक अविश्वसनीय के साथ एक बैठक के लिए तैयार हैं?\n1) कैसे इटली में खरीदने के लिए पर अधिक जानकारी प्राप्त करें"


class TestSplitTSV(unittest.TestCase):
    def _count_lines_in_file(self, file: Path):
        total_lines = 0
        with open(file, "r", encoding="utf-8") as fp:
            total_lines = len(fp.readlines())
        return total_lines

    def _write_str_to_file(
        self,
        name_of_file_to_create: str,
        name_of_tmp_directory_to_create_file_in: str,
        string_contents_of_file: str,
    ):
        created_file = os.path.join(
            name_of_tmp_directory_to_create_file_in, name_of_file_to_create
        )

        with open(created_file, "w", encoding="utf-8") as input_file:
            print(string_contents_of_file, file=input_file)

        return created_file

    def test_split_tsv_files(self):
        with tempfile.TemporaryDirectory() as dir_name:
            test_path_to_tsv_file = self._write_str_to_file(
                "test_input", dir_name, test_input_str
            )
            test_desired_output_file_lang1 = self._write_str_to_file(
                "test_desired_output_file_lang1",
                dir_name,
                test_desired_output_lang1_ben_str,
            )
            test_desired_output_file_lang2 = self._write_str_to_file(
                "test_desired_output_file_lang2",
                dir_name,
                test_desired_output_lang2_hin_str,
            )

            test_output_file_lang_1 = os.path.join(
                dir_name, f"test_step1_lang1_split_tsv_file.001.{test_src_lang}"
            )
            test_output_file_lang_2 = os.path.join(
                dir_name, f"test_step1_lang2_split_tsv_file.001.{test_tgt_lang}"
            )

            total_accepted_lines = split_TSV(
                test_output_file_lang_1,
                test_output_file_lang_2,
                test_path_to_tsv_file,
                test_size,
                test_threshold,
            )

            self.assertLessEqual(
                total_accepted_lines,
                test_size,
            )

            file_1_line_count = self._count_lines_in_file(test_output_file_lang_1)
            file_2_line_count = self._count_lines_in_file(test_output_file_lang_2)
            self.assertEqual(
                file_1_line_count,
                total_accepted_lines,
            )
            self.assertEqual(
                file_2_line_count,
                total_accepted_lines,
            )

            self.assertTrue(
                filecmp.cmp(
                    test_desired_output_file_lang1,
                    test_output_file_lang_1,
                    shallow=False,
                ),
                f"test_output_file_lang_1 file: {test_output_file_lang_1} should have the same contents as test_desired_output_file_lang1: {test_desired_output_file_lang1}, but it doesn't",
            )

            self.assertTrue(
                filecmp.cmp(
                    test_desired_output_file_lang2,
                    test_output_file_lang_2,
                    shallow=False,
                ),
                f"test_output_file_lang_2 file: {test_output_file_lang_2} should have the same contents as test_desired_output_file_lang2: {test_desired_output_file_lang2}, but it doesn't",
            )
