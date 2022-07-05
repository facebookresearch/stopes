# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import typing as tp
from enum import Enum
from itertools import islice

import sentencepiece as spm
import submitit

import stopes.pipelines.prepare_data.data_types as data_types

SPM_PREFIX = "spm."  # could probably be a config


class SPMOutputFormat(Enum):
    ID = "id"
    PIECE = "piece"


class SPMTokenizer(submitit.helpers.Checkpointable):
    def __init__(self, output_format: SPMOutputFormat = SPMOutputFormat.PIECE):
        # we do basic checkpointing with submitit Checkpointable which will store the state of this
        # callable. The basic idea here is to remember the last line processed
        self.processed_lines = 0
        self.output_format = output_format

    def _encode_line(self, line: str, sp: spm.SentencePieceProcessor) -> tp.List[str]:
        line = line.strip()
        if self.output_format == SPMOutputFormat.PIECE:
            return sp.EncodeAsPieces(line)
        elif self.output_format == SPMOutputFormat.ID:
            return sp.EncodeAsIds(line)

    def __call__(self, input_file: str, output_dir: str, vocab: data_types.BuiltVocab):
        sp = spm.SentencePieceProcessor()
        sp.Load(vocab.model_file)

        basename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, SPM_PREFIX + basename)

        # we could run this in multiprocessing, it would make checkpointing harder
        with open(
            output_file, "a", encoding="utf-8", newline="\n"
        ) as output:  # we append for checkpointing
            with open(
                input_file, "r", encoding="utf-8", newline="\n", errors="replace"
            ) as input:
                for line in islice(
                    input, self.processed_lines, None
                ):  # in case we are checkpointing, start from where we stopped
                    encoded = self._encode_line(line, sp)
                    output.write(" ".join(encoded) + "\n")
                    self.processed_lines += 1

        return output_file
