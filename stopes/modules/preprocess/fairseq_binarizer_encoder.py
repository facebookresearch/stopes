# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import typing as tp
from pathlib import Path

import sentencepiece as spm
from fairseq.binarizer import BinarizeSummary, VocabularyDatasetBinarizer
from fairseq.data import Dictionary, indexed_dataset
from fairseq.tokenizer import tokenize_line

from stopes.modules.preprocess.line_processor import LineProcessorCallback
from stopes.utils.mining_utils import extract_shard_id

log = logging.getLogger(__name__)


class FairSeqBinarizerEncoder(LineProcessorCallback):
    def __init__(
        self,
        outfile_prefix: str,
        input_file: str,
        input_file_idx: int,
        output_dir: str,
        vocab_file_path: str,
        outfile_postfix: str = "",
        spm_model_path: tp.Optional[str] = None,
        dataset_impl: str = "mmap",
        append_eos: bool = True,
        reverse_order: bool = False,
    ) -> None:

        super().__init__(
            outfile_prefix=outfile_prefix,
            input_file=input_file,
            input_file_idx=input_file_idx,
            output_dir=output_dir,
            outfile_postfix=outfile_postfix,
        )
        self.output_file = self.final_result()
        self.spm_tokenizer = None
        if spm_model_path is not None:
            self.spm_tokenizer = spm.SentencePieceProcessor()
            assert self.spm_tokenizer.Load(str(spm_model_path))

        self.vocab = Dictionary.load(str(vocab_file_path))
        self.binarizer = VocabularyDatasetBinarizer(
            self.vocab,
            self._tokenize,
            append_eos=append_eos,
            reverse_order=reverse_order,
        )
        self.dataset_impl = dataset_impl
        SUPPORTED = ("mmap", "cached", "lazy")
        assert dataset_impl in SUPPORTED, (
            f"Unsupported datasetimpl: {dataset_impl}." f" Chose from: {SUPPORTED}."
        )
        self.summary = BinarizeSummary()

    def final_result(self) -> Path:
        shard_idx = extract_shard_id(self.input_file, default=self.input_file_idx)
        outfile = (
            Path(self.output_dir).resolve()
            / f"{self.outfile_prefix}.{shard_idx:03d}{self.outfile_postfix}"
        )
        outfile = Path(indexed_dataset.data_file_path(str(outfile)))
        return outfile

    def _tokenize(self, line: str) -> tp.List[str]:
        if self.spm_tokenizer is not None:
            return self.spm_tokenizer.EncodeAsPieces(line)
        else:
            return tokenize_line(line)

    def __enter__(self):
        self.dataset_builder = indexed_dataset.make_builder(
            str(self.output_file),
            impl=self.dataset_impl,
            vocab_size=len(self.vocab),
        )
        return self

    def process_lines(self, lines_with_number: tp.Iterator[tp.Tuple[int, str]]) -> None:
        summary = BinarizeSummary()
        for (_, s) in lines_with_number:
            self.dataset_builder.add_item(self.binarizer.binarize_line(s, summary))
        self.summary.merge(summary)
        log.info(self.summary)

    def __exit__(self, _exc_type, _exc_value, _traceback):
        log.info(self.summary)
        self.dataset_builder.finalize(self.output_file.with_suffix(".idx"))

        return None

    @classmethod
    def version(cls):
        return "0.2"
