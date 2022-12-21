# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
import typing as tp
from pathlib import Path

import sentencepiece as spm
from fairseq.binarizer import BinarizeSummary, VocabularyDatasetBinarizer
from fairseq.data import Dictionary, indexed_dataset
from fairseq.tokenizer import tokenize_line

from stopes.core import utils
from stopes.modules.preprocess.multiproc_line_processor import (
    MultiprocLineProcessorCallback,
)

logger = logging.getLogger(__name__)


class MultiProcFairSeqBinarizerEncoder(MultiprocLineProcessorCallback):
    def __init__(
        self,
        outfile_prefix: str,
        input_file: str,
        input_file_idx: int,
        output_dir: str,
        offset_start: tp.Optional[int],
        offset_end: tp.Optional[int],
        merging: bool,
        # Custom params
        vocab_file_path: str,
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
            offset_start=offset_start,
            offset_end=offset_end,
            merging=merging,
        )

        out_dir = Path(output_dir).resolve()
        out_filename = Path(input_file).name
        if merging:
            shard_str = self._extract_shard_str(out_filename)
            utils.ensure_dir(out_dir / shard_str)
            out_filename = out_filename.replace(f".{shard_str}", "")
            outfile = out_dir / shard_str / out_filename
        else:
            outfile = out_dir / f"{out_filename}_{offset_start}_{offset_end}"
        self.output_file = Path(indexed_dataset.data_file_path(str(outfile)))
        logger.info(f"Binarizing {self.output_file}")

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

    @staticmethod
    def _extract_shard_str(filename: str) -> str:
        """
        extract shard string from the filename.
        """
        m = re.search(r"shard\d{3}", filename)
        # Default is the first shard.
        if m is None:
            return "shard000"
        return m.group()

    def final_result(self) -> Path:
        return self.output_file

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
        logger.info(self.summary)

    def __exit__(self, _exc_type, _exc_value, _traceback):
        logger.info(self.summary)
        self.dataset_builder.finalize(self.output_file.with_suffix(".idx"))
        return None

    def merge_results(self, splits: tp.List[tp.Any]) -> tp.Any:
        self.dataset_builder = indexed_dataset.make_builder(
            str(self.output_file),
            impl=self.dataset_impl,
            vocab_size=len(self.vocab),
        )
        for split in splits:
            self.dataset_builder.merge_file_(str(split.with_suffix("")))
        self.dataset_builder.finalize(self.output_file.with_suffix(".idx"))
        return self.output_file

    @classmethod
    def version(cls):
        return "0.2"
