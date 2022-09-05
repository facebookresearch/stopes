# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import typing as tp
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path

from stopes.modules.preprocess.multiproc_bitext_processor import (
    MultiprocBitextProcessorCallback,
)
from stopes.pipelines.filtering.dataset import DatasetLine
from stopes.pipelines.monolingual.monolingual_line_processor import (
    FilterConfig,
    FilterLID,
    FilterLogic,
    LIDConfig,
    SentenceSplitClean,
    cat_splits,
)
from stopes.pipelines.monolingual.utils.text_filter import keep_it


@dataclass
class BitextProcessResult:
    src_input_file: Path
    tgt_input_file: Path
    src_output_file: Path
    tgt_output_file: Path
    discarded_src_output_file: Path
    discarded_tgt_output_file: Path
    kept: int = 0
    filtered: int = 0
    extra_filtered: int = 0
    script_filtered: int = 0
    script_mismatch: int = 0
    lid_filtered: int = 0
    lid_mismatch: int = 0
    paragraphs: int = 0
    sentences: int = 0

    @classmethod
    def merge(
        cls,
        src_input_file: Path,
        tgt_input_file: Path,
        src_output_file: Path,
        tgt_output_file: Path,
        discarded_src_output_file: Path,
        discarded_tgt_output_file: Path,
        splits: tp.Iterable["BitextProcessResult"],
    ) -> "BitextProcessResult":
        ret = BitextProcessResult(
            src_input_file,
            tgt_input_file,
            src_output_file,
            tgt_output_file,
            discarded_src_output_file,
            discarded_tgt_output_file,
        )
        for f in fields(ret):
            if f.name in {
                "src_input_file",
                "tgt_input_file",
                "src_output_file",
                "tgt_output_file",
                "discarded_src_output_file",
                "discarded_tgt_output_file",
            }:
                continue
            setattr(ret, f.name, sum([getattr(s, f.name) for s in splits]))
        return ret

    @classmethod
    def table_columns(cls):
        return [f.name for f in fields(cls)]

    def get_data_row(self):
        def str_or_num(v):
            if isinstance(v, Path):
                return str(v)
            return v

        return [str_or_num(getattr(self, f.name)) for f in fields(self)]


def extract_distillation_metadata(line: str) -> tp.Tuple[str, str]:
    """
    returns a tuple with first the paragraph content, then the metadata to keep.

    takes the content in the first column, everything else is metadata
    """
    if (
        line == "\n" or line == ""
    ):  # TODO: remove this check once the extra newlines problem in shard_and_shuffle merge is resolved
        return ("", "")
    line_parts = line.split()
    if len(line_parts) > 1:
        return (" ".join(line_parts[5:]), "\t".join(line_parts[:5]))
    return (line_parts[0], "")


class BitextSplitNormalizeFilterLID(MultiprocBitextProcessorCallback):
    def __init__(
        self,
        # set by MultiProcBitextProcessorModule
        outfile_prefix: str,
        src_input_file: str,
        tgt_input_file: str,
        input_files_idx: int,
        output_dir: str,
        line_offset_start: tp.Optional[int],
        line_offset_end: tp.Optional[int],
        merging: bool,
        # set by the config
        tgt_lang: str,
        split_algo: str,
        filter_config: FilterConfig,
        lid_config: LIDConfig,
        lang_script: str,
        tgt_splitter_lang: str,
        src_splitter_lang: str,
        num_cpu: int,
        local_tmp_dir: str,
        _version: str,  # used to bump config version and invalidate cache
        skip_dedup: bool = False,
    ) -> None:
        super().__init__(
            outfile_prefix=outfile_prefix,
            src_input_file=src_input_file,
            tgt_input_file=tgt_input_file,
            input_files_idx=input_files_idx,
            output_dir=output_dir,
            line_offset_start=line_offset_start,
            line_offset_end=line_offset_end,
            merging=merging,
        )

        self.tgt_lang = tgt_lang
        self.tgt_splitter_lang = tgt_splitter_lang
        self.lang_script = lang_script
        self.lid_config = lid_config
        self.filter_config = filter_config
        self.split_algo = split_algo

        self.local_tmp_dir = local_tmp_dir

        self.num_cpu = num_cpu

        self.skip_dedup = skip_dedup

        # find out the corpus from basename: e.g. /${local_tmp_dir}/source_shard.mai-deu.deu.xz_expanded.txt
        src_input_parts = os.path.basename(src_input_file).split(".")
        if src_input_parts[-1] == "xz":
            self.corpus = src_input_parts[-4]
            self.lang_pair = src_input_parts[-3]
            self.input_lang = src_input_parts[-2]
        elif src_input_parts[-2].endswith("expanded"):
            self.corpus = src_input_parts[-5]
            self.lang_pair = src_input_parts[-4]
            self.input_lang = src_input_parts[-3]
        else:
            self.corpus = src_input_parts[-3]
            self.lang_pair = src_input_parts[-2]
            self.input_lang = src_input_parts[-1]

        self.line_offset_start = line_offset_start

        self.src_output_file = Path(str(self.output_dir)) / (
            f"{self.outfile_prefix}{self.lang_pair}.{self.input_lang}.{input_files_idx:03d}.{line_offset_start}_{line_offset_end}.tsv"
        )

        self.tgt_output_file = Path(self.output_dir) / (
            f"{self.outfile_prefix}{self.lang_pair}.{self.tgt_lang}.{input_files_idx:03d}.{line_offset_start}_{line_offset_end}.tsv"
        )

        print(f"tmp output to files: {self.src_output_file}, {self.tgt_output_file}")

        self.discarded_src_output_file = Path(self.output_dir) / (
            f"{self.outfile_prefix}discarded.{self.lang_pair}.{self.input_lang}.{input_files_idx:03d}.{line_offset_start}_{line_offset_end}.tsv"
        )

        self.discarded_tgt_output_file = Path(self.output_dir) / (
            f"{self.outfile_prefix}discarded.{self.lang_pair}.{self.tgt_lang}.{input_files_idx:03d}.{line_offset_start}_{line_offset_end}.tsv"
        )
        print(
            f"tmp discarded output to files: {self.discarded_src_output_file}, {self.discarded_tgt_output_file}"
        )

        self.src_sentence_split_clean = SentenceSplitClean(
            src_splitter_lang, split_algo
        )
        self.tgt_sentence_split_clean = SentenceSplitClean(
            tgt_splitter_lang, split_algo
        )

        self.tgt_expected_lid_label = f"__label__{self.tgt_lang}"

        self.filter_lid = FilterLID(
            self.filter_config,
            self.lid_config,
            self.lang_script,
            self.tgt_lang,
            self.tgt_expected_lid_label,
        )

        self.result_summary = BitextProcessResult(
            src_input_file=src_input_file,
            tgt_input_file=tgt_input_file,
            src_output_file=self.src_output_file,
            tgt_output_file=self.tgt_output_file,
            discarded_src_output_file=self.discarded_src_output_file,
            discarded_tgt_output_file=self.discarded_tgt_output_file,
        )

    # context manager for the TextEncoder which deals with
    # opening and closing the output file the right way

    def __enter__(self):
        self._outf_one = self.src_output_file.open("w", encoding="utf-8")
        self._outf_two = self.tgt_output_file.open("w", encoding="utf-8")

        self._discarded_outf_one = self.discarded_src_output_file.open(
            "w", encoding="utf-8"
        )
        self._discarded_outf_two = self.discarded_tgt_output_file.open(
            "w", encoding="utf-8"
        )
        return self

    def __exit__(self, *exc):
        self._outf_one.close()
        self._outf_two.close()
        self._discarded_outf_one.close()
        self._discarded_outf_two.close()

    def _keep_cleaned(self, cleaned_sentence: str) -> bool:
        return keep_it(
            cleaned_sentence,
            min_chars=self.filter_config.min_chars,
            max_chars=self.filter_config.max_chars,
            max_punct_ratio=self.filter_config.max_punct_ratio,
            max_number_ratio=self.filter_config.max_number_ratio,
            min_space_ratio=self.filter_config.min_space_ratio,
            max_space_ratio=self.filter_config.max_space_ratio,
            max_emoji_ratio=self.filter_config.max_emoji_ratio,
            max_repeated_char=self.filter_config.max_repeated_char,
        )

    def process_lines(
        self, dataset_reader: tp.Generator[DatasetLine, None, None]
    ) -> None:
        """
        process a batch of lines from two corresponding files, filter the lines from the second file (and delete corresponding lines in first file)
        and write them to the output file
        """
        # split sentences
        for (line_id, dataset_line) in enumerate(dataset_reader):
            (real_tgt_line, _tgt_metadata) = extract_distillation_metadata(
                dataset_line.tgt
            )
            (real_src_line, _src_metadata) = extract_distillation_metadata(
                dataset_line.src
            )

            # we throw away metadata, use corpus+offset+linenumber to rebuild it
            self.result_summary.paragraphs += 1

            for (src_line_hash, src_sent, src_clean), (
                tgt_line_hash,
                tgt_sent,
                tgt_clean,
            ) in zip(
                self.src_sentence_split_clean(real_src_line),
                self.tgt_sentence_split_clean(real_tgt_line),
            ):
                self.result_summary.sentences += 1
                if not (self.result_summary.paragraphs % 50_000):
                    print(
                        datetime.now().strftime("%d/%m/%y %H:%M"),
                        self.result_summary.paragraphs,
                        self.result_summary.sentences,
                        sep="\t",
                    )
                filter_logic, pred_lang, prob_lang = self.filter_lid(
                    tgt_clean, self.result_summary
                )
                if filter_logic == FilterLogic.CONTINUE:
                    continue
                elif filter_logic == FilterLogic.DISCARD:
                    print(
                        f"{self.corpus}\t{line_id}\t{pred_lang}\t{prob_lang:.5f}\t{tgt_clean}",
                        file=self._discarded_outf_two,
                    )
                    print(
                        f"{_src_metadata}\t{src_clean}",
                        file=self._discarded_outf_one,
                    )
                    continue
                else:
                    print(
                        # metadata
                        "target_data",
                        self.line_offset_start,  # skip that many lines
                        line_id,  # after skipping, go to line
                        tgt_line_hash,  # xxhash.xxh3_64 of the original line/paragrph
                        f"{prob_lang:.5f}",  # lid score
                        # sentence
                        tgt_clean,
                        # config
                        sep="\t",
                        file=self._outf_two,
                    )
                    print(
                        f"{_src_metadata}\t{src_clean}",
                        file=self._outf_one,
                    )
                    continue
        print("done filtering")

    def final_result(self) -> BitextProcessResult:
        print(f"finished processing to: {self.src_output_file}, {self.tgt_output_file}")
        return self.result_summary

    def merge_results(
        self, splits: tp.List[BitextProcessResult]
    ) -> BitextProcessResult:

        src_merged_discarded_output = Path(self.output_dir) / (
            f"{self.outfile_prefix}discarded.{self.lang_pair}.{self.input_lang}.{self.input_files_idx:03d}.xz"
        )
        tgt_merged_discarded_output = Path(self.output_dir) / (
            f"{self.outfile_prefix}discarded.{self.lang_pair}.{self.tgt_lang}.{self.input_files_idx:03d}.xz"
        )
        src_discard_splits = [str(f.discarded_src_output_file) for f in splits]
        tgt_discard_splits = [str(f.discarded_tgt_output_file) for f in splits]

        cat_splits(src_discard_splits, src_merged_discarded_output)
        cat_splits(tgt_discard_splits, tgt_merged_discarded_output)

        print(
            f"done merging discarded: {src_merged_discarded_output} {tgt_merged_discarded_output}"
        )

        # skipping sorting and deduping of this file (so that lines of both files remain corresponding)
        src_merged_output = Path(self.output_dir) / (
            f"{self.outfile_prefix}tgt_clean.{self.lang_pair}.{self.input_lang}.{self.input_files_idx:03d}"
            + ".xz"
        )
        tgt_merged_output = Path(self.output_dir) / (
            f"{self.outfile_prefix}tgt_clean.{self.lang_pair}.{self.tgt_lang}.{self.input_files_idx:03d}"
            + ".xz"
        )
        print(
            f"merging final results without dedup to: {src_merged_output}, {tgt_merged_output}"
        )

        src_clean_splits = [str(f.src_output_file) for f in splits]
        tgt_clean_splits = [str(f.tgt_output_file) for f in splits]

        cat_splits(src_clean_splits, src_merged_output)
        cat_splits(tgt_clean_splits, tgt_merged_output)

        print(f"done merging: {src_merged_output}. {tgt_merged_output}")

        return BitextProcessResult.merge(
            src_input_file=self.src_input_file,
            tgt_input_file=self.tgt_input_file,
            src_output_file=src_merged_output,
            tgt_output_file=tgt_merged_output,
            discarded_src_output_file=src_merged_discarded_output,
            discarded_tgt_output_file=tgt_merged_discarded_output,
            splits=splits,
        )
