# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import shlex
import subprocess
import typing as tp
import unicodedata
from dataclasses import dataclass, field, fields
from datetime import datetime
from enum import Enum
from pathlib import Path

import xxhash
from sacremoses import MosesPunctNormalizer

from stopes.core import utils
from stopes.modules.preprocess.multiproc_line_processor import (
    MultiprocLineProcessorCallback,
)
from stopes.pipelines.monolingual.utils import remove_regex, slurm_tmp_maybe
from stopes.pipelines.monolingual.utils.predict_lid import (
    get_lid_predictor,
    get_lid_predictor_date,
)
from stopes.pipelines.monolingual.utils.predict_script import get_script_predictor
from stopes.pipelines.monolingual.utils.sentence_split import get_split_algo
from stopes.pipelines.monolingual.utils.sort import build_sort_command
from stopes.pipelines.monolingual.utils.text_filter import keep_it


class FilterLogic(Enum):
    """
    Three types of logic for filtering.
    """

    DISCARD = "Discard"
    CONTINUE = "Continue"
    WRITE = "Write"


@dataclass
class FilterConfig:
    min_chars: int = 10
    max_chars: int = 500
    max_punct_ratio: float = 0.2
    max_number_ratio: float = 0.2
    min_space_ratio: float = 0.03
    max_space_ratio: float = 0.5
    script_threshold: float = 0.5
    max_emoji_ratio: float = 0.2
    max_repeated_char: float = 10
    tab_index: int = 0


@dataclass
class LIDConfig:
    label_unk: str = "__label__unk"
    latest_models_path: str = None
    # model_date or model_file is required
    model_date: tp.Optional[str] = None
    model_file: tp.Optional[str] = None
    thresholds_file: tp.Optional[str] = None
    probability_threshold: float = 0.5
    lang_thresholds: tp.Dict[str, int] = field(default_factory=lambda: {})


@dataclass
class MonolingualProcessResult:
    input_file: Path
    output_file: Path
    discarded_output_file: Path
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
        input_file: Path,
        output_file: Path,
        discarded_output_file: Path,
        splits: tp.Iterable["MonolingualProcessResult"],
    ) -> "MonolingualProcessResult":
        ret = MonolingualProcessResult(input_file, output_file, discarded_output_file)
        for f in fields(ret):
            if f.name in {"input_file", "output_file", "discarded_output_file"}:
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


def firstcolumn_metadata_splitter(line: str) -> tp.Tuple[str, str]:
    """
    returns a tuple with first the paragraph content, then the metadata to keep.

    takes the content in the first column, everything else is metadata
    """
    line_parts = line.split("\t", maxsplit=1)
    if len(line_parts) > 1:
        return (line_parts[0], line_parts[1])
    return (line_parts[0], "")


def cc200xl_metadata_splitter(line: str) -> tp.Tuple[str, str]:
    """
    returns a tuple with first the paragraph content, then the metadata to keep.

    common crawl format is: content \t metadata metadata metadata
    """
    return firstcolumn_metadata_splitter(line)


# same as above, but keep it separate in case it evolves
def ia_metadata_splitter(line: str) -> tp.Tuple[str, str]:
    """
    returns a tuple with first the paragraph content, then the metadata to keep.

    internet archive format is: content \t metadata metadata metadata
    """
    return firstcolumn_metadata_splitter(line)


def paracrawl_metadata_splitter(line: str) -> tp.Tuple[str, str]:
    """
    returns a tuple with first the paragraph content, then the metadata to keep.

    paracrawl sometimes has the lid score as the first column
    """
    line_parts = line.split("\t", maxsplit=1)
    if len(line_parts) > 1:
        # inverted from the other cases, content is last block
        return (line_parts[1], line_parts[0])
    # some cases do not have the lid score
    return (line_parts[0], "")


def extract_metadata(line: str, corpus: str) -> tp.Tuple[str, str]:
    """
    returns a tuple with first the paragraph content, then the metadata to keep.
    """

    if corpus.startswith("cc200xl"):
        return cc200xl_metadata_splitter(line)
    if corpus.startswith("ia2022"):
        return ia_metadata_splitter(line)
    if corpus.startswith("paracrawl"):
        return paracrawl_metadata_splitter(line)
    else:
        return firstcolumn_metadata_splitter(line)


def cat_splits(splits: tp.List[str], output_file: Path):
    subprocess.run(
        utils.bash_pipefail(
            shlex.join(["cat"] + splits),
            " ".join(["xz", ">", shlex.quote(str(output_file))]),
        ),
        shell=True,
        check=True,
    )


def sort_splits(splits: tp.List[str], output_file: Path, tmp_dir, num_cpu):
    sort_cmd = build_sort_command(
        files=splits,
        num_cpu=num_cpu,
        is_merge=False,
        tmp_dir=tmp_dir,
    )
    subprocess.run(
        utils.bash_pipefail(
            sort_cmd, " ".join(["xz", ">", shlex.quote(str(output_file))])
        ),
        shell=True,
        check=True,
    )


class SentenceSplitClean:
    def __init__(self, splitter_lang: str, split_algo: str):
        # setup sentence splitter
        self.splitter = get_split_algo(splitter_lang, split_algo=split_algo)

        # setup "moses" normalization
        self.mpn = MosesPunctNormalizer(lang="en")
        self.mpn.substitutions = [
            (re.compile(r), sub) for r, sub in self.mpn.substitutions
        ]
        self.replace_nonprint = remove_regex.get_non_printing_char_replacer(" ")

    def __call__(self, line):
        sentence_splits = self.splitter(line)
        line_hash = xxhash.xxh3_64_intdigest(line)

        for sent in sentence_splits:
            # normalize -- moses equivalent
            clean = self.mpn.normalize(sent)
            clean = self.replace_nonprint(clean)
            # replace 𝓕𝔯𝔞𝔫𝔠𝔢𝔰𝔠𝔞 by Francesca
            clean = unicodedata.normalize("NFKC", clean)

            yield (line_hash, sent, clean)


class FilterLID:
    def __init__(
        self, filter_config, lid_config, lang_script, lang, expected_lid_label
    ):
        self.filter_config = filter_config
        self.replace_url = remove_regex.get_url_replacer(" ")
        self.replace_hashtag = remove_regex.get_ascii_hashtag_replacer(" ")
        self.script_predictor = get_script_predictor()
        self.lang_script = lang_script
        # setup LID
        self.lid_threshold = getattr(
            lid_config.lang_thresholds, lang, lid_config.probability_threshold
        )
        if hasattr(lid_config, "model_file") and lid_config.model_file is not None:
            thresholds_file = getattr(lid_config, "thresholds_file", None)
            self.lid_predictor = get_lid_predictor(
                model_file=Path(lid_config.model_file),
                thresholds_file=Path(thresholds_file)
                if thresholds_file is not None
                else None,
                label_unk=lid_config.label_unk,
            )
        else:
            self.lid_predictor = get_lid_predictor_date(
                model_date=lid_config.model_date,
                label_unk=lid_config.label_unk,
                lid_latest_models_path=Path(lid_config.latest_models_path),
            )
        self.expected_lid_label = expected_lid_label

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

    def __call__(self, clean: str, result_summary):
        # filter
        if not self._keep_cleaned(clean):
            result_summary.filtered += 1
            return FilterLogic.CONTINUE, None, None

        # remove hashtags and urls for extra filtering and lid/script prediction
        prediction_clean = clean
        prediction_clean = self.replace_url(prediction_clean)
        prediction_clean = self.replace_hashtag(prediction_clean)

        if not self._keep_cleaned(prediction_clean):
            result_summary.extra_filtered += 1
            return FilterLogic.CONTINUE, None, None

        # filter script
        (script, script_score) = self.script_predictor(prediction_clean)
        if script_score < self.filter_config.script_threshold:
            result_summary.script_filtered += 1
            return FilterLogic.CONTINUE, None, None

        if script != self.lang_script:
            result_summary.script_mismatch += 1
            return FilterLogic.CONTINUE, None, None

        # lid
        pred_lang, prob_lang = self.lid_predictor(prediction_clean)
        if prob_lang < self.lid_threshold:
            result_summary.lid_filtered += 1
            return FilterLogic.DISCARD, pred_lang, prob_lang

        if pred_lang != self.expected_lid_label:
            result_summary.lid_mismatch += 1
            return FilterLogic.DISCARD, pred_lang, prob_lang

        # all filters pass, keep the line
        result_summary.kept += 1
        return FilterLogic.WRITE, pred_lang, prob_lang


class SplitNormalizeFilterLID(MultiprocLineProcessorCallback):
    def __init__(
        self,
        # set by LineProcessorModule
        outfile_prefix: str,
        input_file: str,
        input_file_idx: int,
        output_dir: str,
        offset_start: tp.Optional[int],
        offset_end: tp.Optional[int],
        merging: bool,
        # set by the config
        lang: str,
        split_algo: str,
        filter_config: FilterConfig,
        lid_config: LIDConfig,
        lang_script: str,
        splitter_lang: str,
        num_cpu: int,
        local_tmp_dir: str,
        _version: str,  # used to bump config version and invalidate cache
        skip_dedup: bool = False,
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

        self.lang = lang
        self.splitter_lang = splitter_lang
        self.lid_config = lid_config
        self.filter_config = filter_config
        self.split_algo = split_algo

        self.local_tmp_dir = local_tmp_dir

        self.num_cpu = num_cpu

        self.skip_dedup = skip_dedup

        self.expected_lid_label = f"__label__{self.lang}"
        self.lang_script = lang_script
        # find out the corpus from basename: e.g. cc200xl_v1.ewe.xz or cc200xl_v1.kat.split.ca
        input_parts = os.path.basename(input_file).split(".")
        self.corpus = input_parts[0]

        self.offset_start = offset_start

        self.output_file = Path(self.output_dir) / (
            f"{self.outfile_prefix}{self.corpus}.{self.lang}.{input_file_idx:03d}.{offset_start}_{offset_end}.tsv"
        )

        print(f"tmp output to file: {self.output_file}")

        self.discarded_output_file = Path(self.output_dir) / (
            f"{self.outfile_prefix}_discarded.{self.corpus}.{self.lang}.{input_file_idx:03d}.{offset_start}_{offset_end}.tsv"
        )
        print(f"tmp discarded output to file: {self.discarded_output_file}")

        self.sentence_split_clean = SentenceSplitClean(splitter_lang, split_algo)

        self.filter_lid = FilterLID(
            self.filter_config,
            self.lid_config,
            self.lang_script,
            self.lang,
            self.expected_lid_label,
        )

        self.result_summary = MonolingualProcessResult(
            input_file=input_file,
            output_file=self.output_file,
            discarded_output_file=self.discarded_output_file,
        )

    # context manager for the TextEncoder which deals with
    # opening and closing the output file the right way

    def __enter__(self):
        self._outf = self.output_file.open("w", encoding="utf-8")
        self._discarded_outf = self.discarded_output_file.open("w", encoding="utf-8")
        return self

    def __exit__(self, *exc):
        self._outf.close()
        self._discarded_outf.close()

    def process_lines(self, lines_with_number: tp.Iterator[tp.Tuple[int, str]]) -> None:
        """
        process a batch of lines, filter them, dedup them locally
        and write them to the output file
        """
        # split sentences
        for (line_id, line) in lines_with_number:
            (real_line, _metadata) = extract_metadata(line, self.corpus)
            # we throw away metadata, use corpus+offset+linenumber to rebuild it
            self.result_summary.paragraphs += 1

            for line_hash, sent, clean in self.sentence_split_clean(real_line):
                self.result_summary.sentences += 1
                if not (self.result_summary.paragraphs % 50_000):
                    print(
                        datetime.now().strftime("%d/%m/%y %H:%M"),
                        self.result_summary.paragraphs,
                        self.result_summary.sentences,
                        sep="\t",
                    )

                filter_logic, pred_lang, prob_lang = self.filter_lid(
                    clean, self.result_summary
                )
                if filter_logic == FilterLogic.CONTINUE:
                    continue
                elif filter_logic == FilterLogic.DISCARD:
                    print(
                        f"{self.corpus}\t{line_id}\t{pred_lang}\t{prob_lang:.5f}\t{sent}",
                        file=self._discarded_outf,
                    )
                    continue
                else:
                    print(
                        # metadata
                        self.corpus,  # the original corpus name
                        self.offset_start,  # skip that many bytes (use dd)
                        line_id,  # after skipping, go to line
                        line_hash,  # xxhash.xxh3_64 of the original line/paragrph
                        f"{prob_lang:.5f}",  # lid score
                        # sentence
                        clean,
                        # config
                        sep="\t",
                        file=self._outf,
                    )
                    continue
        print("done filtering")

    def final_result(self) -> MonolingualProcessResult:
        print(f"finished processing to: {self.output_file}")
        return self.result_summary

    def merge_results(
        self, splits: tp.List[MonolingualProcessResult]
    ) -> MonolingualProcessResult:

        merged_discarded_output = Path(self.output_dir) / (
            f"{self.outfile_prefix}_discarded.{self.corpus}.{self.lang}.{self.input_file_idx:03d}.sorted.xz"
        )

        discard_splits = [str(f.discarded_output_file) for f in splits]
        cat_splits(discard_splits, merged_discarded_output)
        print(f"done merging discarded: {merged_discarded_output}")

        # sort and dedup this file based on skip_dedup flag
        sort_suffix = "unsorted" if self.skip_dedup else "sorted"
        sorted_output = Path(self.output_dir) / (
            f"{self.outfile_prefix}{self.corpus}.{self.lang}.{self.input_file_idx:03d}"
            f".{sort_suffix}.xz"
        )
        if self.skip_dedup:
            print("skipping local sorting")
            sort_files = [str(f.output_file) for f in splits]
            cat_splits(sort_files, sorted_output)
        else:
            print("sorting locally")
            tmp_dir = slurm_tmp_maybe(Path(self.local_tmp_dir))
            sort_files = [s.output_file for s in splits]

            print(f"starting merging: {sorted_output}")
            sort_splits(sort_files, sorted_output, tmp_dir, self.num_cpu)
            print(f"done merging: {sorted_output}")

        return MonolingualProcessResult.merge(
            input_file=self.input_file,
            output_file=sorted_output,
            discarded_output_file=merged_discarded_output,
            splits=splits,
        )
