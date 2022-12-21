# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import collections
import contextlib
import dataclasses
import itertools
import logging
import re
import shlex
import subprocess
import tempfile
import typing as tp
from pathlib import Path

import omegaconf
from omegaconf import MISSING, DictConfig

import stopes.core
from stopes.core import utils
from stopes.core.launcher import Launcher
from stopes.modules.evaluation.generate_multi_bleu_detok_module import (
    GenerateMultiBleuDetokConfig,
)
from stopes.modules.preprocess.line_processor import (
    LineProcessorConfig,
    LineProcessorModule,
)
from stopes.modules.preprocess.moses_cli_module import (
    MosesPreprocessConfig,
    MosesPreprocessModule,
    get_moses_script,
)
from stopes.modules.preprocess.train_spm import TrainSpmConfig, TrainSpmModule

if tp.TYPE_CHECKING:
    from stopes.modules.train_fairseq_module import TrainFairseqConfig

log = logging.getLogger("stopes.nmt_bitext_eval")


@dataclasses.dataclass
class ValidAndTestDataDirConfig:
    dataset_name: str = "flores101"
    path: Path = MISSING  # split/lang will be replaced


@dataclasses.dataclass
class NMTBitextEvalConfig:
    # this is the bitext being evaluated
    # It should have 3 columns: (score, src, tgt)
    bitext_tsv: Path
    src_lang: str
    tgt_lang: str
    launcher: tp.Any
    # Pattern to find the test/valid files. eg: flores101/dev/{lang}.dev
    test_data_dir: str
    valid_data_dir: str
    binarize: LineProcessorConfig
    train_spm: TrainSpmConfig
    moses: MosesPreprocessConfig
    moses_filter: "MosesFilterConfig"
    train_fairseq: "TrainFairseqConfig"
    eval: GenerateMultiBleuDetokConfig
    work_dir: Path
    # Bitext with a score lower than this will be removed from training
    bitext_threshold: float = 0.0
    max_tsv_lines: int = 100_000_000
    eval_maximum_epoch: int = 100
    eval_minimum_epoch: int = 50
    public_bitext_base_dir: tp.Optional[Path] = None
    public_corpora_to_ignore: tp.List[str] = dataclasses.field(default_factory=list)
    moses_clean_corpus: bool = True


@dataclasses.dataclass
class ProcessLangMosesInputFiles:
    train: Path
    dev: Path
    test: Path


@dataclasses.dataclass
class ProcessLangMosesOutputFiles:
    train: Path
    dev: Path
    test: Path


@dataclasses.dataclass
class BinarizedDataset:
    lang: str
    spm: Path
    dict_file: Path
    train: Path
    valid: Path
    test: Path


@dataclasses.dataclass
class SplitConcatBitextFilesConfig:
    bitext_tsv: Path
    src_lang: str
    tgt_lang: str
    output_dir: Path
    bitext_threshold: float
    max_tsv_lines: int
    public_bitext_base_dir: tp.Optional[Path]
    public_corpora_to_ignore: tp.List[str]


BitextFiles = tp.List[tp.Tuple[Path, Path]]


class SplitConcatBitextFiles(stopes.core.StopesModule):
    """
    Takes a given bitext file, filter the bitext by score, split it in columnar format,
    and optionally prepend reference bitext for this pair of lang.
    """

    def __init__(self, config: SplitConcatBitextFilesConfig):
        super().__init__(config, SplitConcatBitextFilesConfig)
        self.bitext_tsv = self.config.bitext_tsv
        assert self.bitext_tsv.exists()
        self.output_dir = self.config.output_dir.resolve()
        self.output_dir.mkdir(exist_ok=True)
        self.public_bitext = self.resolve_public_bitext()

    def requirements(self):
        return stopes.core.stopes_module.Requirements(
            mem_gb=5, gpus_per_node=0, cpus_per_task=2, timeout_min=3 * 60
        )

    def cache_key(self):
        """Append the resolved bitext to the cache key as new ones could be added"""
        base_key = super().cache_key()
        return base_key + tuple(self.public_bitext)

    def resolve_public_bitext(self) -> BitextFiles:
        config = self.config
        if not config.public_bitext_base_dir:
            return []
        src = config.src_lang
        tgt = config.tgt_lang
        lang_pair = "-".join(sorted([src, tgt]))
        bitext_dir = Path(config.public_bitext_base_dir) / lang_pair
        # note: this assumes file basenames such as "corpus_name.[...]"
        assert bitext_dir.exists(), f"No public bitext found at {bitext_dir}"
        src_tgt_bitexts = resolve_bitext_files(bitext_dir, src, tgt)
        # Allows to also print warning for the other direction.
        tgt_src_bitexts = resolve_bitext_files(bitext_dir, tgt, src)
        assert src_tgt_bitexts == [
            (s, t) for t, s in tgt_src_bitexts
        ], f"Resolved incompatibles list of bitext files: {src_tgt_bitexts} and {tgt_src_bitexts}."
        # TODO: implement corpus blacklist, check with Kevin the use case
        assert (
            len(config.public_corpora_to_ignore) == 0
        ), "TODO: public_corpora_to_ignore not implemented yet"
        return src_tgt_bitexts

    def run(self, iteration_value: None = None, iteration_index: int = 0):
        threshold = self.config.bitext_threshold
        max_tsv_lines = self.config.max_tsv_lines
        millions = max_tsv_lines // 1_000_000
        prefix = f"bitext{self.sha_key()}_{millions}M_TH-{threshold}"

        output_src = self.output_dir / f"{prefix}.{self.config.src_lang}"
        n_public_src_lines = self.concat_public_bitext(output_src, tgt=False)
        output_tgt = self.output_dir / f"{prefix}.{self.config.tgt_lang}"
        n_public_tgt_lines = self.concat_public_bitext(output_tgt, tgt=True)
        assert (
            n_public_src_lines == n_public_tgt_lines
        ), f"Mismatched public bitext: different number of lines in {output_src} and {output_tgt}"
        n_public_lines = n_public_src_lines
        if n_public_lines > 0:
            log.info(
                f"Found {n_public_lines} / {max_tsv_lines} of public bitex in {self.config.public_bitext_base_dir}"
            )
            if n_public_lines > max_tsv_lines:
                raise RuntimeError(
                    f"Public corpus is already {n_public_lines} lines long, can't evalute bitext with max_tsv_lines={max_tsv_lines}"
                )
        # NOTE: below assumes mined bitext columns are ordered alphabetically by language
        invert_cols = self.config.src_lang > self.config.tgt_lang
        n_tsv_lines = split_mined_tsv(
            self.bitext_tsv,
            output_src if not invert_cols else output_tgt,
            output_tgt if not invert_cols else output_src,
            max_tsv_lines - n_public_lines,
            threshold,
        )
        log.info(
            f"Found {n_tsv_lines} / {max_tsv_lines} of bitex in {self.bitext_tsv} with score >= {threshold}"
        )
        return (output_src, output_tgt)

    def concat_public_bitext(self, output: Path, tgt: bool) -> int:
        """
        write to "output" either the source or target side of the bitext based
        on the value of "tgt", and return the number of lines written
        """
        lines = 0
        with utils.open(output, "wb") as o:
            for (src_file, tgt_file) in self.public_bitext:
                with utils.open(tgt_file if tgt else src_file, "rb") as f:
                    for line in f:
                        lines += 1
                        o.write(line)
        return lines


def resolve_bitext_files(bitext_dir: Path, src: str, tgt: str) -> BitextFiles:
    files = []
    for src_file in bitext_dir.glob(f"*.{src}.*"):
        if not src_file.is_file():
            continue
        tgt_file = bitext_dir / src_file.name.replace(f".{src}.", f".{tgt}.")
        if not tgt_file.is_file():
            log.warn(f"Couldn't find {tgt} file for {src_file}")
            continue
        files.append((src_file, tgt_file))
    return sorted(files)


def split_mined_tsv(
    bitext_tsv: Path,
    output_left: Path,
    output_right: Path,
    max_number_of_lines: int,
    threshold: float,
):
    """
    Takes mined bitext data in tsv file and splits the columns into two files.
    Returns total number of lines that were accepted (score >= threshold)
    """
    tsv_lines = 0
    with contextlib.ExitStack() as stack:
        o_left = stack.enter_context(utils.open(output_left, "a"))
        o_right = stack.enter_context(utils.open(output_right, "a"))
        f_bitext = stack.enter_context(utils.open(bitext_tsv))

        for line in f_bitext:
            score, left, right = line.rstrip("\n").split("\t")
            if float(score) < threshold:
                continue
            print(left, file=o_left)
            print(right, file=o_right)
            tsv_lines += 1

            if tsv_lines >= max_number_of_lines:
                log.warning(
                    f"{bitext_tsv} is very big. Stopping after reading {tsv_lines}."
                )
                break
    return tsv_lines


async def moses_preprocess(
    config: NMTBitextEvalConfig,
    src_file: Path,
    tgt_file: Path,
    launcher: Launcher,
) -> tp.Tuple[Path, Path]:
    moses_config: MosesPreprocessConfig = config.moses
    moses_preprocess_jobs_list = []
    for lang, infile in zip([config.src_lang, config.tgt_lang], [src_file, tgt_file]):
        log.info(f"Lang {lang}: Starting moses_preprocess")
        moses_preprocess_jobs_list.append(
            moses_preprocess_lang(moses_config, infile, launcher, lang)
        )

    moses_src_output, moses_tgt_output = await asyncio.gather(
        *moses_preprocess_jobs_list
    )

    if not config.moses_clean_corpus:
        return moses_src_output, moses_tgt_output

    assert (
        moses_src_output.suffix == f".{config.src_lang}"
        and moses_tgt_output.suffix == f".{config.tgt_lang}"
    ), (
        "moses corpus cleaning expects input files to have lang suffix."
        + f"Inputs given: {moses_src_output} and {moses_tgt_output}"
    )
    # basename for both input files ending in lang suffix
    moses_clean_input_basename = moses_src_output.with_suffix("")
    moses_src_output_clean, moses_tgt_output_clean = moses_clean_corpus(
        f"{config.src_lang}",
        f"{config.tgt_lang}",
        config.moses_filter,
        moses_clean_input_basename,
    )
    return moses_src_output_clean, moses_tgt_output_clean


async def spm_train_encode_binarize(
    lang_pair: str,
    lang: str,
    infile: Path,
    launcher: Launcher,
    config: NMTBitextEvalConfig,
) -> BinarizedDataset:
    log.info(f"Lang {lang}: Starting SPM Train")

    with utils.clone_config(config.train_spm) as spm_config:
        spm_config.train_data_file = str(infile)
        spm_config.model_prefix_spm = f"spm.{spm_config.vocab_size}.{lang}"

    spm_vocab = await launcher.schedule(TrainSpmModule(spm_config))
    spm, dict_file = spm_vocab.model_file, spm_vocab.dict_file
    log.info(f"Lang {lang}: trained SentencePiece model {spm} with {dict_file} dict.")

    binarize_jobs = []
    raw_files = {
        "train": infile,
        "valid": Path(config.valid_data_dir.format(lang=lang)),
        "test": Path(config.test_data_dir.format(lang=lang)),
    }
    for split, infile in raw_files.items():
        # outfile_prefix's naming scheme uses train/valid/test to match faireq defaults.
        # TODO: couldn't we use --valid_subset here ?
        assert infile.exists(), f"{split} data not found at {infile}"
        outfile_prefix = f"{split}.{lang_pair}.{lang}"
        binarize_jobs.append(
            binarize_lang(
                infile,
                dict_file,
                spm,
                outfile_prefix,
                config.binarize,
                launcher,
            )
        )
    binarized_output_files = await asyncio.gather(*binarize_jobs)
    log.info(f"Lang {lang}: binarized {binarized_output_files} with {spm}")

    return BinarizedDataset(lang, spm, dict_file, *binarized_output_files)


async def moses_preprocess_lang(
    moses_config: MosesPreprocessConfig,
    moses_preprocess_input_file: Path,
    launcher: Launcher,
    lang: str,
):
    with utils.clone_config(moses_config) as cfg:
        cfg.shards = [str(moses_preprocess_input_file)]
        cfg.lang = lang

    moses_module = MosesPreprocessModule(cfg)
    moses_preprocess_output_shards = await launcher.schedule(moses_module)
    return moses_preprocess_output_shards[0]


@dataclasses.dataclass
class MosesFilterConfig:
    output_dir: Path = MISSING
    # bitexts filtering using Moses's clean-corpus-n.perl
    filter_ratio: float = 2.5
    filter_min: float = 1
    # removes lines of more than 250 in length
    filter_max: float = 250


def moses_clean_corpus(
    src_lang: str,
    tgt_lang: str,
    config: MosesFilterConfig,
    corpus_prefix: Path,
) -> tp.Tuple[Path, Path]:
    """
    Run moses "clean-corpus-n.perl" which looks at bitext and has some heuristics to clean it,
    based on each side sentence length.
    """
    # TODO: do we need moses ? we already have enough cleaning code,
    # and this could be pretty bad for some languages.
    # If no, we could also compress intermediary files
    clean_corpus_perl = get_moses_script("scripts/training/clean-corpus-n.perl")
    moses_filter_output_file = config.output_dir / (corpus_prefix.name + ".clean")
    raw_cmd: list = [
        "perl",
        clean_corpus_perl,
        "-ratio",
        config.filter_ratio,
        corpus_prefix,
        src_lang,
        tgt_lang,
        moses_filter_output_file,
        config.filter_min,
        config.filter_max,
    ]
    cmd = [str(c) for c in raw_cmd]
    log.info(shlex.join(cmd))
    subprocess.run(cmd, check=True)
    return (
        moses_filter_output_file.with_suffix(f".clean.{src_lang}"),
        moses_filter_output_file.with_suffix(f".clean.{tgt_lang}"),
    )


def determine_valid_test_bleu(results: dict):
    """
    if we have results for both test and valid, return best validation score
    and test result for corresponding checkpoint. If we don't have results for
    both valid and test then return the max result for whichever split we have
    """
    best_valid, test_bleu = None, None
    if "valid" in results and "test" in results:
        # choose test bleu based on best valid
        assert len(results["valid"]) == len(
            results["test"]
        ), "Size mismatch between valid and test files; please inspect outputs"
        best_valid = max(results["valid"], key=lambda x: x["bleu"])
        for test_result in results["test"]:
            if test_result["ckpt"].replace("_test", "_valid") == best_valid["ckpt"]:
                return best_valid, test_result
    elif "valid" in results:
        best_valid = max(results["valid"], key=lambda x: x["bleu"])
    else:
        # choose max test bleu (not recommended)
        test_bleu = max(results["test"], key=lambda x: x["bleu"])
    return best_valid, test_bleu


def parse_generated_bleu_file(input_file: Path) -> float:
    file_content = input_file.read_text()
    try:
        match = re.search("(?<=^BLEU = )[^,]+", file_content)
        assert match
        return float(match.group())
    except Exception:
        raise Exception(f"Invalid BLEU file: {input_file}: {file_content[:500]}")


def parse_generated_bleu_files(
    files: tp.List[Path],
) -> tp.Dict[str, list]:
    results: tp.Dict[str, list] = collections.defaultdict(list)
    for file in files:
        match = re.search(".*_(valid|test).bleu$", str(file))
        assert match, f"(valid|test) not found in {file}"
        split = match.group(1)
        bleu = parse_generated_bleu_file(file)
        results[split].append({"ckpt": str(file), "bleu": bleu})
    return results


async def binarize_lang(
    input_file: Path,
    spm_vocab_file: Path,
    spm_model_file: Path,
    outfile_prefix: str,
    binarizer_config: LineProcessorConfig,
    launcher: Launcher,
):
    with utils.clone_config(binarizer_config) as cfg:
        cfg.outfile_prefix = outfile_prefix
        cfg.shards = [str(input_file)]
        cfg.line_processor.spm_model_path = str(spm_model_file)
        cfg.line_processor.vocab_file_path = str(spm_vocab_file)

    binarized_shards = await launcher.schedule(
        LineProcessorModule(cfg, validate_config=True)
    )
    return binarized_shards[0]
