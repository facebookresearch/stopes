# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import contextlib
import itertools
import logging
import re
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import omegaconf
from fairseq.dataclass.configs import FairseqConfig
from omegaconf import MISSING, DictConfig

import stopes.core
from stopes.core.launcher import SubmititLauncher
from stopes.core.utils import open as stopes_open
from stopes.modules.evaluation.generate_multi_bleu_detok_module import (
    GenerateMultiBleuDetokConfig,
)
from stopes.modules.preprocess.line_processor import LineProcessorModule
from stopes.modules.preprocess.moses_cli_module import (
    MosesPreprocessConfig,
    MosesPreprocessModule,
    get_moses_script,
)
from stopes.modules.preprocess.train_spm import TrainSpmConfig, TrainSpmModule


@dataclass
class ValidAndTestDataDirConfig:
    dataset_name: str = "flores101"
    path: Path = MISSING  # split/lang will be replaced


@dataclass
class PreprocBinarizedMinedConfig:
    version: str = "V0"
    eval: str = "flores101"
    max_number_of_tsv_lines_in_millions: int = 100  # max number of lines of the TSV being read for tsv split; corresponds to variable 'SZ' in bash scripts
    bitext_alignment_minimum_score_threshold: float = 1.06  # default bitext alignment threshold when splitting tsv (only accepts lines with score above this value); corresponds to variable "threshold" in bash scripts
    number_workers: int = 20
    test_data_dir: ValidAndTestDataDirConfig = ValidAndTestDataDirConfig()


@dataclass
class NMTBitextEvalConfig:
    launcher: DictConfig
    preproc_binarize_mined: PreprocBinarizedMinedConfig
    binarize: DictConfig
    spm: TrainSpmConfig
    moses: MosesPreprocessConfig
    train_fairseq: FairseqConfig
    eval: GenerateMultiBleuDetokConfig
    output_dir: Path
    spm_train_and_binarize_output_dir: Path
    bin_dir: Path
    eval_maximum_epoch: int = 100
    eval_minimum_epoch: int = 1  # must be <= eval_maximum_epoch
    src_lang: str = MISSING
    tgt_lang: str = MISSING
    input_file_mined_data_tsv: Path = MISSING  # this is the input file to the 3 scripts


@dataclass
class ProcessLangMosesInputFiles:
    train: Path
    dev: Path
    test: Path


@dataclass
class ProcessLangMosesOutputFiles:
    train: Path
    dev: Path
    test: Path


@dataclass
class BinarizeLangOutputFiles:
    train: Path
    valid: Path
    test: Path


def split_to_mono(
    input_file_mined_data_tsv: str,
    max_tsv_lines_in_millions: int,
    bitext_min_score_threshold: float,
    bin_dir: str,
    src_lang: str,
    tgt_lang: str,
    public_bitext_base_dir: Optional[str] = None,
    public_bitext_margin: float = 2.0,
    corpora_to_ignore: List[str] = [],
) -> List[Path]:
    """
    (optionally) prepends public corpora to a mined TSV bitext
    and then splits the bitext into src and tgt monolingual files
    """
    src_lang_file_after_split: Path = Path(bin_dir) / "_".join(
        [
            f"split_tsv_{max_tsv_lines_in_millions}M",
            f"TH-{bitext_min_score_threshold}.{src_lang}",
        ]
    )
    tgt_lang_file_after_split: Path = Path(bin_dir) / "_".join(
        [
            f"split_tsv_{max_tsv_lines_in_millions}M",
            f"TH-{bitext_min_score_threshold}.{tgt_lang}",
        ]
    )
    with tempfile.NamedTemporaryFile() as tmpfile:
        combined_bitext = prepend_public_bitexts(
            tmpfile.name,
            input_file_mined_data_tsv,
            src_lang,
            tgt_lang,
            public_bitext_base_dir,
            public_bitext_margin,
            corpora_to_ignore,
        )
        bitext_file_left, bitext_file_right = sorted(
            [src_lang_file_after_split, tgt_lang_file_after_split]
        )  # assumes bitext TSV columns stored in sorted lang-pair
        split_TSV(
            bitext_file_left,
            bitext_file_right,
            combined_bitext,
            max_tsv_lines_in_millions * 1_000_000,
            bitext_min_score_threshold,
        )
    return src_lang_file_after_split, tgt_lang_file_after_split


def prepend_public_bitexts(
    combined_outfile: str,
    mined_bitext: str,
    src_lang: str,
    tgt_lang: str,
    public_bitext_base_dir: str = None,
    margin: float = 2.0,
    corpora_to_ignore: List[str] = [],
):
    if not public_bitext_base_dir:
        return mined_bitext
    bitext_dir = Path(public_bitext_base_dir) / "-".join(sorted([src_lang, tgt_lang]))
    bitexts = sorted(
        [
            str(fn)
            for fn in bitext_dir.iterdir()
            if fn.is_file() and fn.name.split(".")[0] not in corpora_to_ignore
        ]
    )  # note: this assumes file basenames such as "corpus_name.[...]"
    bitext_pairs = zip(bitexts[::2], bitexts[1::2])
    with stopes_open(combined_outfile, mode="w") as outf:
        for src_bitext, tgt_bitext in bitext_pairs:
            with stopes_open(src_bitext) as src_file, stopes_open(
                tgt_bitext
            ) as tgt_file:
                for src, tgt in zip(src_file, tgt_file):
                    src, tgt = src.rstrip("\n"), tgt.rstrip("\n")
                    print("\t".join([str(margin), src, tgt]), file=outf)
        for line in stopes_open(mined_bitext):
            print(line, end="", file=outf)
    return combined_outfile


def split_TSV(
    file_left: Path,
    file_right: Path,
    input_file_mined_data_tsv: Path,
    max_number_of_tsv_lines: int,
    bitext_alignment_minimum_score_threshold: float,
):
    """
    Takes input mined data in tsv file (whether gzipped/xzipped/or not zipped) and splits into two files: file_left and file_right.
    Returns total number of lines that were accepted (score > bitext_alignment_minimum_score_threshold)
    Note: assumes that input_file_mined_data_tsv is sorted (descending order) by the scores (first column) of each line
    """
    total_accepted_lines = 0
    with open(file_left, "w", encoding="utf-8") as lang1_out, open(
        file_right, "w", encoding="utf-8"
    ) as lang2_out, stopes_open(input_file_mined_data_tsv) as filep:
        for line in itertools.islice(filep, max_number_of_tsv_lines):
            (score, lang1, lang2) = line.rstrip("\n").split("\t")

            if float(score) >= bitext_alignment_minimum_score_threshold:
                total_accepted_lines += 1
                print(lang1, file=lang1_out)
                print(lang2, file=lang2_out)
    return total_accepted_lines


async def spm_train_module_call(
    spm_config: TrainSpmConfig,
    train_data_file: Path,
    launcher: SubmititLauncher,
    output_dir: str,
):
    with clone_config(spm_config) as edited_spm_config:
        edited_spm_config.train_data_file = str(train_data_file)
        edited_spm_config.output_dir = str(output_dir)

    spm_train_module = TrainSpmModule(edited_spm_config)
    model_file, vocab_as_fairseq_dict = await launcher.schedule(spm_train_module)

    return model_file, vocab_as_fairseq_dict


async def moses_preprocess(
    config: NMTBitextEvalConfig,
    src_file: Path,
    tgt_file: Path,
    launcher: SubmititLauncher,
) -> Tuple[Path, Path]:
    logger = logging.getLogger("moses_preprocessing")
    logger.info("Starting Moses Pre-processing")
    moses_config: MosesPreprocessConfig = config.moses.config

    moses_preprocess_jobs_list = []
    for lang, infile in zip([config.src_lang, config.tgt_lang], [src_file, tgt_file]):
        moses_preprocess_jobs_list.append(
            moses_preprocess_lang(
                config.bin_dir,
                moses_config,
                infile,
                launcher,
                lang,
            )
        )

    moses_src_output, moses_tgt_output = await asyncio.gather(
        *moses_preprocess_jobs_list
    )

    assert (
        moses_src_output.suffix == f".{config.src_lang}"
        and moses_tgt_output.suffix == f".{config.tgt_lang}"
    ), (
        "moses corpus cleaning expects input files to have lang suffix."
        + f"Inputs given: {moses_src_output} and {moses_tgt_output}"
    )
    # basename for both input files ending in lang suffix
    moses_clean_input_basename = moses_src_output.with_suffix("")
    moses_src_output_clean, moses_tgt_output_clean = moses_corpus_clean(
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
    launcher: SubmititLauncher,
    config: NMTBitextEvalConfig,
) -> BinarizeLangOutputFiles:
    logger = logging.getLogger("process_lang")
    logger.info("Starting SPM Train")

    model_file, dict_file = await spm_train_module_call(
        config.spm.train.config,
        infile,
        launcher,
        config.spm_train_and_binarize_output_dir,
    )
    assert isinstance(model_file, Path)
    assert isinstance(dict_file, Path)

    symlinked_dict_file = (
        Path(config.spm_train_and_binarize_output_dir) / f"dict.{lang}.txt"
    )
    stopes.core.utils.symlink(symlinked_dict_file, dict_file)

    if (
        not Path(config.spm_train_and_binarize_output_dir).resolve()
        == model_file.parent.resolve()
    ):
        # will enter this case if spm train is cached, since the two resolved dir's are different
        # symlink is done below to create a copy of model file from cached output dir into spm_train_and_binarize_output_dir
        symlinked_model_file = (
            Path(config.spm_train_and_binarize_output_dir) / model_file.name
        )
        stopes.core.utils.symlink(symlinked_model_file, model_file)
        model_file = symlinked_model_file

    logger.info(
        f"SPM Training's model file and dict file are respectively: \n\t -{model_file} \n\t -{dict_file}"
    )

    logger.info("Starting SPM Encode and Binarize")
    binarize_lang_jobs_list = []
    for split in ("train", "dev", "devtest"):
        if split != "train":  # infile is initially train
            infile = Path(
                config.preproc_binarize_mined.test_data_dir.path.format(
                    split=split, lang=lang
                )
            )

        # necessary to change terminology due to naming difference with flores dataset
        if split == "dev":
            split = "valid"
        elif split == "devtest":
            split = "test"

        # outfile_prefix's naming scheme must strictly be followed; hardcoded requirement by train fairseq module
        outfile_prefix = f"{split}.{lang_pair}.{lang}"
        binarize_lang_jobs_list.append(
            binarize_lang(
                infile,
                dict_file,
                model_file,
                outfile_prefix,
                config.binarize.config,
                launcher,
            )
        )
    binarized_output_files = await asyncio.gather(*binarize_lang_jobs_list)
    binarized_lang_outputs = BinarizeLangOutputFiles(
        train=binarized_output_files[0],
        valid=binarized_output_files[1],
        test=binarized_output_files[2],
    )
    return binarized_lang_outputs


async def moses_preprocess_lang(
    bin_dir: Path,
    moses_config: MosesPreprocessConfig,
    moses_preprocess_input_file: Path,
    launcher: SubmititLauncher,
    lang: str,
):
    # only use moses preprocessing on train data, not dev or test
    moses_output_dir = str(Path(bin_dir) / "moses_preprocess_train_data")

    with clone_config(moses_config) as edited_moses_config:
        edited_moses_config.shards = [str(moses_preprocess_input_file)]
        edited_moses_config.output_dir = moses_output_dir
        edited_moses_config.lang = lang

    moses_module = MosesPreprocessModule(edited_moses_config)
    moses_preprocess_output_shards = await launcher.schedule(moses_module)
    moses_preprocess_output_file = moses_preprocess_output_shards[0]
    return moses_preprocess_output_file


# TODO change into a module and integrate moses filter step into nmt_bitext_eval script
@dataclass
class MosesCorpusCleanConfig:
    output_dir: Path = MISSING
    filter_ratio: float = 2.5  # bitexts filtering using Moses's lean-corpus-n.perl
    filter_min: float = 1
    filter_max: float = 250  # removes lines of more than 250 in length


def moses_corpus_clean(
    src_lang: str,
    tgt_lang: str,
    moses_filter_config: MosesCorpusCleanConfig,
    spm_encode_output_file_prefix_before_lang_name: str,
) -> Tuple[Path, Path]:
    moses_filter_file = get_moses_script("scripts/training/clean-corpus-n.perl")
    # Note: spm_encode_output_file_prefix_before_lang_name is the path to the spm encode output file excluding the last 4 characters which are the lang (excluding, e.g. ".ben")
    logger = logging.getLogger("moses_corpus_clean")

    filter_ratio = str(moses_filter_config.filter_ratio)
    corpus = spm_encode_output_file_prefix_before_lang_name
    filter_min = str(moses_filter_config.filter_min)
    filter_max = str(moses_filter_config.filter_max)
    moses_filter_output_file = Path(
        Path(moses_filter_config.output_dir) / "moses_filtered_output"
    )
    moses_filter_log_file = Path(
        Path(moses_filter_config.output_dir) / "both_langs_moses_filter.logs"
    )

    try:
        with open(moses_filter_log_file, "w") as log:
            subprocess.run(
                [
                    "perl",
                    moses_filter_file,
                    "-ratio",
                    filter_ratio,
                    corpus,
                    src_lang,
                    tgt_lang,
                    moses_filter_output_file,
                    filter_min,
                    filter_max,
                ],
                stderr=log,
                check=True,
            )
    except Exception as e:
        logger.exception(
            f"Error during distance moses_filter_bash_command, see moses_filter_log_file: {moses_filter_log_file}."
        )
        raise e
    return (
        moses_filter_output_file.with_suffix(f".{src_lang}"),
        moses_filter_output_file.with_suffix(f".{tgt_lang}"),
    )


def parse_generated_bleu_file(input_file: str) -> float:
    with open(input_file, "r") as infile:
        result_line = infile.readline()
        # expected format: BLEU = [score],
        bleu = float(re.search("(?<=^BLEU = )[^,]+", result_line).group())
    return bleu


def parse_generated_bleu_files(input_files: List[Path]) -> dict:
    results = {}
    for input_file in input_files:
        split = re.search(".*_(valid|test).bleu", str(input_file)).group(1)
        if split not in results:
            results[split] = []
        bleu = parse_generated_bleu_file(input_file)
        results[split].append(bleu)
    return results


async def binarize_lang(
    input_file: Path,
    spm_vocab_file: Path,
    spm_model_file: Path,
    outfile_prefix: str,
    fairseq_binarizer_encoder_config: DictConfig,
    launcher: SubmititLauncher,
):
    with clone_config(
        fairseq_binarizer_encoder_config
    ) as edited_fairseq_binarizer_config:
        edited_fairseq_binarizer_config.spm_model_path = str(spm_model_file)
        edited_fairseq_binarizer_config.outfile_prefix = outfile_prefix
        edited_fairseq_binarizer_config.shards = [str(input_file)]
        edited_fairseq_binarizer_config.vocab_file_path = str(spm_vocab_file)

    binarize_module = LineProcessorModule(edited_fairseq_binarizer_config)
    binarized_shards = await launcher.schedule(binarize_module)
    return binarized_shards[0]


@contextlib.contextmanager
def clone_config(config: DictConfig):
    with omegaconf.open_dict(config.copy()) as cfg:
        omegaconf.OmegaConf.set_readonly(cfg, False)
        yield cfg
