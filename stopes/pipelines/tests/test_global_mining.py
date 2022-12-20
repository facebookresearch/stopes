# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import hydra
import numpy as np
import pytest
from omegaconf import OmegaConf

from stopes.core import utils
from stopes.modules.preprocess.encode_to_npy import EncodeToNPY
from stopes.pipelines.bitext.global_mining_pipeline import main as mining_main
from stopes.utils.mining_utils import extract_shard_id

DIGIT_NAMES = {
    "en": "zero one two three four five six seven eight nine".split(),
    "fr": "zéro un deux trois quatre cinq six sept huit neuf".split(),
}


def num2digits(number: int) -> tp.List[int]:
    """Convert a number (e.g. 123) to a list of its digits (e.g. [1, 2, 3])"""
    return [int(i) for i in str(number)]


def create_digits(
    dir: Path,
    lang: str,
    n_min: int = 0,
    n_max: int = 1000,
    suffix: str = "",
    use_meta: bool = True,
):
    """Create a file with n text representations of digits and a corresponding meta file."""
    with open(dir / f"{lang}{suffix}_text.tsv", "w") as data_file:
        for i in range(n_min, n_max):
            digits = num2digits(i)
            print(*(DIGIT_NAMES[lang][d] for d in digits), sep=" ", file=data_file)
    if use_meta:
        with open(dir / f"{lang}{suffix}_meta.tsv", "w") as meta_file:
            for i in range(n_min, n_max):
                print(i, file=meta_file)


def word2digit(word: str) -> int:
    """Convert a word (e.g. "nine") to a digit (e.g. 9)"""
    for digit_words in DIGIT_NAMES.values():
        if word in digit_words:
            return digit_words.index(word)
    return 0


def text2number(text: str) -> int:
    """Convert a word (e.g. "nine one") to a number (e.g. 91)"""
    return int("".join(str(word2digit(w)) for w in text.split()))


def digits2vec(text: str, dim: int = 1024) -> np.ndarray:
    """Convert a text (e.g. "one three") to a number (e.g. 13) and then to the vector of its bits (e.g. [1,1,0,1])."""
    number = text2number(text) + 1  # to make sure the norm of the result is > 0
    b = format(number, "b").zfill(dim)
    return np.asarray([int(i) for i in b], dtype=np.float32)


def test_digits2vec():
    def asarr(arr):
        return np.asarray(arr, dtype=np.float32)

    assert np.array_equal(digits2vec("deux", 4), asarr([0, 0, 1, 1]))
    assert np.array_equal(digits2vec("un cinq", 6), asarr([0, 1, 0, 0, 0, 0]))


class ToyNumbersEncoder(EncodeToNPY):
    """A toy sentence encoder that wraps digits2vec into a stopes Module interface."""

    def __init__(
        self,
        encoder_model: str,
        input_file: str,
        _name: str = "test_numbers_vectorizer",
        output_dir: str = ".",
        input_file_idx: int = 0,
        outfile_prefix: str = "encf",
        outfile_postfix: str = "",
        normalize: bool = True,
        fp16: bool = False,
        # ignored
        spm_vocab: str = "",
        spm_model: str = "",
        **ignored_kwargs,
    ) -> None:
        super().__init__(
            outfile_prefix=outfile_prefix,
            outfile_postfix=outfile_postfix,
            input_file=input_file,
            input_file_idx=input_file_idx,
            output_dir=output_dir,
            normalize=normalize,
            fp16=fp16,
        )

    def name_output_file(self) -> str:
        shard_idx = extract_shard_id(self.input_file, default=self.input_file_idx)
        return str(
            (
                Path(self.output_dir)
                / f"{self.outfile_prefix}.{shard_idx:03d}.{self.outfile_postfix}"
            ).resolve()
        )

    def encode_to_np(
        self, lines_with_number: tp.Iterator[tp.Tuple[int, str]]
    ) -> np.ndarray:
        return np.stack([digits2vec(s) for (_, s) in lines_with_number])


@pytest.mark.parametrize("split_langs", [True, False])
@pytest.mark.parametrize(
    "use_meta", [True, False]
)  # when we split langs, different code branches merge them with and without meta files
@pytest.mark.parametrize("fp16", [True, False])
def test_global_mining_pipeline(
    tmp_path: Path, split_langs: bool, use_meta: bool, fp16: bool
):
    n = 1000

    # prepare the test directories
    demo_dir = tmp_path / "source_data"
    model_dir = tmp_path / "empty_model_dir"
    output_dir = tmp_path / "result"
    tmp_dir = tmp_path / "tmp"
    demo_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    tmp_dir.mkdir(exist_ok=True)

    # prepare the input data: 1 shard for en and 3 shards for fr
    create_digits(dir=demo_dir, lang="en", n_min=0, n_max=1000, use_meta=use_meta)

    thresholds = [0, 100, 300, 1000]
    for i in range(3):
        create_digits(
            dir=demo_dir,
            lang="fr",
            n_min=thresholds[i],
            n_max=thresholds[i + 1],
            suffix=f".00{i}.",
            use_meta=use_meta,
        )

    overrides = [
        # preset
        "launcher.cluster=local",
        "launcher.partition=null",
        "launcher.cache=null",
        "+data.data_version=V32m",
        "+data.iteration=1",
        f"+data.data_shard_dir={demo_dir}",
        "+data.bname=demo_wmt22",
        f"+data.shard_glob={demo_dir}/\\{{lang\\}}*_text.tsv",
        f"+data.meta_glob={demo_dir}/\\{{lang\\}}*_meta.tsv",
        # typical CLI arguments
        "src_lang=fr",
        "tgt_lang=en",
        f"+demo_dir={demo_dir}",
        f"+model_dir={model_dir}",
        f"output_dir={output_dir}",
        # special settings
        "hydra.searchpath=[pkg://stopes.pipelines.tests/conf]",  # path to test_numbers_encoder cfg
        "embed_text=test_numbers_encoder",  # do not test moses and laser; they are external
        "calculate_distances.config.gpu_type=",  # no gpu on test servers
        "train_index.config.use_gpu=False",
        "+populate_index.config.use_gpu=False",
        "mine_sentences.config.score_max=100000",  # use an infinite upper bound
        "+lang_configs.fr.index_type='IVF8,Flat'",  # for flat indexes, merging is not supported
        "embedding_sample.sample_shards=False",  # TODO: understand why test was flaky with sampling
        # cleanup
        f"+launcher.config_dump_dir={tmp_path}/conf",
        f"launcher.log_folder={tmp_path}/logs",
        f"local_tmp_dir={tmp_dir}",
        # fp16 flags
        f"embed_text.config.encoder.fp16={fp16}",
        f"train_index.config.fp16={fp16}",
        f"+populate_index.config.fp16={fp16}",
        f"+calculate_distances.config.fp16={fp16}",
    ]
    if split_langs:
        overrides.append("max_shard_size=500")  # this splits the langs in two

    with hydra.initialize(version_base=None, config_path="../bitext/conf"):
        cfg = hydra.compose(
            config_name="global_mining",
            overrides=overrides,
        )
    print(cfg)
    print("embed config is")
    print(OmegaConf.to_yaml(cfg.embed_text))
    print("demo dir is", str(demo_dir))
    print("result is", str(output_dir))
    out_texts_path, out_meta_path = mining_main(cfg)
    print("out texts are", out_texts_path, out_meta_path)

    with utils.open(out_texts_path, "r") as f:
        matched = [line.strip().split("\t") for line in f.readlines()]
    true_positives = sum(text2number(m[1]) == text2number(m[2]) for m in matched)

    if use_meta:
        with utils.open(out_meta_path, "r") as f:
            matched_meta = [line.strip().split("\t") for line in f.readlines()]
        assert len(matched) == len(matched_meta)
        # check that meta and bitexts are properly aligned
        assert all(
            text2number(line[1]) == int(m[1]) and text2number(line[2]) == int(m[2])
            for line, m in zip(matched, matched_meta)
        )

    # because of the exact embedder, matching rate and precision both should be 100%
    print(len(matched), "matched", true_positives, "correctly")
    precision = true_positives / len(matched)
    assert (
        true_positives == n
    ), f"Not all true pairs have matched: {true_positives} correct pairs out of {n} total."
    if split_langs:
        # when we split languages and then merge, spurious pairs may be matched
        assert (
            precision >= 0.25
        ), f"Matching precision is too low: {true_positives} correct pairs out of {len(matched)}"
        assert (
            len(matched) >= n
        ), f"Matching rate is too low: {len(matched)} pairs out of {n}"
        # verify that the number of finally matched shards is correct: two en shards times two fr shards
        assert len(list(out_texts_path.parent.glob("fr_0*en_0*.bitext*")))
    else:
        assert (
            precision == 1
        ), f"Matching precision is too low: {true_positives} correct pairs out of {len(matched)}"
        assert (
            len(matched) == n
        ), f"Matching rate is too low: {len(matched)} pairs out of {n}"
