# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import asyncio
import typing as tp
from collections import defaultdict
from pathlib import Path

from stopes.core import utils
from stopes.core.launcher import Launcher
from stopes.modules.preprocess.train_spm import TrainSpmModule, Vocab
from stopes.pipelines.filtering.dataset import Dataset
from stopes.pipelines.prepare_data.configs import VocabConfig
from stopes.pipelines.prepare_data.sample_corpus import sample_corpus


async def train_spm(
    corpus_file, vocab_config, output_dir, launcher, direction: str
) -> Vocab:
    with utils.clone_config(vocab_config.spm_config) as spm_config:
        spm_config.output_dir = str(output_dir)
        spm_config.train_data_file = str(corpus_file)
        spm_config.training_lines = vocab_config.sampling_config.sampled_data_size
        spm_config.model_prefix_spm = str(
            output_dir
            / f"sentencepiece.{direction}.{vocab_config.spm_config.vocab_size}"
        )
    vocab = await launcher.schedule(TrainSpmModule(spm_config))
    return vocab


async def sample_train_spm(
    pretrained,
    sampling_config,
    lang_files,
    train_counts_map,
    vocab_config,
    launcher,
    output_dir,
    direction: str,
) -> Vocab:
    if not pretrained:
        train_sampled_corpus = await sample_corpus(
            sampling_config,
            lang_files,
            train_counts_map,
            launcher,
            output_dir / f"{direction}.sampled_corpus_vocab",
        )
        # Train src SPM model
        vocab = await train_spm(
            train_sampled_corpus,
            vocab_config,
            output_dir,
            launcher,
            direction,
        )
    else:
        vocab = Vocab(
            model_file=Path(pretrained.model_file),
            vocab_file=Path(pretrained.vocab_file),
        )
    return vocab


async def build_vocab(
    retrieved_datasets: tp.List[Dataset],
    vocab_config: VocabConfig,
    train_src_counts_map: tp.Dict[str, int],
    train_tgt_counts_map: tp.Dict[str, int],
    launcher: Launcher,
    output_dir: Path,
) -> tp.Tuple[Vocab, Vocab]:

    pretrained_src = vocab_config.src_vocab.pretrained
    pretrained_tgt = vocab_config.tgt_vocab.pretrained

    src_lang_files = defaultdict(list)
    tgt_lang_files = defaultdict(list)
    for dataset in retrieved_datasets:
        if dataset.fold.startswith("train"):
            src, tgt = dataset.lang_dir.split("-")
            src_lang_files[src].append(dataset.src)
            tgt_lang_files[tgt].append(dataset.tgt)

    if not pretrained_src and not pretrained_tgt and vocab_config.use_joined_data:
        lang_files = src_lang_files.copy()
        for lang, paths in tgt_lang_files.items():
            lang_files[lang].extend(paths)

        train_counts_map = train_src_counts_map.copy()
        for lang, count in train_tgt_counts_map.items():
            train_counts_map[lang] += count

        train_sampled_corpus = await sample_corpus(
            # Use src sampling_config by default
            vocab_config.src_vocab.sampling_config,
            lang_files,
            train_counts_map,
            launcher,
            output_dir / "source.sampled_corpus_vocab",
        )
        # Train joint SPM model
        joint_vocab = await train_spm(
            train_sampled_corpus, vocab_config.src_vocab, output_dir, launcher, "source"
        )
        src_vocab, tgt_vocab = joint_vocab, joint_vocab
    else:
        src_vocab, tgt_vocab = await asyncio.gather(
            sample_train_spm(
                pretrained_src,
                vocab_config.src_vocab.sampling_config,
                src_lang_files,
                train_src_counts_map,
                vocab_config.src_vocab,
                launcher,
                output_dir,
                "source",
            ),
            sample_train_spm(
                pretrained_tgt,
                vocab_config.tgt_vocab.sampling_config,
                tgt_lang_files,
                train_tgt_counts_map,
                vocab_config.tgt_vocab,
                launcher,
                output_dir,
                "target",
            ),
        )
    return src_vocab, tgt_vocab
