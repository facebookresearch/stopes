# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import sentencepiece as spm

import stopes.pipelines.prepare_data.data_types as data_types
from stopes.pipelines.prepare_data.cache import cache_step

logger = logging.getLogger(__name__)


def _sample_corpus_for_vocab(
    lang_files: Dict[str, List[str]],
    lang_counts_map: Dict[str, int],
    vocab_build_params: data_types.VocabBuildParams,
    output_prefix: str,
) -> str:
    """
    Get a random sample of sentences from each language with sample proportion
    calculated by temperature-based sampling.
    Returns path of sampled data
    """
    logger.info("sample_corpus_for_vocab")
    logger.info(f"lang_counts_map: {lang_counts_map}")

    sum_count = sum(lang_counts_map.values())
    if not sum_count:
        raise ValueError("empty training data")

    lang_probs_map_smoothed = {
        lang: (count / sum_count) ** (1 / vocab_build_params.sampling_temperature)
        for lang, count in lang_counts_map.items()
    }
    sum_prob_smoothed = sum(lang_probs_map_smoothed.values())

    total_size = min(vocab_build_params.sampled_data_size, sum_count)
    lang_counts_sampled_map = {
        lang: int(total_size * (prob / sum_prob_smoothed))
        for lang, prob in lang_probs_map_smoothed.items()
    }
    logger.info(f"samples count per lang: {lang_counts_sampled_map}")

    output_path = f"{output_prefix}.sampled_corpus_vocab"
    with open(output_path, "wb") as f_out:
        np.random.seed(vocab_build_params.random_seed)
        for lang, all_corpora in lang_files.items():
            total_count = lang_counts_map[lang]
            sample_count = lang_counts_sampled_map[lang]
            idx = set(np.random.choice(total_count, sample_count, replace=True))
            i = 0
            for corpus in all_corpora:
                with open(corpus, mode="rb") as in_f:
                    for line in in_f:
                        if i in idx:
                            f_out.write(line)
                        i += 1
            assert (
                total_count == i
            ), f"Number of lines mismatch expected {total_count} vs {i}"
    return output_path


def _vocab_to_dict(vocab_path: str, dict_path: str) -> None:
    """
    Generate dictionary for vocab file
    """
    logger.info("vocab_to_dict")
    logger.info(f"vocab dict path: {dict_path}\n")

    with open(vocab_path) as vocab_f, open(dict_path, "w") as dict_f:
        for line in vocab_f.readlines()[3:]:
            vocab_piece = line.split("\t")[0]
            dict_f.write(f"{vocab_piece} 1\n")
    return dict_path


@cache_step("train_new_vocab")
async def train_new_vocab(
    lang_files: Dict[str, List[str]],
    lang_counts_map: Dict[str, int],
    vocab_build_params: data_types.VocabBuildParams,
    sample_pref: str,
    output_pref: str,
    output_dir: str,
    custom_step_name: str,
) -> data_types.BuiltVocab:
    """
    Get corpus sample and train new vocab
    Returns trained vocab
    """
    logger.info("train_new_vocab")
    logger.info(f"output_pref: {output_pref}")

    source_vocab_corpus = _sample_corpus_for_vocab(
        lang_files=lang_files,
        lang_counts_map=lang_counts_map,
        vocab_build_params=vocab_build_params,
        output_prefix=sample_pref,
    )

    vocab_size = vocab_build_params.vocab_size
    model_prefix = f"{output_pref}.{vocab_size}"
    logger.info(f"train vocab: {vocab_build_params.model_type} type\n")
    spm.SentencePieceTrainer.train(
        input=source_vocab_corpus,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=vocab_build_params.character_coverage,
        input_sentence_size=vocab_build_params.sampled_data_size,
        shuffle_input_sentence=vocab_build_params.shuffle_input_sentence,
        model_type=vocab_build_params.model_type,
        minloglevel=1,
    )
    return data_types.BuiltVocab(
        model_file=f"{model_prefix}.model", vocab_file=f"{model_prefix}.vocab"
    )


async def get_vocab(
    data_config: data_types.DataConfig,
    train_corpora_dict: Dict[str, Dict[str, data_types.ParallelDataset]],
    src_counts_map: Dict[str, int],
    tgt_counts_map: Dict[str, int],
    output_dir: str,
) -> Tuple[data_types.BuiltVocab, data_types.BuiltVocab]:
    logger.info("Building vocabulary")

    vocab_dir = os.path.join(output_dir, "vocab_bin")
    if not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir, exist_ok=True)

    pretrained_source = data_config.source_vocab_config.pretrained
    pretrained_target = data_config.target_vocab_config.pretrained

    if pretrained_source and pretrained_target:
        logger.info("Using pretrained vocab, skipping\n")

        src_vocab, tgt_vocab = pretrained_source, pretrained_target
    else:
        # get lang files
        src_lang_files = defaultdict(list)
        tgt_lang_files = defaultdict(list)
        for _, train_corpora in train_corpora_dict.items():
            for direction, parallel_dataset in train_corpora.items():
                src_lang_files[direction.split("-")[0]].append(parallel_dataset.source)
                tgt_lang_files[direction.split("-")[1]].append(parallel_dataset.target)

        joint_source = (
            data_config.source_vocab_config.vocab_build_params.use_joined_data
        )
        joint_target = (
            data_config.target_vocab_config.vocab_build_params.use_joined_data
        )
        if ((not pretrained_source) and joint_source) or (
            (not pretrained_target) and joint_target
        ):
            joint_lang_files = src_lang_files.copy()
            for direction, paths in tgt_lang_files.items():
                joint_lang_files[direction].extend(paths)
            joint_counts_map = src_counts_map.copy()
            for lang, count in tgt_counts_map.items():
                joint_counts_map[lang] += count

        # get models
        if pretrained_source:
            src_vocab = pretrained_source
        else:
            src_vocab = await train_new_vocab(
                output_dir=output_dir,
                custom_step_name="train_new_vocab.src",
                lang_files=joint_lang_files if joint_source else src_lang_files,
                lang_counts_map=joint_counts_map if joint_source else src_counts_map,
                vocab_build_params=data_config.source_vocab_config.vocab_build_params,
                sample_pref=f"{vocab_dir}/source",
                output_pref=f"{vocab_dir}/sentencepiece.source",
            )

        if pretrained_target:
            tgt_vocab = pretrained_target
        elif data_config.target_vocab_config == data_config.source_vocab_config:
            tgt_vocab = src_vocab
        else:
            tgt_vocab = await train_new_vocab(
                output_dir=output_dir,
                custom_step_name="train_new_vocab.tgt",
                lang_files=joint_lang_files if joint_target else tgt_lang_files,
                lang_counts_map=joint_counts_map if joint_target else tgt_counts_map,
                vocab_build_params=data_config.target_vocab_config.vocab_build_params,
                sample_pref=f"{vocab_dir}/target",
                output_pref=f"{vocab_dir}/sentencepiece.target",
            )

    # get vocab dictionary
    src_dict_file = _vocab_to_dict(
        src_vocab.vocab_file,
        f"{output_dir}/{os.path.basename(src_vocab.vocab_file).rsplit('.',1)[0]}.source.dict.txt",
    )

    tgt_dict_file = _vocab_to_dict(
        tgt_vocab.vocab_file,
        f"{output_dir}/{os.path.basename(tgt_vocab.vocab_file).rsplit('.', 1)[0]}.target.dict.txt",
    )
    src_vocab.dict_file = src_dict_file
    tgt_vocab.dict_file = tgt_dict_file
    return (src_vocab, tgt_vocab)
