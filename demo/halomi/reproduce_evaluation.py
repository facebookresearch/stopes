# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os

import numpy as np
import pandas as pd
from evaluation_utils import (
    evaluate_token_level_features,
    get_word2chars_mapping,
    percent_correct_pairs,
)

SENTENCE_METRIC_NAMES = [
    "score_log_loss",
    "score_alti_mean",
    "score_alti_t_mean",
    "score_attn_ot",
    "score_comet_qe",
    "score_labse",
    "score_laser",
    "score_xnli",
    "score_blaser2_qe",
]

TOKEN_METRIC_NAMES = [
    "score_log_loss",
    "score_log_loss_contrastive",
    "score_alti_sum",
    "score_alti_max",
]


def main(data_root):
    dataset_full = pd.read_csv(os.path.join(data_root, "halomi_full.tsv"), sep="\t")
    target_token_df = pd.read_csv(
        os.path.join(data_root, "halomi_full_target_tokens.tsv"),
        sep="\t",
        keep_default_na=False,
    )
    source_token_df = pd.read_csv(
        os.path.join(data_root, "halomi_full_source_tokens.tsv"),
        sep="\t",
        keep_default_na=False,
    )

    print("\n\nReproducing sentence-level scores...\n")
    df2 = dataset_full[dataset_full.perturbation == "natural"].copy()

    hallucinations_cmp = df2.groupby("direction", group_keys=True).apply(
        lambda x: pd.Series(
            {
                m: percent_correct_pairs(x["class_hall"], x[m])
                for m in SENTENCE_METRIC_NAMES
            }
        )
    )
    print("Direction-wise mean score for hallucination detection:")
    print(hallucinations_cmp.mean())
    assert hallucinations_cmp.mean().round(2).to_dict() == {
        "score_log_loss": 0.80,
        "score_alti_mean": 0.75,
        "score_alti_t_mean": 0.58,
        "score_attn_ot": 0.53,
        "score_comet_qe": 0.75,
        "score_labse": 0.78,
        "score_laser": 0.75,
        "score_xnli": 0.67,
        "score_blaser2_qe": 0.83,
    }

    omissions_cmp = (
        df2[df2.class_hall == "1_No_hallucination"]
        .groupby("direction")
        .apply(
            lambda x: pd.Series(
                {
                    m: percent_correct_pairs(x["class_omit"], x[m])
                    for m in SENTENCE_METRIC_NAMES
                }
            )
        )
    )
    print("Direction-wise mean score for omissions detection:")
    print(omissions_cmp.mean())
    assert omissions_cmp.mean().round(2).to_dict() == {
        "score_log_loss": 0.59,
        "score_alti_mean": 0.48,
        "score_alti_t_mean": 0.77,
        "score_attn_ot": 0.7,
        "score_comet_qe": 0.59,
        "score_labse": 0.72,
        "score_laser": 0.64,
        "score_xnli": 0.65,
        "score_blaser2_qe": 0.74,
    }

    print("\n\nReproducing word-level scores...\n")

    feature_sets = {k: [k] for k in TOKEN_METRIC_NAMES}
    feature_sets["joint"] = TOKEN_METRIC_NAMES

    # Target-side evaluation
    sentences = df2.mt_text_normalized
    character_masks = df2.hall_mask_normalized
    tokens_dataset = target_token_df[
        (target_token_df.perturbation != "perturbed")
        & (target_token_df.token_weight > 0)
    ].copy()

    word_to_character_maps = [get_word2chars_mapping(sent) for sent in sentences]
    word_labels = [
        [max(int(mask[i]) for i in w) for w in words]
        for words, mask in zip(word_to_character_maps, character_masks)
    ]
    print("Direction-wise mean score for hallucination detection:")
    results = {}
    for fe_name, fe in feature_sets.items():
        (word_level_preds, direction2auc,) = evaluate_token_level_features(
            tokens_dataset, fe, word_to_character_maps, word_labels
        )
        auc = np.mean(list(direction2auc.values()))
        results[fe_name] = auc.round(2)
        print(fe_name, auc)
    assert results == {
        "score_log_loss": 0.7,
        "score_log_loss_contrastive": 0.7,
        "score_alti_sum": 0.78,
        "score_alti_max": 0.68,
        "joint": 0.84,
    }

    # Source-side evaluation
    sentences = df2.src_text_normalized
    character_masks = df2.omit_mask_normalized
    tokens_dataset = source_token_df[
        (source_token_df.perturbation != "perturbed")
        & (source_token_df.token_weight > 0)
    ].copy()

    # extracting word boundaries and ground truth word labels
    word_to_character_maps = [get_word2chars_mapping(sent) for sent in sentences]
    word_labels = [
        [max(int(mask[i]) for i in w) for w in words]
        for words, mask in zip(word_to_character_maps, character_masks)
    ]
    results = {}
    print("Direction-wise mean score for omission detection:")
    for fe_name, fe in feature_sets.items():
        (word_level_preds, direction2auc,) = evaluate_token_level_features(
            tokens_dataset, fe, word_to_character_maps, word_labels
        )
        auc = np.mean(list(direction2auc.values()))
        results[fe_name] = auc.round(2)
        print(fe_name, auc)
    assert results == {
        "score_log_loss": 0.78,
        "score_log_loss_contrastive": 0.73,
        "score_alti_sum": 0.78,
        "score_alti_max": 0.76,
        "joint": 0.84,
    }

    print("\n\nEverything reproduced as expected!")


if __name__ == "__main__":
    main(data_root="data")
