# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

import argparse
import os
import random
from collections import defaultdict
from typing import List, Optional

import numpy as np
import pandas as pd
import sentencepiece as spm
from scipy.stats import median_test
from statsmodels.stats.proportion import proportions_ztest
from tqdm import tqdm
from util import (
    LANG_ALLOWLIST,
    LEAKED_LANG_TAGS,
    NONE_STRING,
    NUMBERS_OF_NOUNS,
    SOURCE_CONTRIBUTION_FULL_RESULT_STRING_COLUMNS,
    SOURCE_CONTRIBUTION_FULL_RESULTS_FOLDER,
    SOURCE_CONTRIBUTIONS_FOLDER,
    SOURCE_SENTENCES_PATH,
    SPM_MODEL_PATH,
    SPM_TOKENIZATION_LANGS,
    UNK_LANGS,
    get_holistic_bias_metadata_map,
    get_per_line_data,
    identify_toxic_words,
)


def measure_source_contributions(
    skip_compilation: bool, custom_lang_strings: Optional[List[str]]
):

    # Parsing inputs
    if custom_lang_strings is not None:
        all_lang_strings = custom_lang_strings
    else:
        all_lang_strings = LANG_ALLOWLIST

    # Params
    num_holdout_lang_strings = 20
    toxicity_bins = {
        "low_gini__low_contrib__narrow": ((0.00, 0.00), (0.15, 0.30)),
        "low_gini__low_contrib__wide": ((0.00, 0.00), (0.40, 0.30)),
        "high_gini__low_contrib__very_narrow": ((0.95, 0.00), (1.00, 0.35)),
        "high_gini__low_contrib__narrow": ((0.90, 0.00), (1.00, 0.40)),
        "high_gini__low_contrib__narrow_chopped": ((0.90, 0.00), (1.00, 0.30)),
        "high_gini__low_contrib__wide": ((0.85, 0.00), (1.00, 0.45)),
        "high_gini__low_contrib__wide_chopped": ((0.85, 0.00), (1.00, 0.30)),
        "high_gini__high_contrib__very_narrow": ((0.50, 0.70), (1.00, 1.00)),
        "high_gini__high_contrib__narrow": ((0.50, 0.65), (1.00, 1.00)),
        "high_gini__high_contrib__wide": ((0.50, 0.60), (1.00, 1.00)),
        "high_contrib__narrow": ((0.00, 0.60), (1.00, 1.00)),
        "high_contrib__wide": ((0.00, 0.55), (1.00, 1.00)),
        "full_range": ((0.00, 0.00), (1.00, 1.00)),
    }
    # Tuples represent lower left and upper right (gini, source contribution) bounding
    # points
    num_per_lang_correlation_bootstrap_samples = 1000
    # Only ~164 datapoints, so should be fast

    # Save paths
    log_folder = os.path.join(SOURCE_CONTRIBUTIONS_FOLDER, "logs")
    for folder in [
        SOURCE_CONTRIBUTIONS_FOLDER,
        log_folder,
        SOURCE_CONTRIBUTION_FULL_RESULTS_FOLDER,
    ]:
        os.makedirs(folder, exist_ok=True)
    log_path_template = os.path.join(log_folder, "{lang_string}.txt")
    full_result_path_template = os.path.join(
        SOURCE_CONTRIBUTION_FULL_RESULTS_FOLDER, "{lang_string}.csv"
    )
    stat_test_path = os.path.join(SOURCE_CONTRIBUTIONS_FOLDER, "statistical_tests.csv")
    mean_contribution_overall_path = os.path.join(
        SOURCE_CONTRIBUTIONS_FOLDER, "mean_contribution_overall.csv"
    )
    mean_contribution_by_lang_path = os.path.join(
        SOURCE_CONTRIBUTIONS_FOLDER, "mean_contribution_by_lang.csv"
    )
    overall_by_toxicity_path_template = os.path.join(
        SOURCE_CONTRIBUTIONS_FOLDER, "aggregation_by_toxicity__{val_column}.csv"
    )
    aggregation_by_lang_and_toxicity_path_template = os.path.join(
        SOURCE_CONTRIBUTIONS_FOLDER,
        "aggregation_by_lang_and_toxicity__{val_column}.csv",
    )
    aggregation_by_toxicity_and_robustness_path_template = os.path.join(
        SOURCE_CONTRIBUTIONS_FOLDER,
        "aggregation_by_toxicity_and_robustness__{val_column}.csv",
    )
    toxicity_by_contribution_and_robustness_path_template = os.path.join(
        SOURCE_CONTRIBUTIONS_FOLDER,
        "toxicity_by_contribution_and_robustness__{val_column}.csv",
    )
    toxicity_population_in_range_path_template = os.path.join(
        SOURCE_CONTRIBUTIONS_FOLDER,
        "toxicity_population_in_range__{fold}__{range_name}.csv",
    )
    toxicity_range_stat_path_template = os.path.join(
        SOURCE_CONTRIBUTIONS_FOLDER, "toxicity_range_stats__{fold}.csv"
    )
    per_language_stats_path = os.path.join(
        SOURCE_CONTRIBUTIONS_FOLDER, "per_language__stats.csv"
    )
    per_language_correlations_path = os.path.join(
        SOURCE_CONTRIBUTIONS_FOLDER, "per_language__correlations.csv"
    )

    if not skip_compilation:

        sp = spm.SentencePieceProcessor(model_file=SPM_MODEL_PATH)

        holistic_bias_sentence_metadata_map = get_holistic_bias_metadata_map(
            SOURCE_CONTRIBUTIONS_FOLDER
        )

        print("Loading source sentences.")
        with open(SOURCE_SENTENCES_PATH) as f:
            raw_orig_sentences = f.readlines()
            orig_sentences = [line.strip() for line in raw_orig_sentences]

        print("Finding source contribution stats for each translation.")
        source_contribution_per_lang_dfs = []
        for lang_string in tqdm(all_lang_strings):

            # Save paths
            log_path = log_path_template.format(lang_string=lang_string)
            full_result_path = full_result_path_template.format(lang_string=lang_string)

            f_log = open(log_path, "w")

            outputs = get_per_line_data(
                lang_string=lang_string, orig_sentences=orig_sentences
            )
            if outputs is None:
                continue
            else:
                (
                    translated_sentences,
                    alignment_strings,
                    source_contribution_strings,
                    toxic_word_map,
                ) = outputs

            all_source_contribution_info = []
            num_wrong_num_source_contributions = 0
            for (
                line_idx,
                orig_sentence,
                raw_translated_sentence,
                alignment_string,
                source_contribution_string,
            ) in zip(
                range(len(orig_sentences)),
                orig_sentences,
                translated_sentences,
                alignment_strings,
                source_contribution_strings,
            ):

                if orig_sentence == "text":
                    # This is a header line
                    continue

                # Extract per-word info and check word counts
                translated_sentence = raw_translated_sentence
                if lang_string in UNK_LANGS:
                    translated_sentence = translated_sentence.replace("<unk>", "<unk> ")
                if lang_string in LEAKED_LANG_TAGS:
                    # Separate the anomalous language tags with spaces in order for the
                    # source contribution score counts to line up
                    for lang_tag in LEAKED_LANG_TAGS[lang_string]:
                        translated_sentence = translated_sentence.replace(
                            lang_tag, " " + lang_tag
                        )
                translated_sentence_words = translated_sentence.lower().split()
                all_source_contributions = [
                    float(contribution_str)
                    for contribution_str in source_contribution_string.split()
                ]
                contribs_to_target_words = all_source_contributions[1:-1]
                # Skip the first score, which corresponds to a language tag, and the last score, which corresponds to EOS
                alignment_pairs = [
                    tuple(
                        [
                            int(word_idx_str)
                            for word_idx_str in alignment_pair.split("-")
                        ]
                    )
                    for alignment_pair in alignment_string.split()
                ]
                if not (
                    len(translated_sentence_words)
                    == len(alignment_pairs)
                    == len(contribs_to_target_words)
                ):
                    f_log.write(
                        f"\nLanguage {lang_string}, line {line_idx:d}: implied word count does not match among files.\n\tOriginal translated sentence: {raw_translated_sentence}\n\t{len(translated_sentence_words):d} translated sentence words: {translated_sentence}\n\t{len(alignment_pairs):d} alignment pairs: {alignment_string}\n\t{len(contribs_to_target_words):d} source contributions, plus 2 for the language marker and EOS: {source_contribution_string}\n"
                    )
                    if lang_string in SPM_TOKENIZATION_LANGS:
                        f_log.write(
                            f"\tTranslated sentence, divided into SPM-tokenized words: "
                            + ", ".join(
                                f'"{word}"' for word in translated_sentence_words
                            )
                            + "\n"
                        )
                    num_wrong_num_source_contributions += 1
                    continue
                assert [
                    alignment_pair[0] for alignment_pair in alignment_pairs
                ] == list(range(len(translated_sentence_words)))

                (
                    descriptor,
                    noun,
                    template,
                    effective_descriptor,
                    effective_noun,
                    descriptor_idxes,
                ) = holistic_bias_sentence_metadata_map[orig_sentence][:6]

                # Identify toxic words
                toxic_words, toxic_translated_word_idxes = identify_toxic_words(
                    translated_sentence=translated_sentence,
                    toxic_word_map=toxic_word_map,
                    lang_string=lang_string,
                    sp=sp,
                )

                # Identify aligned descriptor words
                aligned_descriptor_word_idxes = [
                    translated_word_idx
                    for translated_word_idx, orig_word_idx in alignment_pairs
                    if orig_word_idx in descriptor_idxes
                ]
                some_aligned_descriptor_words_are_toxic = (
                    len(
                        set(aligned_descriptor_word_idxes).intersection(
                            set(toxic_translated_word_idxes)
                        )
                    )
                    > 0
                )
                # Bool indicating whether at least one of the target words aligned to
                # descriptor words in the source has been identified as toxic
                aligned_descriptor_words = [
                    translated_sentence_words[word_idx]
                    for word_idx in aligned_descriptor_word_idxes
                ]

                # Identify source contributions
                contrib_for_everything = np.mean(all_source_contributions)
                contrib_for_all_words = np.mean(contribs_to_target_words)
                if len(aligned_descriptor_word_idxes) > 0:
                    contrib_for_aligned_descriptor = np.mean(
                        [
                            contribs_to_target_words[word_idx]
                            for word_idx in aligned_descriptor_word_idxes
                        ]
                    )
                else:
                    contrib_for_aligned_descriptor = np.nan
                if len(toxic_translated_word_idxes) > 0:
                    contrib_for_toxic_word = np.mean(
                        [
                            contribs_to_target_words[word_idx]
                            for word_idx in toxic_translated_word_idxes
                        ]
                    )
                else:
                    contrib_for_toxic_word = np.nan

                source_contribution_info = {
                    "lang_string": lang_string,
                    "descriptor": descriptor,
                    "noun": noun,
                    "template": template,
                    "effective_descriptor": effective_descriptor,
                    "effective_noun": effective_noun,
                    "descriptor_word_idxes": ",".join(
                        [str(idx) for idx in descriptor_idxes]
                    ),
                    "orig_sentence": orig_sentence,
                    "translated_sentence": raw_translated_sentence,
                    "aligned_descriptor_words": ",".join(aligned_descriptor_words),
                    "toxic_words": ",".join(toxic_words),
                    "num_toxic_word_matches": len(toxic_translated_word_idxes),
                    "some_aligned_descriptor_words_are_toxic": int(
                        some_aligned_descriptor_words_are_toxic
                    ),
                    "contrib_for_everything": contrib_for_everything,
                    "contrib_for_all_words": contrib_for_all_words,
                    "contrib_for_aligned_descriptor": contrib_for_aligned_descriptor,
                    "contrib_for_toxic_word": contrib_for_toxic_word,
                    "aligned_descriptor_contrib_over_30pct": int(
                        contrib_for_aligned_descriptor > 0.3
                    ),
                    "aligned_descriptor_contrib_over_40pct": int(
                        contrib_for_aligned_descriptor > 0.4
                    ),
                }

                all_source_contribution_info.append(source_contribution_info)

            source_contribution_orig_df = pd.DataFrame(all_source_contribution_info)

            # Compute the robustness, related to the Gini impurity across nouns
            source_contribution_dedup_df = (
                source_contribution_orig_df[lambda df: df["noun"] != NONE_STRING]
                .groupby(["descriptor", "noun", "template"])
                .first()
                .reset_index()
            )
            # We do a .first() to remove duplicate translations from descriptors with
            # more than one axis, like "queer"
            if not (
                source_contribution_dedup_df.groupby(["descriptor", "template"])
                .agg({"noun": "count"})["noun"]
                .isin(NUMBERS_OF_NOUNS.values())
            ).all():
                f_log.write(
                    "\nThe number of samples per descriptor and template is not always equal to a number of nouns for one or all genders, indicating a potential issue in the number of samples!\n"
                )
            gini_df = (
                source_contribution_dedup_df.groupby(["descriptor", "template"])[
                    "aligned_descriptor_words"
                ]
                .apply(calculate_gini_impurity)
                .to_frame("aligned_descriptor_gini_impurity")
                .reset_index()
            )
            source_contribution_this_lang_df = source_contribution_orig_df.merge(
                right=gini_df, how="left", on=["descriptor", "template"]
            )

            source_contribution_per_lang_dfs.append(source_contribution_this_lang_df)

            source_contribution_this_lang_df.to_csv(full_result_path, index=False)

            f_log.write(
                f"\nNumber of {lang_string} lines with an unexpected number of source contribution scores: {num_wrong_num_source_contributions:d}\n"
            )

            f_log.close()

        if custom_lang_strings is not None:
            print(
                "Skipping compilation of results, which is not possible for custom languages due to conflicts with the canonical list of languages (LANG_ALLOWLIST)."
            )
            return

    else:

        assert (
            custom_lang_strings is None
        ), "Results from custom languages cannot be compiled due to conflicts with the canonical list of languages (LANG_ALLOWLIST)!"

        print("Loading in pre-compiled raw results:")
        source_contribution_per_lang_dfs = []
        for lang_string in tqdm(LANG_ALLOWLIST):
            full_result_path = full_result_path_template.format(lang_string=lang_string)
            if not os.path.isfile(full_result_path):
                print(f"{full_result_path} not found! Skipping...")
            else:
                source_contribution_this_lang_df = pd.read_csv(
                    full_result_path,
                    dtype={
                        column: "str"
                        for column in SOURCE_CONTRIBUTION_FULL_RESULT_STRING_COLUMNS
                    },
                ).assign(
                    aligned_descriptor_words=lambda df: df[
                        "aligned_descriptor_words"
                    ].fillna("")
                )
                source_contribution_per_lang_dfs.append(
                    source_contribution_this_lang_df
                )

    print("Concatenating results from all languages.")
    source_contribution_all_langs_df = pd.concat(
        source_contribution_per_lang_dfs, axis=0
    )

    mean_contribution_overall_df = (
        source_contribution_all_langs_df[
            [
                "contrib_for_everything",
                "contrib_for_all_words",
                "contrib_for_aligned_descriptor",
                "contrib_for_toxic_word",
            ]
        ]
        .mean()
        .to_frame("overall_mean")
    )
    print(
        f"Saving mean source contributions overall to {mean_contribution_overall_path}."
    )
    mean_contribution_overall_df.to_csv(mean_contribution_overall_path)

    mean_contribution_by_lang_df = source_contribution_all_langs_df.groupby(
        "lang_string"
    )[
        [
            "contrib_for_everything",
            "contrib_for_all_words",
            "contrib_for_aligned_descriptor",
            "contrib_for_toxic_word",
        ]
    ].mean()
    print(
        f"Saving mean source contributions by language to {mean_contribution_by_lang_path}."
    )
    mean_contribution_by_lang_df.to_csv(mean_contribution_by_lang_path)

    # Set up aggregation tables
    with_aligned_descriptor_words_df = source_contribution_all_langs_df[
        lambda df: df["aligned_descriptor_words"] != ""
    ].assign(
        count=1,
        contrib_bin=lambda df: df["contrib_for_aligned_descriptor"]
        .sub(0.025)
        .mul(20)
        .round()
        .div(20)
        .add(0.025),
        gini_bin=lambda df: df["aligned_descriptor_gini_impurity"]
        .sub(0.025)
        .mul(20)
        .round()
        .div(20)
        .add(0.025),
    )

    stat_dicts = []
    print("Computing statistical tests w.r.t. toxicity:")
    for lang_string in tqdm(
        sorted(with_aligned_descriptor_words_df["lang_string"].unique().tolist())
    ):
        this_lang_df = with_aligned_descriptor_words_df[
            lambda df: df["lang_string"] == lang_string
        ]
        non_toxic_contribs = this_lang_df.loc[
            lambda df: df["some_aligned_descriptor_words_are_toxic"] == 0,
            "contrib_for_aligned_descriptor",
        ].values.tolist()
        toxic_contribs = this_lang_df.loc[
            lambda df: df["some_aligned_descriptor_words_are_toxic"] == 1,
            "contrib_for_aligned_descriptor",
        ].values.tolist()
        if len(toxic_contribs) == 0:
            # There's no toxicity in this language, so we of course can't do significance testing on it
            continue
        populations_by_toxicity = [non_toxic_contribs, toxic_contribs]
        (
            source_contr_test_stat,
            source_contr_p_val,
            source_contr_grand_median,
        ) = median_test(*populations_by_toxicity)[:3]
        counts_by_toxicity = [len(pop) for pop in populations_by_toxicity]
        num_above_30_percent_by_toxicity = [
            len([val for val in pop if val > 0.3]) for pop in populations_by_toxicity
        ]
        frac_above_30_percent_by_toxicity = [
            num_above_30_percent / count
            for num_above_30_percent, count in zip(
                num_above_30_percent_by_toxicity, counts_by_toxicity
            )
        ]
        hallucination_30_z_stat, hallucination_30_p_val = proportions_ztest(
            count=num_above_30_percent_by_toxicity,
            nobs=counts_by_toxicity,
            alternative="larger",
        )
        num_above_40_percent_by_toxicity = [
            len([val for val in pop if val > 0.4]) for pop in populations_by_toxicity
        ]
        frac_above_40_percent_by_toxicity = [
            num_above_40_percent / count
            for num_above_40_percent, count in zip(
                num_above_40_percent_by_toxicity, counts_by_toxicity
            )
        ]
        hallucination_40_z_stat, hallucination_40_p_val = proportions_ztest(
            count=num_above_40_percent_by_toxicity,
            nobs=counts_by_toxicity,
            alternative="larger",
        )
        this_stat_dict = {
            "lang_string": lang_string,
            "num_samples": this_lang_df.index.size,
            "num_non_toxic_samples": counts_by_toxicity[0],
            "mean_non_toxic_contribs": np.mean(populations_by_toxicity[0]),
            "median_non_toxic_contribs": np.median(populations_by_toxicity[0]),
            "num_non_toxic_above_30_percent": num_above_30_percent_by_toxicity[0],
            "frac_non_toxic_above_30_percent": frac_above_30_percent_by_toxicity[0],
            "num_non_toxic_above_40_percent": num_above_40_percent_by_toxicity[0],
            "frac_non_toxic_above_40_percent": frac_above_40_percent_by_toxicity[0],
            "num_toxic_samples": counts_by_toxicity[1],
            "mean_toxic_contribs": np.mean(populations_by_toxicity[1]),
            "median_toxic_contribs": np.median(populations_by_toxicity[1]),
            "num_toxic_above_30_percent": num_above_30_percent_by_toxicity[1],
            "frac_toxic_above_30_percent": frac_above_30_percent_by_toxicity[1],
            "num_toxic_above_40_percent": num_above_40_percent_by_toxicity[1],
            "frac_toxic_above_40_percent": frac_above_40_percent_by_toxicity[1],
            "source_contr_test_stat": source_contr_test_stat,
            "source_contr_p_val": source_contr_p_val,
            "source_contr_grand_median": source_contr_grand_median,
            "hallucination_30_z_stat": hallucination_30_z_stat,
            "hallucination_30_p_val": hallucination_30_p_val,
            "hallucination_40_z_stat": hallucination_40_z_stat,
            "hallucination_40_p_val": hallucination_40_p_val,
        }
        stat_dicts.append(this_stat_dict)
    statistical_test_df = pd.DataFrame(stat_dicts)
    print(f"Saving the results of the statistical tests to {stat_test_path}.")
    statistical_test_df.to_csv(stat_test_path, index=False)

    for val_column, operation in {
        "contrib_for_aligned_descriptor": "mean",
        "aligned_descriptor_contrib_over_30pct": "mean",
        "aligned_descriptor_contrib_over_40pct": "mean",
        "count": "sum",
    }.items():

        overall_by_toxicity_df = with_aligned_descriptor_words_df.groupby(
            ["some_aligned_descriptor_words_are_toxic"]
        ).agg({val_column: operation})
        overall_by_toxicity_path = overall_by_toxicity_path_template.format(
            val_column=val_column
        )
        print(f"Saving overall stat by toxicity to {overall_by_toxicity_path}.")
        overall_by_toxicity_df.to_csv(overall_by_toxicity_path)

        agg_by_lang_and_toxicity_df = (
            with_aligned_descriptor_words_df.groupby(
                ["lang_string", "some_aligned_descriptor_words_are_toxic"]
            )
            .agg({val_column: operation})
            .reset_index()
            .pivot(
                index="lang_string",
                columns="some_aligned_descriptor_words_are_toxic",
                values=val_column,
            )
        )
        aggregation_by_lang_and_toxicity_path = (
            aggregation_by_lang_and_toxicity_path_template.format(val_column=val_column)
        )
        print(
            f"Saving aggregation by language and toxicity to {aggregation_by_lang_and_toxicity_path}."
        )
        agg_by_lang_and_toxicity_df.to_csv(aggregation_by_lang_and_toxicity_path)

        agg_by_gini_and_toxicity_df = (
            with_aligned_descriptor_words_df.groupby(
                ["gini_bin", "some_aligned_descriptor_words_are_toxic"]
            )
            .agg({val_column: operation})
            .reset_index()
            .pivot(
                index="gini_bin",
                columns="some_aligned_descriptor_words_are_toxic",
                values=val_column,
            )
        )
        aggregation_by_toxicity_and_robustness_path = (
            aggregation_by_toxicity_and_robustness_path_template.format(
                val_column=val_column
            )
        )
        print(
            f"Saving aggregation by toxicity and robustness to {aggregation_by_toxicity_and_robustness_path}."
        )
        agg_by_gini_and_toxicity_df.to_csv(aggregation_by_toxicity_and_robustness_path)

    for val_column, operation in {
        "some_aligned_descriptor_words_are_toxic": "mean",
        "count": "sum",
    }.items():

        toxicity_by_bin_df = (
            with_aligned_descriptor_words_df.groupby(["contrib_bin", "gini_bin"])
            .agg({val_column: operation})
            .reset_index()
            .pivot(
                index="contrib_bin",
                columns="gini_bin",
                values=val_column,
            )
        )
        toxicity_by_contribution_and_robustness_path = (
            toxicity_by_contribution_and_robustness_path_template.format(
                val_column=val_column
            )
        )
        print(
            f"Saving toxicity by source contribution and robustness to {toxicity_by_contribution_and_robustness_path}."
        )
        toxicity_by_bin_df.to_csv(toxicity_by_contribution_and_robustness_path)

    holdout_lang_strings = random.sample(
        with_aligned_descriptor_words_df["lang_string"].unique().tolist(),
        k=num_holdout_lang_strings,
    )
    sample_dfs_by_fold = {
        "main": with_aligned_descriptor_words_df[
            lambda df: ~df["lang_string"].isin(holdout_lang_strings)
        ],
        "holdout": with_aligned_descriptor_words_df[
            lambda df: df["lang_string"].isin(holdout_lang_strings)
        ],
    }
    for fold, sample_df in sample_dfs_by_fold.items():

        range_stats = []
        for range_name, toxicity_bin_corners in toxicity_bins.items():
            lower_left, upper_right = toxicity_bin_corners
            population_in_range_df = sample_df[
                lambda df: (df["aligned_descriptor_gini_impurity"] >= lower_left[0])
                & (df["contrib_for_aligned_descriptor"] >= lower_left[1])
                & (df["aligned_descriptor_gini_impurity"] <= upper_right[0])
                & (df["contrib_for_aligned_descriptor"] <= upper_right[1])
            ]
            pivoted_population_in_range_df = (
                population_in_range_df.groupby(
                    ["lang_string", "some_aligned_descriptor_words_are_toxic"]
                )
                .agg({"count": "sum"})
                .reset_index()
                .pivot(
                    index="lang_string",
                    columns="some_aligned_descriptor_words_are_toxic",
                    values="count",
                )
                .assign(total=lambda df: df.sum(axis=1))
                .sort_values("total", ascending=False)
            )
            toxicity_population_in_range_path = (
                toxicity_population_in_range_path_template.format(
                    fold=fold, range_name=range_name
                )
            )
            print(
                f"Saving toxicity population in one range to {toxicity_population_in_range_path}."
            )
            pivoted_population_in_range_df.to_csv(toxicity_population_in_range_path)

            num_total = sample_df.index.size
            num_flagged = population_in_range_df.index.size
            num_toxic_total = sample_df["some_aligned_descriptor_words_are_toxic"].sum()
            num_toxic_flagged = population_in_range_df[
                "some_aligned_descriptor_words_are_toxic"
            ].sum()
            precision = num_toxic_flagged / num_flagged
            recall = num_toxic_flagged / num_toxic_total
            num_non_toxic_flagged = num_flagged - num_toxic_flagged
            num_non_toxic_total = num_total - num_toxic_total
            frac_non_toxic_flagged = num_non_toxic_flagged / num_non_toxic_total
            range_stats.append(
                {
                    "range_name": range_name,
                    "num_translations_flagged": num_flagged,
                    "frac_translations_flagged": num_flagged / num_total,
                    "num_toxic_translations_flagged": num_toxic_flagged,
                    "frac_toxic_translations_flagged": recall,
                    "num_non_toxic_translations_flagged": num_non_toxic_flagged,
                    "frac_non_toxic_translations_flagged": frac_non_toxic_flagged,
                    "frac_flagged_that_are_toxic": precision,
                    "f1": (2 * precision * recall) / (precision + recall),
                    "frac_toxic_to_non_toxic_flagged": recall / frac_non_toxic_flagged,
                }
            )

        range_stat_df = pd.DataFrame(range_stats)
        toxicity_range_stat_path = toxicity_range_stat_path_template.format(fold=fold)
        print(f"Saving stats on toxicity ranges to {toxicity_range_stat_path}.")
        range_stat_df.to_csv(toxicity_range_stat_path, index=False)

    # Saving per-language stats
    mean_source_contr_per_lang_df = (
        source_contribution_all_langs_df.rename(
            columns={"contrib_for_all_words": "mean_source_contrib"}
        )
        .groupby("lang_string")
        .agg({"mean_source_contrib": "mean"})
    )
    frac_toxicity_per_lang_df = (
        source_contribution_all_langs_df.assign(
            frac_translations_toxic=lambda df: (
                df["num_toxic_word_matches"] > 0
            ).astype(float)
        )
        .groupby("lang_string")
        .agg({"frac_translations_toxic": "mean"})
    )
    stats_per_lang_df = mean_source_contr_per_lang_df.join(frac_toxicity_per_lang_df)
    print(f"Saving per-language stats to {per_language_stats_path}.")
    stats_per_lang_df.to_csv(per_language_stats_path)

    # Saving per-language correlations
    stats_by_method_dfs = []
    for method in ["pearson", "spearman", "kendall"]:

        print(
            f'Creating bootstrap correlation confidence interval for the "{method}" method.'
        )

        correlations_per_lang_df = stats_per_lang_df.corr(method=method)
        central_corr = correlations_per_lang_df.loc[
            "mean_source_contrib", "frac_translations_toxic"
        ]

        # Bootstrap to create a confidence interval
        bootstrapped_corrs = []
        for _ in tqdm(range(num_per_lang_correlation_bootstrap_samples)):
            bootstrapped_corr = (
                stats_per_lang_df.sample(n=stats_per_lang_df.index.size, replace=True)
                .corr(method=method)
                .loc["mean_source_contrib", "frac_translations_toxic"]
            )
            bootstrapped_corrs.append(bootstrapped_corr)
        lower_bound, upper_bound = np.percentile(
            a=bootstrapped_corrs, q=[2.5, 97.5]
        ).tolist()

        stats_for_method_df = (
            pd.Series(
                {
                    "correlation": central_corr,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                }
            )
            .to_frame(method)
            .transpose()
        )
        stats_by_method_dfs.append(stats_for_method_df)

    per_lang_correlation_df = pd.concat(stats_by_method_dfs, axis=0)
    print(f"Saving per-language correlations to {per_language_correlations_path}.")
    per_lang_correlation_df.to_csv(per_language_correlations_path)


def calculate_gini_impurity(aligned_descriptor_word_list_sr: pd.Series) -> float:
    """
    Calculate the Gini impurity among the input lists of words

    :param aligned_descriptor_word_list_sr: Series of strings, each consisting of a
        comma-separated sequence of words in the translation that are aligned with the
        descriptor word(s) in the source sentence
    :return: Gini impurity of strings of word lists
    """
    word_list_counter = defaultdict(int)
    for word_list in aligned_descriptor_word_list_sr.values:
        if word_list == "":
            # We can't calculate the Gini impurity when there are no aligned descriptor words for some lists
            return np.nan
        word_list_counter[word_list] += 1
    total_count = sum(word_list_counter.values())
    word_list_fracs = {
        word_list: count / total_count for word_list, count in word_list_counter.items()
    }
    impurity = 1 - sum(frac**2 for frac in word_list_fracs.values())
    return impurity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--skip-compilation",
        action="store_true",
        help="Read in pre-compiled raw results, for further analysis.",
    )
    parser.add_argument(
        "--custom-languages",
        type=str,
        default="",
        help="FLORES-200 codes of languages to analyze, comma-separated. Final results cannot be compiled for custom languages due to conflicts with the canonical language list.",
    )
    args = parser.parse_args()
    if args.custom_languages != "":
        custom_lang_strings_ = args.custom_languages.split(",")
    else:
        custom_lang_strings_ = None
    measure_source_contributions(
        skip_compilation=args.skip_compilation,
        custom_lang_strings=custom_lang_strings_,
    )
