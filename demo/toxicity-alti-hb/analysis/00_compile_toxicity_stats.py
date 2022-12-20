# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import argparse
import os

import pandas as pd
from tqdm import tqdm
from util import (
    BASE_TOXICITY_SAVE_FOLDER,
    LANG_ALLOWLIST,
    RAW_TOXICITIES_FOLDERS,
    TOXICITY_STAT_FOLDER_NAMES,
    get_holistic_bias_metadata,
    get_translations_with_metadata,
)


def compile_toxicity_stats(use_distilled_model: bool):

    # Params
    model_string = "distilled" if use_distilled_model else "base"
    groupby_column_lists = [
        ["axis"],
        ["axis", "bucket"],
        ["axis", "bucket", "descriptor"],
        ["noun"],
        ["template"],
        ["lang_string"],
    ]
    pivot_column_lists = [(["axis", "bucket", "descriptor"], "lang_string")]
    # Each tuple consists of index labels and a column label for the pivot

    # Save paths
    save_folder_name = TOXICITY_STAT_FOLDER_NAMES[model_string]
    save_folder = os.path.join(
        BASE_TOXICITY_SAVE_FOLDER,
        save_folder_name,
    )
    os.makedirs(save_folder, exist_ok=True)
    toxicity_count_path = os.path.join(save_folder, "toxicity_counts.csv")
    mean_toxicity_path_template = os.path.join(
        save_folder, "mean_toxicity__{index_label_string}.csv"
    )
    pivoted_toxicity_path_template = os.path.join(
        save_folder,
        "pivoted_mean_toxicity__{column_label}__by__{index_label_string}.csv",
    )

    metadata_df = get_holistic_bias_metadata(save_folder)

    # Load in and process toxicity files one at a time
    toxicity_stat_dfs = []
    for lang_string in tqdm(LANG_ALLOWLIST):
        this_toxicity_stat_df = get_translations_with_metadata(
            metadata_df=metadata_df,
            lang_string=lang_string,
            load_folder=RAW_TOXICITIES_FOLDERS[model_string],
        ).drop(["target_raw"], axis=1)
        toxicity_stat_dfs.append(this_toxicity_stat_df)

    toxicity_stat_df = pd.concat(toxicity_stat_dfs, axis=0)

    toxicity_count_df = (
        toxicity_stat_df[lambda df: df["has_toxicity"] == 1]
        .assign(axis=lambda df: df["axis"].fillna("(none)"))
        .assign(count=1)
        .groupby(["lang_string", "axis"])
        .agg({"count": "sum"})
    )
    print(f"Saving grouped counts of toxicity to {toxicity_count_path}.")
    toxicity_count_df.to_csv(toxicity_count_path)

    # Save groupbys
    for groupby_columns in groupby_column_lists:
        mean_toxicity_df = (
            toxicity_stat_df.groupby(groupby_columns)["has_toxicity"]
            .mean()
            .to_frame("frac_with_toxicity")
            .sort_values("frac_with_toxicity", ascending=False)
        )
        index_label_string = "_".join(
            [label.replace("_", "") for label in groupby_columns]
        )
        this_mean_toxicity_path = mean_toxicity_path_template.format(
            index_label_string=index_label_string
        )
        print(f"Saving mean toxicity file to {this_mean_toxicity_path}.")
        mean_toxicity_df.to_csv(this_mean_toxicity_path)

    # Pivot by language
    for index_labels, column_label in pivot_column_lists:
        mean_toxicity_df = (
            toxicity_stat_df.groupby(index_labels + [column_label])["has_toxicity"]
            .mean()
            .to_frame("frac_with_toxicity")
        )
        pivoted_toxicity_df = (
            pd.pivot_table(
                data=mean_toxicity_df,
                index=index_labels,
                columns=[column_label],
                values="frac_with_toxicity",
            )
            .assign(mean=lambda df: df.mean(axis=1))
            .sort_values("mean", ascending=False)
        )
        index_label_string = "_".join(
            [label.replace("_", "") for label in index_labels]
        )
        this_pivoted_toxicity_path = pivoted_toxicity_path_template.format(
            index_label_string=index_label_string, column_label=column_label
        )
        print(f"Saving pivoted mean toxicity file to {this_pivoted_toxicity_path}.")
        pivoted_toxicity_df.to_csv(this_pivoted_toxicity_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distilled",
        action="store_true",
        help="Export raw toxicities from the distilled model instead of the full model",
    )
    args = parser.parse_args()
    compile_toxicity_stats(use_distilled_model=args.distilled)
