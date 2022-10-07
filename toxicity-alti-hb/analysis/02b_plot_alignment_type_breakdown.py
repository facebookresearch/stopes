# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
from util import BASE_TOXICITY_SAVE_FOLDER, LANG_NAMES_PATH, TOXICITY_SOURCE_FOLDER


def plot_alignment_type_breakdown():

    # Params
    alignment_category_display_names = {
        "descriptor": "Descriptor word",
        "template": "Template word",
        "noun": "Person noun",
    }
    sorted_categories = list(alignment_category_display_names.keys())
    sorted_category_display_names = [
        alignment_category_display_names[category] for category in sorted_categories
    ]
    num_langs_for_stacked_bars = 40

    # Plot params (horizontal)
    left_margin = 0.15
    bar_width = 0.9
    plot_width = 0.65

    # Plot params (vertical)
    bottom_margin = 0.05
    plot_height = 0.93

    # Load paths
    aligned_word_category_frac_path = os.path.join(
        TOXICITY_SOURCE_FOLDER, "aligned_word_category_fractions.csv"
    )
    aligned_word_category_frac_by_language_path = os.path.join(
        TOXICITY_SOURCE_FOLDER, "aligned_word_category_fractions_by_language.csv"
    )

    # Save results
    save_folder = os.path.join(
        BASE_TOXICITY_SAVE_FOLDER,
        "02b_plot_alignment_type_breakdown",
    )
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, "plot.png")

    print("Loading in toxicity results.")
    with open(LANG_NAMES_PATH) as f:
        # From the Flores-200 README
        lang_name_map = json.load(f)
    alignment_frac_df = (
        pd.read_csv(aligned_word_category_frac_path)
        .set_index("aligned_word_category")["frac"]
        .to_frame("(Overall)")
        .transpose()
        .loc[:, sorted_categories[::-1]]
    )
    # Reverse the order of the categories so that .cumsum() will allow the first
    # categories to be at the top of the plot
    alignment_frac_by_lang_df = pd.read_csv(aligned_word_category_frac_by_language_path)
    sorted_frac_per_lang_df = (
        alignment_frac_by_lang_df.set_index("lang_string")
        .sort_values("lang_count", ascending=False)
        .loc[:, sorted_categories[::-1]]
        .iloc[:num_langs_for_stacked_bars]
        .rename(index=lang_name_map)
    )
    cumul_frac_df = pd.concat(
        [alignment_frac_df, sorted_frac_per_lang_df], axis=0
    ).cumsum(axis=1)

    print("Creating plot.")
    fig = plt.figure(figsize=(11, 3.5), dpi=400)

    ax = fig.add_axes([left_margin, bottom_margin, plot_width, plot_height])
    handles = []
    for category in sorted_categories:
        handle = ax.bar(
            list(cumul_frac_df.index),
            cumul_frac_df[category].values * 100,
            bar_width,
        )
        handles.append(handle)
    ax.set_xlim([-0.6, cumul_frac_df.index.size - 0.4])
    ax.set_ylim([0, 100])
    ax.set_xticklabels(list(cumul_frac_df.index), rotation=90)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel("Distribution of toxic translations")
    ax.legend(
        handles,
        sorted_category_display_names,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.99),
    )

    print(f"Saving figure to {save_path}.")
    fig.savefig(save_path, bbox_inches="tight")


if __name__ == "__main__":
    plot_alignment_type_breakdown()
