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
from util import (
    BASE_TOXICITY_SAVE_FOLDER,
    LANG_ALLOWLIST,
    LANG_NAMES_PATH,
    TOXICITY_STAT_FOLDER_NAMES,
)


def plot_toxicity_per_lang():

    # Params
    axis_colors = {
        "ability": "#D1BBD7",
        "age": "#AE76A3",
        "body_type": "#882E72",
        "characteristics": "#1965B0",
        "cultural": "#5289C7",
        "gender_and_sex": "#7BAFDE",
        "nationality": "#4EB265",
        "nonce": "#90C987",
        "political_ideologies": "#CAE0AB",
        "race_ethnicity": "#F7F056",
        "religion": "#F4A736",
        "sexual_orientation": "#E8601C",
        "socioeconomic_class": "#DC050C",
    }
    sorted_axes = sorted(list(axis_colors.keys()))
    axis_display_names = {
        axis: "Race and ethnicity"
        if axis == "race_ethnicity"
        else axis[0].upper() + axis[1:].replace("_", " ")
        for axis in sorted_axes
    }
    sorted_axis_names = sorted(list(axis_colors.keys()))
    sorted_axis_display_names = [axis_display_names[axis] for axis in sorted_axis_names]
    num_langs_for_stacked_bars = 40

    # Plot params (horizontal)
    left_margin = 0.10
    bottom_plot_width = 0.61
    bar_width = 0.9
    top_plot_width = 0.80

    # Plot params (vertical)
    bottom_margin = 0.20
    stacked_plot_height = 0.40
    middle_vmargin = 0.10
    line_plot_height = 0.25
    top_margin = 0.05
    assert (
        sum(
            [
                bottom_margin,
                stacked_plot_height,
                middle_vmargin,
                line_plot_height,
                top_margin,
            ]
        )
        == 1.0
    )

    # Load paths
    load_folder = os.path.join(
        BASE_TOXICITY_SAVE_FOLDER,
        TOXICITY_STAT_FOLDER_NAMES["base"],
    )
    frac_toxic_by_language_path = os.path.join(
        load_folder, "mean_toxicity__langstring.csv"
    )
    toxicity_count_path = os.path.join(
        load_folder,
        "toxicity_counts.csv",
    )

    # Save results
    save_folder = os.path.join(
        BASE_TOXICITY_SAVE_FOLDER,
        "00c_plot_toxicity_per_lang",
    )
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, "plot.png")

    print("Loading in toxicity results.")
    with open(LANG_NAMES_PATH) as f:
        # From the Flores-200 README
        lang_name_map = json.load(f)
    frac_toxic_by_language_df = pd.read_csv(frac_toxic_by_language_path)
    sorted_lang_df = (
        frac_toxic_by_language_df.set_index("lang_string")
        .loc[LANG_ALLOWLIST, :]
        .sort_values("frac_with_toxicity", ascending=False)
    )
    toxicity_count_df = pd.read_csv(toxicity_count_path)
    selected_langs = (
        sorted_lang_df.reset_index()["lang_string"]
        .iloc[:num_langs_for_stacked_bars]
        .values
    )
    axis_count_df = (
        toxicity_count_df.pivot(index="lang_string", columns="axis", values="count")
        .fillna(0)
        .loc[:, sorted_axis_names[::-1]]
    )
    # Reverse the order of the categories so that .cumsum() will allow the first
    # axes to be at the top of the plot
    overall_axis_count_df = axis_count_df.sum(axis=0).to_frame("(Overall)").transpose()
    combined_axis_count_df = pd.concat(
        [overall_axis_count_df, axis_count_df.loc[selected_langs]], axis=0
    )
    axis_cumul_frac_df = (
        combined_axis_count_df.div(combined_axis_count_df.sum(axis=1), axis=0)
        .cumsum(axis=1)
        .rename(index=lang_name_map)
    )

    print("Creating plot.")
    fig = plt.figure(figsize=(11, 8), dpi=400)

    # Top plot: frac toxic by lang rank
    ax = fig.add_axes(
        [
            left_margin,
            bottom_margin + stacked_plot_height + middle_vmargin,
            top_plot_width,
            line_plot_height,
        ]
    )
    ax.plot(sorted_lang_df["frac_with_toxicity"].values)
    ax.set_xlabel("Index of languages ordered by toxicity")
    ax.set_xlim([0, len(LANG_ALLOWLIST) - 1])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel("% translations marked toxic")
    ax.set_yscale("log")

    # Bottom plot: axis distribution by language
    ax = fig.add_axes(
        [left_margin, bottom_margin, bottom_plot_width, stacked_plot_height]
    )
    handles = []
    for axis in sorted_axes:
        handle = ax.bar(
            list(axis_cumul_frac_df.index),
            axis_cumul_frac_df[axis].values * 100,
            bar_width,
            color=axis_colors[axis],
        )
        handles.append(handle)
    ax.set_xlim([-0.6, axis_cumul_frac_df.index.size - 0.4])
    ax.set_ylim([0, 100])
    ax.set_xticklabels(list(axis_cumul_frac_df.index), rotation=90)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel("Distribution of toxic translations")
    ax.legend(
        handles,
        sorted_axis_display_names,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.99),
    )

    print(f"Saving figure to {save_path}.")
    fig.savefig(save_path, bbox_inches="tight")


if __name__ == "__main__":
    plot_toxicity_per_lang()
