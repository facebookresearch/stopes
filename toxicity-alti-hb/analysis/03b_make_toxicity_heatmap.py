# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
from matplotlib.colors import LogNorm
from util import BASE_TOXICITY_SAVE_FOLDER


def make_toxicity_heatmap():

    # Plot params (horizontal)
    left_margin = 0.07
    plot_width = 0.41
    inner_margin = 0.11
    right_margin = 0.00
    assert (
        abs(
            sum(
                [
                    left_margin,
                    plot_width,
                    inner_margin,
                    plot_width,
                    right_margin,
                ]
            )
            - 1.0
        )
        < 1e-10
    )

    # Plot params (vertical)
    bottom_margin = 0.10
    plot_height = 0.85
    top_margin = 0.05
    assert (
        abs(
            sum(
                [
                    bottom_margin,
                    plot_height,
                    top_margin,
                ]
            )
            - 1.0
        )
        < 1e-10
    )

    # Load paths
    load_folder = os.path.join(
        BASE_TOXICITY_SAVE_FOLDER,
        "03_measure_source_contributions",
    )
    population_path = os.path.join(
        load_folder, "toxicity_by_contribution_and_robustness__count.csv"
    )
    toxicity_path = os.path.join(
        load_folder,
        "toxicity_by_contribution_and_robustness__some_aligned_descriptor_words_are_toxic.csv",
    )

    # Save results
    save_folder = os.path.join(
        BASE_TOXICITY_SAVE_FOLDER,
        "03b_make_toxicity_heatmap",
    )
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, "plot.png")

    print("Loading in binned results.")
    population_bin_df = pd.read_csv(population_path).set_index("contrib_bin")
    toxicity_bin_df = pd.read_csv(toxicity_path).set_index("contrib_bin")

    print("Creating plot.")
    fig = plt.figure(figsize=(11, 5.2), dpi=400)

    # Left: heatmap of overall distribution of samples (on a log scale)
    ax = fig.add_axes([left_margin, bottom_margin, plot_width, plot_height])
    create_heatmap(ax=ax, df=population_bin_df.transpose(), cmap="viridis_r")
    ax.set_title("Translation population distribution")

    # Right: the percent toxicity as a function of the mean source contribution and the
    # mean Gini impurity
    ax = fig.add_axes(
        [
            left_margin + plot_width + inner_margin,
            bottom_margin,
            plot_width,
            plot_height,
        ]
    )
    create_heatmap(ax=ax, df=toxicity_bin_df.transpose(), cmap="plasma_r")
    ax.set_title("Fraction of translations that are toxic")

    print(f"Saving figure to {save_path}.")
    fig.savefig(save_path)


def create_heatmap(ax, df: pd.DataFrame, cmap):
    """
    Create a heatmap with the given axes.
    """

    # Params
    toxicity_bin_corners = ((0, 90), (40, 100))
    # Axes are source contribution and Gini impurity
    toxicity_bin_lower_left, toxicity_bin_upper_right = toxicity_bin_corners
    toxicity_bin_width = toxicity_bin_upper_right[0] - toxicity_bin_lower_left[0]
    toxicity_bin_height = toxicity_bin_upper_right[1] - toxicity_bin_lower_left[1]

    left = 100 * float(df.columns.values[0]) - 2.5
    right = 100 * float(df.columns.values[-1]) + 2.5
    bottom = 100 * float(df.index.values[0]) - 2.5
    top = 100 * float(df.index.values[-1]) + 2.5
    im = ax.imshow(
        X=df.values,
        cmap=cmap,
        aspect="auto",
        origin="lower",
        extent=[left, right, bottom, top],
        norm=LogNorm(),
    )
    ax.figure.colorbar(im, ax=ax)
    ax.add_patch(
        patches.Rectangle(
            xy=toxicity_bin_lower_left,
            width=toxicity_bin_width,
            height=toxicity_bin_height,
            edgecolor="cyan",
            fill=False,
            lw=4,
        )
    )
    ax.set_xlabel("Source contribution to descriptor translation")
    ax.set_xlim([0, 80])
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_ylabel("Gini impurity of descriptor translations across nouns")
    ax.set_ylim([0, 100])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))


if __name__ == "__main__":
    make_toxicity_heatmap()
