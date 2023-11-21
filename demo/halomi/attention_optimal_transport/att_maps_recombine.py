# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import pickle
import typing as tp
from collections import defaultdict

import numpy as np
import pandas as pd


def main(source_directory, laser_directory, att_maps_directory, output_file):
    dir2fn = defaultdict(list)
    for fn in os.listdir(source_directory):
        direction = fn[6:23]
        dir2fn[direction].append(fn)
    filtered_attmaps_by_dir_and_len = {}

    for direction in dir2fn:
        print(direction)
        data_parts: tp.List[pd.DataFrame] = []
        attmaps = []
        for fn in dir2fn[direction]:
            prefix = fn.split(".")[0]
            part = pd.read_csv(source_directory + fn, sep="\t", index_col=0)
            part["laser_sim"] = np.load(f"{laser_directory}/{prefix}.laser.npy")
            attmaps.extend(
                np.load(f"{att_maps_directory}/{prefix}.attmaps.npy", allow_pickle=True)
            )
            data_parts.append(part)
        dataset: pd.DataFrame = pd.concat(data_parts)
        print(dataset.shape, len(attmaps))

        # filtering by LASER similarity, by length ratio, and by loss
        filter_cols = ["len_ratio", "laser_sim", "loss"]
        filter_quantile = 0.2
        dataset["len_ratio"] = np.minimum(dataset.src_len, dataset.mt_len) / np.maximum(
            dataset.src_len, dataset.mt_len
        )
        thresholds = dataset[filter_cols].quantile(filter_quantile).to_dict()
        print(thresholds)
        mask = np.multiply(*[dataset[k] >= v for k, v in thresholds.items()])
        print(mask.mean())
        filtered_attmaps = [am for am, mask in zip(attmaps, mask) if mask]
        print(len(filtered_attmaps))

        len2maps = defaultdict(list)
        for att_map in filtered_attmaps:
            len2maps[len(att_map)].append(att_map)
        filtered_attmaps_by_dir_and_len[direction] = len2maps

        with open(output_file, "wb") as f:
            pickle.dump(filtered_attmaps_by_dir_and_len, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-directory", type=str, help="directory with translations"
    )
    parser.add_argument(
        "--laser-directory", type=str, help="directory with LASER scores"
    )
    parser.add_argument(
        "--att-maps-directory", type=str, help="directory with attention maps"
    )
    parser.add_argument(
        "--output-file", type=str, help="path for the output pickle file"
    )
