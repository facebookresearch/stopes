#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
from collections import defaultdict
from glob import glob
from pathlib import Path

from omegaconf import OmegaConf

from stopes.pipelines.filtering.dataset import Dataset


def get_bt_datasets(root: Path, args):
    datasets = defaultdict(dict)  # direction -> corpus -> paths
    direction_directories = glob(str(root / "*-*"))
    for direction_directory in direction_directories:
        _, direction = os.path.split(direction_directory)
        src, tgt = direction.split("-")
        for src_path in glob(f"{direction_directory}/*.{src}.gz"):
            _, src_filename = os.path.split(src_path)
            src_filename = src_filename[:-3]
            corpus_name = src_filename[: src_filename.rfind(".")]
            if "EXCLUDE" in src_path or corpus_name in args.exclude_corpora:
                logging.debug(f"Excluding {src_path}")
                continue
            tgt_filename = f"{corpus_name}.{tgt}.gz"
            tgt_path = Path(direction_directory) / tgt_filename
            if not os.path.isfile(tgt_path):
                logging.warning(
                    f"Skipping {src_path}: the corresponding {tgt} file is missing"
                )
                continue
            datasets[direction][corpus_name] = Dataset(src=src_path, tgt=tgt_path)
    return dict(datasets)


def get_mined_datasets(root: Path, args):
    datasets = defaultdict(dict)

    for path in glob(str(root / "bitexts.mined" / "bitext.*.tsv.gz")):
        _, filename = os.path.split(path)
        direction = filename.split(".")[1]
        datasets[direction]["mined"] = Dataset(tsv=path)

    return dict(datasets)


def get_primary_datasets(paths, args):
    datasets = defaultdict(dict)  # direction -> corpus -> paths
    for path in paths:
        direction_directories = glob(str(path / "*-*"))
        for direction_directory in direction_directories:
            _, direction = os.path.split(direction_directory)
            src, tgt = direction.split("-")
            src_gz_glob = glob(f"{direction_directory}/*.{src}.gz")
            src_glob = glob(f"{direction_directory}/*.{src}")
            for src_path in src_gz_glob + src_glob:
                _, src_filename = os.path.split(src_path)
                if src_path.endswith(".gz"):
                    src_filename = src_filename[:-3]
                corpus_name = src_filename[: src_filename.rfind(".")]
                if "EXCLUDE" in src_path or corpus_name in args.exclude_corpora:
                    logging.debug(f"Excluding {src_path}")
                    continue
                tgt_filename = f"{corpus_name}.{tgt}"
                if src_path.endswith(".gz"):
                    tgt_filename += ".gz"
                tgt_path = Path(direction_directory) / tgt_filename
                if not os.path.isfile(tgt_path):
                    logging.warning(
                        f"Skipping {src_path}: the corresponding {tgt} file is missing"
                    )
                    continue

                assert (
                    corpus_name not in datasets[direction]
                ), f"duplicated direction {direction} for corpus {corpus_name}"
                datasets[direction][corpus_name] = Dataset(src=src_path, tgt=tgt_path)
    return dict(datasets)


def main(args):
    data_path = args.data_conf_dir / "unfiltered_corpora"
    data_path.mkdir(parents=True, exist_ok=True)
    if args.data_type == "train_primary":
        with open(data_path / "train_primary.yaml", "wt") as fout:
            fout.write(
                OmegaConf.to_yaml(
                    get_primary_datasets(args.primary_train_paths, args),
                    sort_keys=True,
                )
            )
    elif args.data_type == "train_mined":
        with open(data_path / "train_mined.yaml", "wt") as fout:
            fout.write(
                OmegaConf.to_yaml(
                    get_mined_datasets(args.mined_data_root, args),
                    sort_keys=True,
                )
            )
    elif args.data_type == "train_bt":
        with open(data_path / "train_bt.yaml", "wt") as fout:
            fout.write(
                OmegaConf.to_yaml(
                    get_bt_datasets(args.bt_root, args),
                    sort_keys=True,
                )
            )
    else:
        raise ValueError(args.data_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bt-root",
        type=Path,
        required=True,
        help="Location of backtranslated data (format: $direction/$corpus.$lang.gz).",
    )
    parser.add_argument(
        "--mined-data-root", type=Path, required=True, help="Location of mined data"
    )
    parser.add_argument(
        "--primary-train-paths",
        nargs="*",
        type=Path,
        required=True,
        help="Directories containing datasets (format: $direction/$corpus.$lang.gz).",
    )
    parser.add_argument(
        "--data-conf-dir",
        type=Path,
        required=True,
        help="Directory where the configuration files are stored.",
    )
    parser.add_argument(
        "--exclude-corpora",
        nargs="*",
        default=["testcorpus"],
    )
    parser.add_argument(
        "data_type",
        type=str,
        choices=["train_primary", "train_mined", "train_bt"],
        help="What type of data to populate the config for; "
        "choices: train_primary, train_mined, train_bt",
    )
    args = parser.parse_args()
    main(args)
