#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from glob import glob
from pathlib import Path
from typing import Dict

import yaml


def get_scaling_factors(path):
    devsets = glob(str(Path(path) / "dev" / "*.dev"))
    factors: Dict[str, float] = {}  # lang -> factor
    for devset in devsets:
        _, fname = os.path.split(devset)
        lang = fname[:-4].replace("-", "_")
        with open(devset, "rt") as fin:
            factors[lang] = len(fin.read())

    # rescale everything based on English
    return {lang: factors["eng"] / lang_factor for lang, lang_factor in factors.items()}


def main(args):
    args.data_conf_dir.mkdir(parents=True, exist_ok=True)
    with open(args.data_conf_dir / "length_factors.yaml", "wt") as fout:
        yaml.safe_dump(get_scaling_factors(args.flores_path), fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flores-path",
        type=Path,
        required=True,
        help="Location of FLORES data used for computing factors.",
    )
    parser.add_argument(
        "--data-conf-dir",
        type=Path,
        required=True,
        help="Directory where the configuration files are stored.",
    )
    args = parser.parse_args()
    main(args)
