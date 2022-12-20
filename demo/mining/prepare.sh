#! /bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

echo "downloading martin monolingual dataset from 'https://statmt.org/wmt22/large-scale-multilingual-translation-task.html'"
wget -e robots=off -A gz -r -nd -np -nH https://data.statmt.org/martin/
for f in `ls *.gz`; do zcat $f | wc -l | awk 'END {print $1}' > "${f%.*}.nl"; done

echo "downloading LASER models"
curl -sSL 'https://raw.githubusercontent.com/facebookresearch/LASER/main/tasks/wmt22/download_models.sh' | LASER=. bash

# you can download more encoders using
# https://github.com/facebookresearch/LASER/tree/main/nllb/download_models.sh
# beware, these use different language codes.

echo "checkout demo/mining/README.md for the next steps."
