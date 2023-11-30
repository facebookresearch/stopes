# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#for SPK in {01..10} ; do
for SPK in {01..10} ; do
  python3 preprocess_extraction.py \
    input_config.input_manifest=en_s${SPK}/manifest.tsv \
    input_config.output_folder=en_s${SPK} \
    input_config.nshards=6 &
done
