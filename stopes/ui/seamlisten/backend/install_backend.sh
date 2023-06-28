#! /bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

conda create -n seamlisten python=3.10
conda activate seamlisten
pip install -r requirements.txt
# According to https://github.com/pytorch/audio/issues/2363#issuecomment-1179089175
# torchaudio >= 0.12 switched codec and requires ffmpeg 4 to load .mp3 files.
conda install 'ffmpeg<5'
