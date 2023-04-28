#! /bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

echo "Downloading eval data in data"
mkdir data
wget -nH -nd -c -e robots=off http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/datasets/GigaS2S/EN2ZH/TEST.zip -P data/
cd data
unzip TEST.zip
cd -

echo "Downloading LASER speech encoders in encoders"
mkdir encoders
wget -nH -nd -c -e robots=off http://dl.fbaipublicfiles.com/speechlaser_encoders/english.pt  http://dl.fbaipublicfiles.com/speechlaser_encoders/mandarin.pt -P encoders/

echo "preparing the input manifest for English"
pip install func_argparse
python mk_manifest.py manifest -i data/en > data/en/source.tsv

echo "downloading blaser model"
wget -nH -nd -c -e robots=off http://dl.fbaipublicfiles.com/blaser/blaser.tar.gz
mkdir blaser_model
tar -zxf blaser.tar.gz -C blaser_model

echo "Checkout the README on how to run the blaser eval"
