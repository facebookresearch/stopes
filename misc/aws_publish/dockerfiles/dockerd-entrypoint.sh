#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -ex

echo "will load model" $MODEL_STORE $MODEL_NAME
printenv
echo "--- model store content: $MODEL_STORE ---"
if [[ -d $MODEL_STORE ]] ; then
    ls $MODEL_STORE/*
fi

echo "--- working dir content: ---"
ls .

torchserve --start --foreground \
    --ts-config /home/model-server/config.properties \
    --model-store $MODEL_STORE \
    --models $MODEL_NAME \
