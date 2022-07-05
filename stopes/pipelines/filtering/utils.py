#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unicodedata


def ngrams(string: str, order: int = 6):
    string = string.strip().replace(" ", "")
    return [string[i : i + order] for i in range(len(string) - order + 1)]


def normalize_unicode(string: str):
    normalized = unicodedata.normalize("NFKC", string)
    # normalize whitespace
    return " ".join(normalized.split())
