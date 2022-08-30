# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import functools
import logging
import re
from pathlib import Path

logger = logging.getLogger("mining_utils")


def tokenization_type(lang: str):
    return _token_map().get(lang, lang)


@functools.lru_cache()
def _token_map():
    token_lang_file = Path(__file__).parent / "map_token_lang.tsv"
    with token_lang_file.open() as f:
        rows = csv.reader(f, delimiter="\t")
        return {row[0]: row[1] for row in rows}


def test_tokenization_types():
    assert tokenization_type("ara_Arab") == "ar"
    assert tokenization_type("ara_Latn") == "en"
    assert tokenization_type("swe") == "en"
    assert tokenization_type("arz") == "ar"
    assert tokenization_type("tgk") == "ru"
    # Those are suprising, but we used them for last data freeze.
    assert tokenization_type("fra") == "en"
    assert tokenization_type("rus") == "en"
    assert tokenization_type("oci") == "en"


def determine_faiss_index_type(nb_sent: int) -> str:
    if nb_sent > 500000000:
        return "OPQ64,IVF262144,PQ64"
    elif nb_sent > 100000000:
        return "OPQ64,IVF131072,PQ64"
    elif nb_sent > 10000000:
        return "OPQ64,IVF65536,PQ64"
    elif nb_sent > 4000000:
        return "OPQ64,IVF32768,PQ64"
    elif nb_sent > 700000:
        return "OPQ64,IVF16384,PQ64"
    elif nb_sent > 250000:
        return "OPQ64,IVF8192,PQ64"
    elif nb_sent > 160000:
        return "OPQ64,IVF4096,PQ64"
    elif nb_sent > 80000:
        return "OPQ64,IVF2048,PQ64"
    elif nb_sent > 10000:
        return "OPQ64,IVF1024,PQ64"
    return "Flat"  # this is faster, and with very few points the clustering fails


def extract_shard_id(filename: str, default: int = 0) -> int:
    """
    extract shard index from the input file name.
    """
    m = re.search(r"\.([0-9]{3})\.", filename)
    if m is None:
        return default
    return int(m[1])
