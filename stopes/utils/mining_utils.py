# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import functools
import logging
import re
import typing as tp
from pathlib import Path

from stopes.utils.data_utils import DataConfig

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


def _find_nl_file(
    lang: str,
    data_cfg: DataConfig,
) -> Path:
    nl_file = Path(data_cfg.data_shard_dir) / data_cfg.nl_file_template.format(
        lang=lang
    )
    assert nl_file.is_file(), f"ERROR: {nl_file} missing"
    return nl_file


def get_cached_line_count(
    lang: str,
    data_cfg: DataConfig,
    shard: tp.Optional[int] = None,
) -> int:
    """
    the xxx.nl file contains the number of lines for each shard of that lang. Sum this up.
    If you ask for a specific shard, return just the number for that shard.
    """
    nl_file = _find_nl_file(
        lang,
        data_cfg,
    )
    count = 0
    with nl_file.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if shard is not None and shard == idx:
                return int(line)
            count += int(line)
    return count


def get_cached_num_parts(
    lang: str,
    data_cfg: DataConfig,
) -> int:
    """
    the xxx.nl file contains the number of lines for each shard of that lang. Get number of shards
    from that.
    """
    nl_file = _find_nl_file(
        lang,
        data_cfg,
    )
    count = 0
    with nl_file.open("r", encoding="utf-8") as _:
        count += 1
    return count


def get_faiss_index_type(
    lang: str,
    data_cfg: DataConfig,
) -> str:
    nb_sent = get_cached_line_count(data_cfg=data_cfg, lang=lang)
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
    return "OPQ64,IVF1024,PQ64"


def extract_shard_id(filename: str, default: int = 0) -> int:
    """
    extract shard index from the input file name.
    """
    m = re.search(r"\.([0-9]{3})\.", filename)
    if m is None:
        return default
    return int(m[1])
