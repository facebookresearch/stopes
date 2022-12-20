# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from stopes.pipelines.monolingual.monolingual_line_processor import SentenceSplitClean


def test_split_clean():
    splitter = SentenceSplitClean("eng", split_algo="default")
    text = (
        "Moses : /moʊzɪz/ is considered the most important prophet in Judaism[3][4]"
        " and one of the most important prophets in Christianity, Islam, the Druze faith,[5][6] the Bahá’í Faith and other Abrahamic religions."
        " According    to both the Bible and the Quran,[7] Moses was the leader of the Israelites and lawgiver to whom the authorship,"
        " or «acquisition from heaven», of the Torah (the first five books of the Bible) is attributed…"
    )
    results = list(splitter(text))
    sentences = [s for (h, s, c) in results]
    cleaned = [c for (h, s, c) in results]

    assert len(sentences) == 2
    assert len(cleaned) == len(sentences)
    assert cleaned == [
        "Moses: /moʊzɪz/ is considered the most important prophet in Judaism[3][4]"
        " and one of the most important prophets in Christianity, Islam, the Druze faith,[5][6] the Bahá'í Faith and other Abrahamic religions.",
        "According to both the Bible and the Quran,[7] Moses was the leader of the Israelites and lawgiver to whom the authorship,"
        ' or "acquisition from heaven," of the Torah (the first five books of the Bible) is attributed...',
    ]
