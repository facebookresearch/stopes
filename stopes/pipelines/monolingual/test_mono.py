# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from stopes.pipelines.monolingual.monolingual_line_processor import SentenceSplitClean


def test_split_clean_eng():
    splitter = SentenceSplitClean("eng", split_algo="default")
    text = (
        "Moses : /moʊzɪz/ is considered the most important prophet in Judaism[3][4] and"
        " one of the most important prophets in Christianity, Islam, the Druze"
        " faith,[5][6] the Bahá’í Faith and other Abrahamic religions. According    to"
        " both the Bible and the Quran,[7] Moses was the leader of the Israelites and"
        " lawgiver to whom the authorship, or «acquisition from heaven», of the Torah"
        " (the first five books of the Bible) is attributed…"
    )
    results = list(splitter(text))
    sentences = [s for (h, s, c) in results]
    cleaned = [c for (h, s, c) in results]

    assert len(sentences) == 2
    assert len(cleaned) == len(sentences)
    assert cleaned == [
        "Moses: /moʊzɪz/ is considered the most important prophet in Judaism[3][4] and"
        " one of the most important prophets in Christianity, Islam, the Druze"
        " faith,[5][6] the Bahá'í Faith and other Abrahamic religions.",
        "According to both the Bible and the Quran,[7] Moses was the leader of the"
        ' Israelites and lawgiver to whom the authorship, or "acquisition from heaven,"'
        " of the Torah (the first five books of the Bible) is attributed...",
    ]


def test_split_clean_zho_Hans():
    splitter = SentenceSplitClean("zho", split_algo="default")
    text = (
        "热带风暴尚塔尔是2001年大西洋飓风季的一场在8月穿越了加勒比海的北大西洋热带气旋。尚塔尔于8月14"
        "日由热带大西洋的一股东风波发展而成，其存在的大部分时间里都在快速向西移动，退化成东风波后穿越了"
        "向风群岛。"
    )
    results = list(splitter(text))
    sentences = [s for (h, s, c) in results]
    cleaned = [c for (h, s, c) in results]

    assert len(sentences) == 2
    assert len(cleaned) == len(sentences)
    assert cleaned == [
        "热带风暴尚塔尔是2001年大西洋飓风季的一场在8月穿越了加勒比海的北大西洋热带气旋。",
        "尚塔尔于8月14日由热带大西洋的一股东风波发展而成,其存在的大部分时间里都在快速向西移动,退化成东风波后穿越了向风群岛。",
    ]


def test_split_clean_tha():
    splitter = SentenceSplitClean("tha", split_algo="default")
    text = (
        "พระราชบัญญัติธรรมนูญการปกครองแผ่นดินสยามชั่วคราว พุทธศักราช ๒๔๗๕ "
        "เป็นรัฐธรรมนูญฉบับชั่วคราว ซึ่งถือว่าเป็นรัฐธรรมนูญฉบับแรกแห่งราชอาณาจักรสยาม "
        "ประกาศใช้เมื่อวันที่ 27 มิถุนายน พ.ศ. 2475 "
        "โดยเป็นผลพวงหลังการปฏิวัติเมื่อวันที่ 24 มิถุนายน พ.ศ. 2475 โดยคณะราษฎร"
    )
    results = list(splitter(text))
    sentences = [s for (h, s, c) in results]
    cleaned = [c for (h, s, c) in results]

    assert len(sentences) == 3
    assert len(cleaned) == len(sentences)
    assert cleaned == [
        "พระราชบัญญัติธรรมนูญการปกครองแผ่นดินสยามชั่วคราว พุทธศักราช ๒๔๗๕"
        " เป็นรัฐธรรมนูญฉบับชั่วคราว",
        "ซึ่งถือว่าเป็นรัฐธรรมนูญฉบับแรกแห่งราชอาณาจักรสยาม",
        "ประกาศใช้เมื่อวันที่ 27 มิถุนายน พ.ศ. 2475"
        " โดยเป็นผลพวงหลังการปฏิวัติเมื่อวันที่ 24 มิถุนายน พ.ศ. 2475 โดยคณะราษฎร",
    ]
