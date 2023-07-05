# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import string
import typing as tp
from collections import Counter, defaultdict
from pathlib import Path

adlm_ranges = [(0x1E900, 0x1E95F)]
arab_ranges = [(0x0600, 0x06FF), (0x0750, 0x077F), (0x0870, 0x089F), (0x08A0, 0x08FF)]
armn_ranges = [(0x0530, 0x058F)]
bali_ranges = [(0x1B00, 0x1B7F)]
beng_ranges = [(0x0980, 0x09FF)]
bugi_ranges = [(0x1A00, 0x1A1F)]
cans_ranges = [(0x1400, 0x167F), (0x18B0, 0x18FF), (0x11AB0, 0x11ABF)]
cyrl_ranges = [
    (0x0400, 0x04FF),
    (0x0500, 0x052F),
    (0x2DE0, 0x2DFF),
    (0xA640, 0xA69F),
    (0x1C80, 0x1C8F),
]
deva_ranges = [(0x0900, 0x097F), (0xA8E0, 0xA8FF)]
ethi_ranges = [
    (0x1200, 0x137F),
    (0x1380, 0x139F),
    (0x2D80, 0x2DDF),
    (0x1E7E0, 0x1E7FF),
    (0xAB00, 0xAB2F),
]
geor_ranges = [(0x10A0, 0x10FF), (0x1C90, 0x1CBF), (0x2D00, 0x2D2F)]
grek_ranges = [(0x0370, 0x03FF), (0x1F00, 0x1FFF)]
gujr_ranges = [(0x0A80, 0x0AFF)]
guru_ranges = [(0x0A00, 0x0A7F)]
han_ranges = [(0x3300, 0x33FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF)]
hang_ranges = [
    (0x1100, 0x11FF),
    (0xA960, 0xA97F),
    (0xAC00, 0xD7AF),
    (0xD7B0, 0xD7FF),
    (0x3130, 0x318F),
    (0xFFA0, 0xFFDC),
]
hebr_ranges = [(0x0590, 0x05FF)]
java_ranges = [(0xA980, 0xA9DF)]
jpan_ranges = [(0x3040, 0x309F), (0x30A0, 0x30FF), (0xFF60, 0xFF97)]
khmr_ranges = [(0x1780, 0x17FF), (0x19E0, 0x19FF)]
knda_ranges = [(0x0C80, 0x0CFF)]
laoo_ranges = [(0x0E80, 0x0EFF)]
latn_ranges = [
    (0x0000, 0x024F),
    (0x10780, 0x107BF),
    (0x1E00, 0x1EFF),
    (0x0250, 0x02AF),
    (0xA720, 0xA7FF),
    (0xAB30, 0xAB6F),
]
marc_ranges = [(0x11C70, 0x11CBF)]
mlym_ranges = [(0x0D00, 0x0D7F)]
mong_ranges = [(0x1800, 0x18AF), (0x11660, 0x1167F)]
mtei_ranges = [(0xAAE0, 0xAAF6), (0xABC0, 0xABFF)]
mymr_ranges = [(0x1000, 0x109F), (0xA9E0, 0xA9FF), (0xAA60, 0xAA7F)]
nkoo_ranges = [(0x07C0, 0x07FF)]
olck_ranges = [(0x1C50, 0x1C7F)]
orya_ranges = [(0x0B00, 0x0B7F)]
phag_ranges = [(0xA840, 0xA87F)]
sinh_ranges = [(0x0D80, 0x0DFF)]
sund_ranges = [(0x1B80, 0x1BBF), (0x1CC0, 0x1CCF)]
taml_ranges = [(0x0B80, 0x0BFF), (0x11FC0, 0x11FFF)]
telu_ranges = [(0x0C00, 0x0C7F)]
tfng_ranges = [(0x2D30, 0x2D7F)]
thai_ranges = [(0x0E00, 0x0E7F)]
tibt_ranges = [(0x0F00, 0x0FFF)]

SCRIPT_RANGES = {
    "Arab": arab_ranges,
    "Armn": armn_ranges,
    "Bali": bali_ranges,
    "Beng": beng_ranges,
    "Bugi": bugi_ranges,
    "Cans": cans_ranges,
    "Cyrl": cyrl_ranges,
    "Deva": deva_ranges,
    "Ethi": ethi_ranges,
    "Geor": geor_ranges,
    "Grek": grek_ranges,
    "Gujr": gujr_ranges,
    "Guru": guru_ranges,
    "Han": han_ranges,
    "Hang": hang_ranges,
    "Hebr": hebr_ranges,
    "Java": java_ranges,
    "Jpan": jpan_ranges,
    "Khmr": khmr_ranges,
    "Knda": knda_ranges,
    "Laoo": laoo_ranges,
    "Latn": latn_ranges,
    "Marc": marc_ranges,
    "Mlym": mlym_ranges,
    "Mong": mong_ranges,
    "Mtei": mtei_ranges,
    "Mymr": mymr_ranges,
    "Nkoo": nkoo_ranges,
    "Olck": olck_ranges,
    "Orya": orya_ranges,
    "Phag": phag_ranges,
    "Sinh": sinh_ranges,
    "Sund": sund_ranges,
    "Taml": taml_ranges,
    "Telu": telu_ranges,
    "Tfng": tfng_ranges,
    "Thai": thai_ranges,
    "Tibt": tibt_ranges,
}


def get_script_map(language_script_file: Path) -> tp.Dict[str, str]:
    """Returns a dict mapping a lang to its expected script in a single read run"""
    lang_map: tp.Dict[str, str] = defaultdict(str)
    with language_script_file.open("r", encoding="utf-8") as ls:
        for row in ls:
            columns = row.split("\t")
            lang_map[columns[0]] = columns[1]
    return lang_map


def find_lang_script(lang: str, language_script_file: Path) -> tp.Optional[str]:
    """Returns the expected script for a single lang"""
    with language_script_file.open("r", encoding="utf-8") as ls:
        for row in ls:
            if row.startswith(lang):
                columns = row.split("\t")
                return columns[1]
        return None


ScoredScript = tp.Tuple[tp.Optional[str], float]


def get_script_predictor() -> tp.Callable[[str], ScoredScript]:

    hist_map: tp.Dict[int, tp.Set[str]] = {}
    for key, ranges in SCRIPT_RANGES.items():
        for r in ranges:
            for ordinal in range(r[0], r[1] + 1):
                if ordinal not in hist_map:
                    hist_map[ordinal] = set()
                hist_map[ordinal].add(key)

    replace_by = ""  # we just get rid of characters that are ubiquitous
    replacement_map = {
        ord(c): replace_by
        for c in string.whitespace + string.punctuation + string.digits
    }

    def predict_script(sent: str) -> ScoredScript:
        sent = sent.translate(replacement_map)

        char_counts = Counter(sent).most_common()

        script_count: tp.Dict[str, int] = defaultdict(int)
        total = 0

        for char, count in char_counts:
            ordinal = ord(char)
            for script_name in hist_map.get(ordinal, []):
                total += count
                script_count[script_name] += count

        max_score = 0.0
        max_script = None
        for script, count in script_count.items():
            score = abs(count / total)
            if score > max_score:
                max_score = score
                max_script = script

        if len(script_count) > 1 and max_score == (1 / len(script_count)):
            return (None, 0)

        return (max_script, max_score)

    return predict_script


def test_predict_script():
    predictor_fn = get_script_predictor()

    assert predictor_fn("this is a latin script.") == ("Latn", 1.0)
    assert predictor_fn("isso é escrita latina 1234") == ("Latn", 1.0)
    assert predictor_fn("এটি বাংলা লিপি") == ("Beng", 1.0)
    assert predictor_fn("นี่คืออักษรไทย") == ("Thai", 1.0)
    assert predictor_fn(
        "자미로콰이 Jamiroquai는 영국의 애시드 재즈 밴드이다 자미로콰이는 년대 초반 런던에서 활발하게 일어난 애시드 재즈"
    ) == ("Hang", 0.8148148148148148)
    assert predictor_fn("이어지는기사  에서그점  에관해알려줄것  입니다") == ("Hang", 1.0)

    # not sure about this behaviour
    assert predictor_fn("এ 1234 b") == (None, 0)
    assert predictor_fn("এ.ก,b గ") == (None, 0)

    # empty
    assert predictor_fn(string.digits) == (None, 0)
    assert predictor_fn(string.whitespace) == (None, 0)
    assert predictor_fn("") == (None, 0)
