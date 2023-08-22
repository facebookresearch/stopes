# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Based on https://github.com/keithito/tacotron

import re
from functools import partial

import inflect
from num2words import num2words as num2words_orig

from stopes.utils.tts_preprocessing import numbers

_inflect = inflect.engine()
_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_space_number_re = re.compile(r"([0-9][0-9 ]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_percent_re = re.compile(r"([0-9\.\,]*[0-9]+[ ]*\%)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_ordinal_re = {
    "eng_Latn": re.compile(r"[0-9]+(st|nd|rd|th)"),
    "spa_Latn": re.compile(r"[0-9]+(ero|ro|do|to)"),
    "fra_Latn": re.compile(r"[0-9]+(er|ème)"),
    "nld_Latn": re.compile(r"[0-9]+(de|te)"),
    "deu_Latn": re.compile(r"[0-9]+ (te|ste)"),
    "ben_Beng": re.compile(r"[0-9]+ (ম|য়|ই|এ)"),
}
_number_re = re.compile(r"[0-9]+")
_cent_lang = {
    "eng_Latn": "cent",
    "spa_Latn": "centavo",
    "fra_Latn": "cent",
    "nld_Latn": "cent",
    "deu_Latn": "cent",
    "ben_Beng": "শতক",
}
_cents_lang = {
    "eng_Latn": "cents",
    "spa_Latn": "centavos",
    "fra_Latn": "centimes",
    "nld_Latn": "centen",
    "deu_Latn": "cent",
    "ben_Beng": "সেন্ট",
}
_dollar_lang = {
    "eng_Latn": "dollar",
    "spa_Latn": "dólar",
    "fra_Latn": "dollar",
    "nld_Latn": "dollar",
    "deu_Latn": "dolel_Bengar",
}
_dollars_lang = {
    "eng_Latn": "dollars",
    "spa_Latn": "dolares",
    "fra_Latn": "dollars",
    "nld_Latn": "dollars",
    "deu_Latn": "dollar",
    "ben_Beng": "ডলার",
}
_percent_lang = {
    "eng_Latn": "percent",
    "spa_Latn": "por ciento",
    "fra_Latn": "pour cent",
    "nld_Latn": "procent",
    "deu_Latn": "prozent",
    "ben_Beng": "শতাংশ",
}
_pound_lang = {
    "eng_Latn": "pounds",
    "spa_Latn": "libras",
    "fra_Latn": "livres",
    "nld_Latn": "pond",
    "deu_Latn": "pfund",
    "ben_Beng": "পাউন্ড",
}
_point_lang = {
    "eng_Latn": "point",
    "spa_Latn": "punto",
    "fra_Latn": "virgule",
    "nld_Latn": "punt",
    "deu_Latn": "punkt",
    "ben_Beng": "বিন্দু",
}

_num2words_codemap = {
    "amh_Ethi": "am",
    "arb_Arab": "ar",
    "ary_Arab": "ar",
    "arz_Arab": "ar",
    "ces_Latn": "cs",
    "dan_Latn": "dk",  # note: non-standard code
    "deu_Latn": "de",
    "eng_Latn": "en",
    "fin_Latn": "fi",
    "fra_Latn": "fr",
    "heb_Hebr": "he",
    "hun_Latn": "hu",
    "ind_Latn": "id",
    "ita_Latn": "it",
    "jap_Jpan": "ja",
    "kan_Knda": "kn",
    "kaz_Cyrl": "kz",
    "kor_Hang": "ko",
    "lit_Latn": "lt",
    "lvs_Latn": "lv",
    "nld_Latn": "nl",
    "nno_Latn": "no",
    "nob_Latn": "no",
    "pes_Arab": "fa",
    "pol_Latn": "pl",
    "por_Latn": "pt",
    "ron_Latn": "ro",
    "rus_Cyrl": "ru",
    "slv_Latn": "sl",
    "spa_Latn": "es",
    "srp_Cyrl": "sr",
    "swe_Latn": "sv",
    "tel_Telu": "te",
    "tgk_Cyrl": "tg",
    "tha_Thai": "th",
    "tur_Latn": "tr",
    "ukr_Cyrl": "uk",
    "vie_Latn": "vi",
}

SUPPORTED_LANGS = set(_num2words_codemap.keys()) | {"ben_Beng"}


def _remove_commas(m):
    return m.group(1).replace(",", "")


def _remove_space(m):
    clean_m = m.group(1).replace(" ", "")
    if len(clean_m) < len(m.group(1)) - 3:  # a sequence of digits, not like "20 000"
        return m.group(1)
    else:
        return clean_m


def _expand_percent(lang, m):
    return (
        m.group(0)
        .replace(" ", "")
        .replace("%", f" {_percent_lang[lang]}")
        .replace(",", ".")
    )


def num2words(d, to="cardinal", lang="eng_Latn"):
    assert lang in _num2words_codemap, f"Unsupported num2words language `{lang}`"
    return num2words_orig(d, to=to, lang=_num2words_codemap[lang])


def _expand_decimal_point(lang, m):
    if lang == "ben_Beng":
        func = numbers.ben_Beng.expand
    else:
        func = num2words
    pid = m.group(1).find(".")
    d_str = []
    for d in m.group(1)[pid + 1 :]:
        d_str.append(func(d, lang=lang))
    d_str = " ".join(d_str)
    return m.group(1).replace(m.group(1)[pid:], f" {_point_lang[lang]} {d_str}")


def _expand_dollars(lang, m):
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + f" {_dollars_lang[lang]}"  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = _dollar_lang[lang] if dollars == 1 else _dollars_lang[lang]
        cent_unit = _cent_lang[lang] if cents == 1 else _cents_lang[lang]
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = _dollars_lang[lang] if dollars == 1 else _dollars_lang[lang]
        return "%s %s" % (dollars, dollar_unit)
    elif cents:
        cent_unit = _cent_lang[lang] if cents == 1 else _cents_lang[lang]
        return "%s %s" % (cents, cent_unit)
    else:
        return f"{num2words(0, lang=lang)} {_dollars_lang[lang]}"


def _expand_ordinal(lang, m):
    if lang == "eng_Latn":
        return _inflect.number_to_words(m.group(0))
    elif lang == "ben_Beng":
        return numbers.ben_Beng.to_word(m.group(0)[: -len(m.group(1))])
    else:
        return num2words(m.group(0)[: -len(m.group(1))], to="ordinal", lang=lang)


def _expand_number(lang, m):
    num = int(m.group(0))
    if lang == "ben_Beng":
        return numbers.ben_Beng.to_word(num)
    if num > 1000 and num < 3000:
        if num == 2000:
            return num2words(num, lang=lang)
        elif num > 2000 and num < 2010:
            if lang == "eng_Latn":
                return "two thousand " + _inflect.number_to_words(num % 100)
            else:
                return (
                    num2words(2000, lang=lang) + " " + num2words(num % 100, lang=lang)
                )
        elif num % 100 == 0:
            if lang == "eng_Latn":
                return _inflect.number_to_words(num // 100) + " hundred"
            else:
                return num2words(num, lang=lang).replace(", ", " ")
        else:
            if lang == "eng_Latn":
                return _inflect.number_to_words(
                    num, andword="", zero="oh", group=2
                ).replace(", ", " ")
            else:
                return num2words(num, lang=lang).replace(", ", " ")
    else:
        if lang == "eng_Latn":
            return _inflect.number_to_words(num, andword="")
        else:
            return num2words(num, lang=lang)


def expand_numbers(text, lang="eng_Latn"):
    assert lang in SUPPORTED_LANGS, f"Unsupported language {lang}"

    text = re.sub(_space_number_re, _remove_space, text)
    text = re.sub(_percent_re, partial(_expand_percent, lang), text)
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, rf"\1 {_pound_lang[lang]}", text)
    text = re.sub(_dollars_re, partial(_expand_dollars, lang), text)
    text = re.sub(_decimal_number_re, partial(_expand_decimal_point, lang), text)
    text = re.sub(_ordinal_re[lang], partial(_expand_ordinal, lang), text)
    text = re.sub(_number_re, partial(_expand_number, lang), text)
    return text
