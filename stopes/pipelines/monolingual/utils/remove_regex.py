# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Remove non printable char as per:
#  https://stackoverflow.com/questions/92438/stripping-non-printable-characters-from-a-string-in-python
#
# This is supposed to be a drop in replacement to moses strip-non-printing-char.perl

import re
import sys
import typing as tp
import unicodedata


def regex_replacer(
    regex_pattern: str, replace_by: str = " ", flags: int = re.IGNORECASE
) -> tp.Callable[[str], str]:
    prog = re.compile(regex_pattern, flags=flags)

    def replacer(line) -> str:
        return prog.sub(replace_by, line)

    return replacer


def get_url_replacer(replace_by: str = " ") -> tp.Callable[[str], str]:
    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    return regex_replacer(url_pattern, replace_by)


def test_remove_url():
    replaceby_ = get_url_replacer("_")

    assert (
        replaceby_("bonjour bonjour http://www.facebook.com hello how")
        == "bonjour bonjour _ hello how"
    )

    replacebyspace = get_url_replacer(" ")

    assert replacebyspace("https://instagram.com/test files") == "  files"


def get_ascii_hashtag_replacer(replace_by: str = " ") -> tp.Callable[[str], str]:
    hashtag_pattern = r"[#@](\w\w\w\w+)"
    return regex_replacer(hashtag_pattern, replace_by, flags=(re.IGNORECASE | re.ASCII))


def test_remove_hashtag():
    replacebytag = get_ascii_hashtag_replacer("[HASHTAG]")

    assert (
        replacebytag("ከኢትዮዳዕዋ  አብዶ ከኢስታ #bonjour ከኢትዮዳዕዋ ቢንመሊክ ")
        == "ከኢትዮዳዕዋ  አብዶ ከኢስታ [HASHTAG] ከኢትዮዳዕዋ ቢንመሊክ "
    )


def get_non_printing_char_replacer(replace_by: str = " ") -> tp.Callable[[str], str]:
    non_printable_map = {
        ord(c): replace_by
        for c in (chr(i) for i in range(sys.maxunicode + 1))
        # same as \p{C} in perl
        # see https://www.unicode.org/reports/tr44/#General_Category_Values
        if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
    }

    def replace_non_printing_char(line) -> str:
        return line.translate(non_printable_map)

    return replace_non_printing_char


def test_remove():
    replaceby_ = get_non_printing_char_replacer("_")

    assert (
        replaceby_("See what's hidden in your string…   or be​hind﻿")
        == "See what's hidden in your string…   or be_hind_"
    )

    replacebyspace = get_non_printing_char_replacer(" ")

    assert replacebyspace("\x00\x11Hello\u200bWorld") == "  Hello World"
