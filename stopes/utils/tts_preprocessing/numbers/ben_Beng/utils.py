# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

from stopes.utils.tts_preprocessing.numbers.ben_Beng.words import numeric_words, units


def input_sanitizer(number):
    if isinstance(number, float) or isinstance(number, int) or isinstance(number, str):
        if isinstance(number, str):
            try:
                if "." in number:
                    number = float(number)
                else:
                    number = int(number)
            except ValueError:
                return None
        return number
    else:
        return None


def generate_segments(number):
    """
    Generating the unit segments such as koti, lokkho
    """
    segments = dict()
    segments["koti"] = math.floor(number / 10000000)
    number = number % 10000000
    segments["lokkho"] = math.floor(number / 100000)
    number = number % 100000
    segments["hazar"] = math.floor(number / 1000)
    number = number % 1000
    segments["sotok"] = math.floor(number / 100)
    number = number % 100
    segments["ekok"] = number

    return segments


def float_int_extraction(number):
    """
    Extracting the float and int part from the passed number. The first return
    is the part before the decimal point and the rest is the fraction.
    """
    _number = str(number)
    if "." in _number:
        return tuple([int(x) for x in _number.split(".")])
    else:
        return number, None


def whole_part_word_gen(segments):
    """
    Generating the Bengali word for the whole part of the number
    """
    generated_words = ""
    for segment in segments:
        if segments[segment]:
            generated_words += (
                numeric_words[str(segments[segment])] + " " + units[segment] + " "
            )

    return generated_words[:-2]


def fraction_to_words(fraction):
    """
    Generating Bengali words for the part after the decimal point
    """
    generated_words = ""
    for digit in str(fraction):
        generated_words += numeric_words[digit] + " "
    return generated_words[:-1]
