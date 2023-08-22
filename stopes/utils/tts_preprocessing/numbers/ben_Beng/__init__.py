# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from stopes.utils.tts_preprocessing.numbers.ben_Beng.utils import (
    float_int_extraction,
    fraction_to_words,
    generate_segments,
    input_sanitizer,
    whole_part_word_gen,
)


def expand(number, lang="ben_Beng"):
    """
    Takes a number and outputs the word form in Bengali for that number.
    """

    assert (
        lang == "ben_Beng"
    ), f"This routine only supports `ben_Beng` but was given `{lang}`"

    generated_words = ""
    number = input_sanitizer(number)

    whole, fraction = float_int_extraction(number)

    whole_segments = generate_segments(whole)

    generated_words = whole_part_word_gen(whole_segments)

    if fraction:
        if generated_words:
            return generated_words + " দশমিক " + fraction_to_words(fraction)
        else:
            return "দশমিক " + fraction_to_words(fraction)
    else:
        return generated_words
