# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter
from itertools import groupby
from string import punctuation, whitespace

import emoji

PUNCT = punctuation + "—|–"
NUMBER = "0123456789"


def count_char_type(sent_counts: Counter, char_class: str) -> int:
    count = 0
    for c in char_class:
        count += sent_counts[c]
    return count


def count_emoji(sent_counts: Counter) -> int:
    total = 0
    for c, count in sent_counts.most_common():
        if emoji.is_emoji(c):
            total += count
    return total


def keep_it(
    sent: str,
    *,
    min_chars: int,
    max_chars: int,
    max_punct_ratio: float,
    max_number_ratio: float,
    min_space_ratio: float,
    max_space_ratio: float,
    max_emoji_ratio: float,
    max_repeated_char: int,
) -> bool:
    slen = len(sent)
    if slen < min_chars:
        # sentence too short
        return False
    if slen > max_chars:
        # sentence too long
        return False

    char_counts = Counter(sent)

    if count_char_type(char_counts, PUNCT) / slen > abs(max_punct_ratio):
        # too much punctuation
        return False

    if count_char_type(char_counts, NUMBER) / slen > abs(max_number_ratio):
        # too much numbers
        return False

    space_ratio = count_char_type(char_counts, whitespace) / slen
    if space_ratio < abs(min_space_ratio) or space_ratio > abs(max_space_ratio):
        # ratio of spaces to length of sentence is wrong
        return False

    if count_emoji(char_counts) / slen > abs(max_emoji_ratio):
        # too much emoji
        return False

    repeated_char = max(sum(1 for _ in g) for _, g in groupby(sent))

    if repeated_char > max_repeated_char:
        return False

    return True


def test_keep_it():
    def keep_it_test_func(
        string,
        min_chars=1,
        max_chars=200,
        max_punct_ratio=1,
        max_number_ratio=1,
        min_space_ratio=0,
        max_space_ratio=1,
        max_emoji_ratio=1,
        max_repeated_char=10,
    ):
        return keep_it(
            string,
            min_chars=min_chars,
            max_chars=max_chars,
            max_punct_ratio=max_punct_ratio,
            max_number_ratio=max_number_ratio,
            min_space_ratio=min_space_ratio,
            max_space_ratio=max_space_ratio,
            max_emoji_ratio=max_emoji_ratio,
            max_repeated_char=max_repeated_char,
        )

    assert keep_it_test_func(
        "I like this one",
        # test reasonable conditions
        min_chars=5,
        max_chars=200,
        max_punct_ratio=0.5,
        max_number_ratio=0.5,
        min_space_ratio=0.05,
        max_space_ratio=0.5,
        max_emoji_ratio=0.2,
        max_repeated_char=10,
    )

    assert not keep_it_test_func(
        "R72 a b c d e f g h                              i\tj",
        # this sentence has too many spaces, so it should be picked up by the space filter
        min_chars=10,
        max_chars=100,
        min_space_ratio=0.05,
        max_space_ratio=0.3,
    )

    assert not keep_it_test_func(
        "you like numbers? 123564136576101, 020, 03. 0800, 084",
        # this sentence has too many numbers
        min_chars=10,
        max_chars=100,
        max_number_ratio=0.1,
    )

    assert not keep_it_test_func(
        "short?",
        # this sentence is too short, we are testing min chars
        min_chars=100,
        max_chars=1000,
    )

    assert not keep_it_test_func(
        "long?",
        # this sentence is too long, we are testing max chars
        min_chars=1,
        max_chars=2,
    )

    assert not keep_it_test_func(
        "cats😺😸😹😻😼😽🙀😿😾",
        # this sentence has too many emojis
        max_emoji_ratio=0.1,
        max_repeated_char=10,
    )

    assert not keep_it_test_func(
        "ÀÄÄÄÄÍÍ´ NFO Last Updated: [09/04/98] By: Tklp ÃÍÍÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÙ[/ascii]",
        max_repeated_char=15,
    )

    assert not keep_it_test_func(
        "hahahahhahahhahahhahahaha",
        # this sentence doesn't have enough spaces
        min_space_ratio=0.05,
        max_space_ratio=0.5,
    )

    assert not keep_it_test_func(
        "afsdfasasdfsaf hahahahhahahhahahhahahaha",
        # this sentence doesn't have enough spaces
        min_space_ratio=0.05,
        max_space_ratio=0.5,
    )

    assert keep_it_test_func(
        "hello world this is a pretty normal sentence and we should definitely be keeping it, just making it a bit longer in case something is weird when there is a run on sentence like this!",
        # this is a good sentence
        min_space_ratio=0.05,
        max_space_ratio=0.5,
    )

    assert not keep_it_test_func(
        "R72 O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O V",
        # this is a real example where there are too many spaces
        min_space_ratio=0.05,
        max_space_ratio=0.3,
    )

    assert keep_it_test_func(
        "R72 O O O O Oafdsffasdfasfdasfasgadsfgawegesg O O O O O O O O afdsafdsafaafdsfsaO O O O O O O O Oafdsafsd O O O O O O O O V",
        # similar as above, but this actually should pass, i suppose it's a failure case of our filtering
        min_space_ratio=0.05,
        max_space_ratio=0.3,
    )
