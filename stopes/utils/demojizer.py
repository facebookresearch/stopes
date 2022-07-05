# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import emoji


class Demojizer:
    """
    based on:
    https://github.com/carpedm20/emoji/blob/d8bbfe455c6fcd12b96ed1dce6e0978fe7a47431/emoji/core.py#L141

    Copyright (c) 2014-2021, Taehoon Kim, Kevin Wurster and Tahir Jalilov
    All rights reserved.
    """

    def _get_search_tree(self):
        _SEARCH_TREE = {}
        for emj in emoji.unicode_codes.EMOJI_DATA:
            sub_tree = _SEARCH_TREE
            lastidx = len(emj) - 1
            for i, char in enumerate(emj):
                if char not in sub_tree:
                    sub_tree[char] = {}
                sub_tree = sub_tree[char]
                if i == lastidx:
                    sub_tree["data"] = emoji.unicode_codes.EMOJI_DATA[emj]
        return _SEARCH_TREE

    def __init__(self) -> None:
        self.search_tree = self._get_search_tree()

    def __call__(self, string: str, replace_str: str):
        result = []
        i = 0
        length = len(string)
        state = 0
        while i < length:
            consumed = False
            char = string[i]
            if char in self.search_tree:
                j = i + 1
                sub_tree = self.search_tree[char]
                while j < length and string[j] in sub_tree:
                    sub_tree = sub_tree[string[j]]
                    j += 1
                if "data" in sub_tree:
                    state = 1
                    consumed = True
                    result.append(replace_str)
                    i = j - 1
                else:
                    state = 0
            elif state == 1:
                if char.isspace():
                    consumed = True
                else:
                    state = 0

            if not consumed and char != "\ufe0e" and char != "\ufe0f":
                result.append(char)
            i += 1

        return "".join(result)


def test_demojizer():
    dem = Demojizer()

    assert dem("😺😸😹😻😼cats😽🙀😿😾", "") == "cats"
    # TODO this is probably not the most logical behaviour
    assert dem("more 😻😼😽 cats", "") == "more cats"
    # TODO same here
    assert dem("some 😻😼😽 cats", " ") == "some    cats"
    assert dem("no cats", " ") == "no cats"
    assert dem("", " ") == ""
