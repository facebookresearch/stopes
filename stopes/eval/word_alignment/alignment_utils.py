# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import typing as tp

TAlignment = tp.Set[tp.Tuple[int, int]]  # set of (src_word_id, tgt_word_id) tuples


def alignments_to_text(
    strong_alignment: TAlignment,
    weak_alignment: tp.Optional[TAlignment],
) -> str:
    """
    Convert word alignments (sets of word index pairs) to strings like `1-2 1p3`,
    where `-` indicates a strong (sure) alingment, and `p` indicates a weaker (possible) alignment.
    """
    items = []
    for x, y in sorted(strong_alignment):
        items.append(f"{x}-{y}")
    if weak_alignment is not None:
        for x, y in sorted(weak_alignment.difference(strong_alignment)):
            items.append(f"{x}p{y}")
    return " ".join(items)
