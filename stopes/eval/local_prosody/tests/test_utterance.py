# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from stopes.eval.local_prosody.utterance import Utterance


def test_utterance_format():
    """Check that Utterance correctly computes speech and pauses duration"""
    utterance = Utterance(
        words="the cat sat on the mat".split(),
        starts=[1, 2, 4, 5, 6, 7.1],
        ends=[2, 3, 5, 6, 7, 9.1],
    )
    assert utterance.net_duration == 7
    assert utterance.trimmed_duration == 8.1

    assert (
        utterance.get_text_with_markup(min_pause_duration=0.11)
        == "the cat [pause x 1.00] sat on the mat"
    )
    assert (
        utterance.get_text_with_markup(min_pause_duration=0.09)
        == "the cat [pause x 1.00] sat on the [pause x 0.10] mat"
    )
