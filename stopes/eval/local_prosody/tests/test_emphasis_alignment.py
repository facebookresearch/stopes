# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from stopes.eval.local_prosody.emphasis_detection import evaluate_emphasis_alignment
from stopes.eval.local_prosody.utterance import Utterance


def test_evaluate_emphasis_alignment():
    utt1 = Utterance(words="Ich hasse es früh aufzustehen".split())
    utt2 = Utterance(words="Je déteste me lever tôt".split())
    sa = {
        (0, 0),
        (1, 1),
        (3, 4),
        (4, 3),
    }  # Ich=Je, hasse=déteste, früh=tôt, aufzustehen=lever
    wa = {(0, 2), (2, 1), (4, 2)}  # Ich=me, es=déteste, aufzustehen=me

    # Case 0: no annotation
    nsrcs, ntgts, recalls, precisions = evaluate_emphasis_alignment(
        [utt1], [utt2], [sa], [wa]
    )
    assert nsrcs == [0]
    assert ntgts == [0]
    assert recalls == [1]
    assert precisions == [1]

    # Case 1: perfect alignment
    # Ich hasse es, *früh* aufzustehen
    # Je déteste me lever *tôt*
    utt1.emphasis_scores, utt2.emphasis_scores = [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]
    nsrcs, ntgts, recalls, precisions = evaluate_emphasis_alignment(
        [utt1], [utt2], [sa], [wa]
    )
    assert nsrcs == [1]
    assert ntgts == [1]
    assert recalls == [1]
    assert precisions == [1]

    # Case 2: incomplete recall
    # Ich hasse *es*, *früh* aufzustehen
    # Je déteste me lever *tôt*
    utt1.emphasis_scores, utt2.emphasis_scores = [0, 0, 1, 1, 0], [0, 0, 0, 0, 1]
    nsrcs, ntgts, recalls, precisions = evaluate_emphasis_alignment(
        [utt1], [utt2], [sa], [wa], weak_weight=1
    )
    assert nsrcs == [2]
    assert ntgts == [1]
    assert recalls == [0.5]
    assert precisions == [1]
    nsrcs, ntgts, recalls, precisions = evaluate_emphasis_alignment(
        [utt1], [utt2], [sa], [wa], weak_weight=0.6
    )
    assert recalls == [1 / 1.6]

    # Case 3: incomplete precision (because `me` is weakly aligned to `Ich`)
    # Ich hasse es, *früh aufzustehen*
    # Je déteste *me lever tôt*
    utt1.emphasis_scores, utt2.emphasis_scores = [0, 0, 0, 1, 1], [0, 0, 1, 1, 1]
    nsrcs, ntgts, recalls, precisions = evaluate_emphasis_alignment(
        [utt1], [utt2], [sa], [wa], weak_weight=0.1
    )
    assert nsrcs == [2]
    assert ntgts == [3]
    assert recalls == [1]
    assert precisions == [2.1 / 2.2]
