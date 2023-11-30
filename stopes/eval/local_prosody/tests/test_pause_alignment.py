# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from stopes.eval.local_prosody.compare_utterances import (
    CompareUtterancesConfig,
    align_pauses,
    compute_pause_alignment_metrics,
    get_noncross_ratio,
)
from stopes.eval.local_prosody.utterance import Utterance


def test_get_noncross_ratio():
    # emptly alignment makes all pauses consistent with itself
    assert get_noncross_ratio(1, 1, set()) == 1

    strong_pairs = {(0, 0), (1, 1), (2, 2)}
    # both pauses after the word 1
    assert get_noncross_ratio(1, 1, strong_pairs) == 1
    # pauses after words 0 and 1; crossed by 1/3 edges
    assert get_noncross_ratio(0, 1, strong_pairs) == 2 / 3
    assert (
        get_noncross_ratio(0, 1, strong_pairs, weak_pairs={(3, 3)}, weak_weight=0.2)
        == 2.2 / 3.2
    )
    assert (
        get_noncross_ratio(0, 1, strong_pairs, weak_pairs={(0, 3)}, weak_weight=0.2)
        == 2.0 / 3.2
    )


def test_align_pauses():
    # src: Ich hasse es, [0.5] früh [0.2] aufzustehen
    # tgt: Je déteste [0.4] me lever [0.3] tôt
    base_pauses, pause_pairs, duration_scores, alignment_scores = align_pauses(
        pauses_after_src_words=[0, 0, 0.5, 0.2],
        pauses_after_tgt_words=[0, 0.4, 0, 0.3],
        strong_alignments={(0, 0), (1, 1), (0, 2), (3, 4), (4, 3)},
    )
    # we extract previos word id and pause duration for each source and target pauseß
    assert base_pauses == ([(2, 0.5), (3, 0.2)], [(1, 0.4), (3, 0.3)])
    # the first pause is aligned to the first, the second is to the second one
    assert pause_pairs == {(0, 0), (1, 1)}
    # the scores are reported for all source pauses and then for all target pauses
    assert duration_scores == [0.4 / 0.5, 0.2 / 0.3] * 2
    # the first pause pair is crossed by Ich/me wordalignment (1/5)
    # the second pair is crossed by früh/tôt and aufzustehen/lever
    assert alignment_scores == [4 / 5, 3 / 5] * 2


class MockAligner:
    def __init__(self, result):
        self.result = result

    def force_align_single_pair(self, src_words, tgt_words):
        return self.result


def test_pause_alignment_metrics():
    utterance1 = Utterance(
        words="Ich hasse es früh aufzustehen".split(),
        starts=[0, 1, 2, 3.5, 4.2],
        ends=[1, 2, 3, 4, 5],
    )
    utterance2 = Utterance(
        words="Je déteste me lever tôt".split(),
        starts=[0, 1, 2.4, 3, 4.3],
        ends=[1, 2, 3, 4, 5],
    )
    assert np.allclose(utterance1.get_pauses_after_words(), [0, 0, 0.5, 0.2, 0])
    assert np.allclose(utterance2.get_pauses_after_words(), [0, 0.4, 0, 0.3, 0])
    (
        word_alignment_results,
        pause_level_results,
        utt_level_results,
        speech_rate_corrs,
    ) = compute_pause_alignment_metrics(
        cfg=CompareUtterancesConfig(),
        src_utts=[utterance1],
        tgt_utts=[utterance2],
        word_aligner=MockAligner(
            result=({(0, 0), (1, 1), (0, 2), (3, 4), (4, 3)}, set())
        ),
    )
    print(word_alignment_results)
    assert word_alignment_results.word_alignment[0] == "0-0 0-2 1-1 3-4 4-3"
    assert np.allclose(utt_level_results.n_items, 4)
    # the expected scores were explained in test_align_pauses
    dur_scores = np.array([0.4 / 0.5, 0.2 / 0.3])
    al_scores = np.array([4 / 5, 3 / 5])
    assert np.allclose(utt_level_results.mean_duration_score, dur_scores.mean())
    assert np.allclose(utt_level_results.mean_alignment_score, al_scores.mean())
    assert np.allclose(
        utt_level_results.mean_joint_score, (dur_scores * al_scores).mean()
    )
