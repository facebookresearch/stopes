# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from stopes.eval.local_prosody.ctc_forced_aligner import Wav2Vec2ForcedAligner


def test_real_force_aligner():
    """Test that force aligner correctly predict timestamps"""
    aligner = Wav2Vec2ForcedAligner("philschmid/tiny-random-wav2vec2", device="cpu")

    # Create an array of predicted scores
    proba = torch.zeros([19, 32]) + 1e-10
    proba[:, aligner.blank_id] = 1e-5  # the most probable token is silence, by default
    labels = ["<pad>" if c == "." else c for c in "....THHHEE||C.AAT.."]
    for position, label in enumerate([aligner.label2id[label] for label in labels]):
        proba[position, label] += 1e-4

    # add one wrong character, to make the alignment really forced
    proba[16, aligner.label2id["D"]] = 2e-4

    emission = torch.log_softmax(torch.log(proba), -1)

    # just checking that the ASR postprocessing of the emissions is correct
    assert aligner.processor.tokenizer.decode(emission.argmax(1)) == "THE CAD"

    # Now run the force alignment
    utt = aligner.force_align_emission("THE|CAT", emission)
    # check that we have extracted the words that were forced
    assert utt.words == ["THE", "CAT"]
    # check that THHHEE occupies frames 4:10, and C.AAT occupies frames 12:17
    assert np.allclose(utt.starts, [4 / aligner.fps, 12 / aligner.fps])
    assert np.allclose(utt.ends, [10 / aligner.fps, 17 / aligner.fps])
