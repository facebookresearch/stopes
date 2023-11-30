# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2ForAudioFrameClassification

from stopes.eval.local_prosody.utterance import Utterance
from stopes.eval.word_alignment.alignment_utils import TAlignment


def load_frame_tagger(model_name_or_path: tp.Union[str, Path], use_cuda: bool = True):
    """Load a Wav2Vec2ForAudioFrameClassification for emphasis prediction"""
    model = Wav2Vec2ForAudioFrameClassification.from_pretrained(str(model_name_or_path))
    if use_cuda and torch.cuda.is_available():
        model.cuda()
    return model


def predict_emphasis(
    wav: torch.Tensor,
    model: Wav2Vec2ForAudioFrameClassification,
    utterance: Utterance,
    model_fps: tp.Optional[int] = 50,
    sample_rate: int = 16000,
) -> tp.Tuple[np.ndarray, tp.List[float], tp.List[float]]:
    """
    Make a frame-level emphasis prediction and project it onto the words in the given utterance.
    """
    if model_fps is None:
        # each convolutional layer reduces the sequence duration (and thus frame rate) by its stride
        model_fps = int(
            sample_rate
            / np.prod(
                [
                    layer.conv.stride[0]
                    for layer in model.wav2vec2.feature_extractor.conv_layers
                ]
            )
        )

    # Stage 1: predicting emphasis per frame
    with torch.inference_mode():
        # make the wav 1D by squeezing or by averaging accross channels
        while len(wav.shape) > 1:
            if wav.shape[0] == 1:
                wav = wav.squeeze(0)
            else:
                wav = wav.mean(0)
        model_out = model(wav.unsqueeze(0).to(model.device))
        proba = torch.softmax(model_out.logits, -1)[0, :, 1].cpu().numpy()

    # Stage 2: projecting emphasis onto words
    emph_means: tp.List[float] = []
    emph_maxes: tp.List[float] = []
    for start_second, end_second in zip(utterance.starts, utterance.ends):
        start_frame, end_frame = start_second * model_fps, end_second * model_fps
        start_id, end_id = int(np.floor(start_frame)), int(np.ceil(end_frame))
        chunk = proba[start_id:end_id]
        if len(chunk) == 0:
            # TODO: for very short words, maybe look at the neighbouring frames
            emph_means.append(0.0)
            emph_maxes.append(0.0)
        else:
            emph_means.append(float(chunk.mean()))
            emph_maxes.append(float(chunk.max()))

    return proba, emph_means, emph_maxes


def evaluate_emphasis_alignment(
    src_utts: tp.List[Utterance],
    tgt_utts: tp.List[Utterance],
    strong_alingments: tp.List[TAlignment],
    week_alignments: tp.List[TAlignment],
    weak_weight: float = 0.1,
    score_threshold: float = 0.5,
) -> tp.Tuple[tp.List[int], tp.List[int], tp.List[float], tp.List[float]]:
    """
    Compute metrics of emphasis alignment:
    - number of emphasized words in the source
    - number of emphasized words in the target
    - emphasis recall (how much of the source emphasis is correctly reflected in the target)
    - emphasis precision (how much of the target emphasis correctly reflects the source)
    The last two metrics are weighted by the number of alignment edges with an emphasized word on one end
    (with potentially a lower weight for weak alignments).
    """
    nsrcs, ntgts = [], []
    recalls, precisions = [], []
    for src_utt, tgt_utt, sa, wa in zip(
        src_utts, tgt_utts, strong_alingments, week_alignments
    ):
        emph_src = {
            word_id
            for word_id, score in enumerate(src_utt.emphasis_scores or [])
            if score >= score_threshold
        }
        emph_tgt = {
            word_id
            for word_id, score in enumerate(tgt_utt.emphasis_scores or [])
            if score >= score_threshold
        }
        nsrcs.append(len(emph_src))
        ntgts.append(len(emph_tgt))
        recalls.append(
            compute_alignment_recall(
                emph_src,
                emph_tgt,
                sa,
                wa,
                weak_weight=weak_weight,
                reverse_alignment=False,
            )
        )
        precisions.append(
            compute_alignment_recall(
                emph_tgt,
                emph_src,
                sa,
                wa,
                weak_weight=weak_weight,
                reverse_alignment=True,
            )
        )
    return nsrcs, ntgts, recalls, precisions


def compute_alignment_recall(
    sources: tp.Set[int],
    targets: tp.Set[int],
    strong_alignment: TAlignment,
    possible_alignment: TAlignment,
    weak_weight: float = 0.1,
    reverse_alignment: bool = False,
) -> float:
    """
    Compute the weighted proportion of alignment edges starting in the `sources` set
    that end in the `targets` set, with a lower weight for "possible" ("weak") alignments.
    """
    num, den = 0.0, 0.0
    for src_word in sources:
        for i, j in strong_alignment.union(possible_alignment):
            weight = 1 if (i, j) in strong_alignment else weak_weight
            if reverse_alignment:
                i, j = j, i
            if i == src_word:
                den += weight
                if j in targets:
                    num += weight
    if den == 0:
        return 1.0
    return num / den
