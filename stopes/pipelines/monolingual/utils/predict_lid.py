# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from pathlib import Path

import fasttext
import numpy as np

logger = logging.getLogger("sentence_split")


def get_lid_predictor(
    model_file: Path,
    thresholds_file: tp.Optional[Path],
    label_unk: str = "__label__unk",
) -> tp.Callable[[str], tp.Tuple[str, float]]:
    thresholds_map = None
    nb_predict = 1

    assert model_file and model_file.exists(), f"invalid model file {model_file}"

    logger.info(f"FastText Model: {model_file}")

    if thresholds_file and thresholds_file.exists():
        checkpoint_data = np.load(thresholds_file, allow_pickle=True)
        threshold_after_prediction = False
        if len(checkpoint_data) == 2:  # backward compatibility
            labels, th = checkpoint_data
        else:
            labels, th, threshold_after_prediction = checkpoint_data

        thresholds_map = dict(zip(labels, th.tolist()))
        if not threshold_after_prediction:
            nb_predict = -1

        logger.info(f"Thresholds: {thresholds_file}")
        logger.info(f"Threshold after prediction: {threshold_after_prediction}")

    ft = fasttext.load_model(str(model_file))

    def predict_lid(sent: str) -> tp.Tuple[str, float]:
        labels, probs = ft.predict(sent, k=nb_predict, threshold=0.0)
        if thresholds_map and not threshold_after_prediction:
            n_probas = []
            for label, proba in zip(labels, probs):
                n_proba = -1
                if proba >= thresholds_map[label]:
                    n_proba = proba
                n_probas.append(n_proba)
            n_probas = np.array(n_probas)

            if not (n_probas > 0.0).any():
                return (label_unk, 0.0)
            else:
                bst = np.argmax(n_probas)
                return (labels[bst], probs[bst])
        else:
            predicted_label = labels[0]
            predicted_prob = probs[0]

            if thresholds_map and threshold_after_prediction:
                if predicted_prob < thresholds_map[predicted_label]:
                    predicted_label = label_unk
                    predicted_prob = 0.0
            return (predicted_label, predicted_prob)

    return predict_lid


def get_lid_predictor_date(
    model_date: str,
    label_unk: str = "__label__unk",
    lid_latest_models_path: Path = Path("."),
) -> tp.Callable[[str], tp.Tuple[str, float]]:

    model_file = lid_latest_models_path / (model_date + "_ft_model.bin")
    thresholds_file = lid_latest_models_path / (model_date + "_thresholds.npy")

    return get_lid_predictor(model_file, thresholds_file, label_unk=label_unk)
