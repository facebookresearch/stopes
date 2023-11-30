# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
This module contains a short script for evaluating how different embeddings can predict prosodic similarity.
It relies on a .tsv file with the followint columns:
- src_audio (path to wavs)
- tgt_audoo (path to wavs)
- dataset (the name of the source dataset, used to split the data into groups by domain)
- median_$X, with X being overall_manner/emotions/emphasis/intonation/rhythm; that is, PCP labels.
"""

import typing as tp

import pandas as pd
import scipy
import torch
from tqdm.auto import tqdm

from .comparator_training import get_model_pred, split_data, train_comparator

PCP_DIMENSIONS = [
    "median_overall_manner",
    "median_emotions",
    "median_emphasis",
    "median_intonation",
    "median_rhythm",
    "median_semantics",
]

DEFAULT_COMPARATOR_CONFIG = dict(
    idim=1024,
    odim=5,
    nhid=[2048, 1024, 512],
    dropout=0.1,
    use_gpu=True,
    activation="TANH",
    input_form="qe",
    norm_emb=True,
    output_act=False,
)


class Prober:
    """A convenience class for comparing various speech sentence representations
    on the task of predicting prosodic consistency protocol scores.
    It expects a path to a dataset with PCP annotations.
    """

    def __init__(self, data_path, model_args=None, training_args=None):
        if isinstance(data_path, pd.DataFrame):
            self.dataset = data_path.copy()
        else:
            self.dataset = pd.read_csv(data_path, sep="\t", index_col=0)
        self.target_names = PCP_DIMENSIONS[:5]
        self.manner_names = PCP_DIMENSIONS
        self.default_config = DEFAULT_COMPARATOR_CONFIG.copy()
        if model_args is not None:
            self.default_config.update(model_args)
        self.training_args = dict(
            verbose=False,
            early_stopping=False,
            train_epoch=50,
        )
        if training_args is not None:
            self.training_args.update(training_args)

        # Experiment results
        self.training_results = {}
        self.predictions = {}
        self.scores = {}

    def compute_features(self, featurizer):
        """Compute features for the whole dataset using a given featurizer.
        The featurizer is expected to consume a path to the audio and to return an 1d tensor with its embedding.
        """
        features_src = torch.stack(
            [featurizer(audio_path) for audio_path in tqdm(self.dataset.src_audio)]
        )
        features_tgt = torch.stack(
            [featurizer(audio_path) for audio_path in tqdm(self.dataset.tgt_audio)]
        )
        return features_src, features_tgt

    def experiment(self, name: str, features: tp.Tuple[torch.Tensor, torch.Tensor]):
        """Using the given features, train a PCP predictor for each left-out domain in the training dataset,
        and evaluate their performance on these domains."""
        features_src, features_tgt = features
        cfg = self.default_config
        cfg["idim"] = features_src.shape[-1]

        # Training models for each dataset
        dataset2all = {}
        for data_label in self.dataset.dataset.drop_duplicates():
            train_split, val_split, test_split = split_data(
                features_src,
                features_tgt,
                torch.tensor(self.dataset[self.target_names].values).to(torch.float),
                self.dataset.dataset != data_label,
            )
            print(
                data_label,
                "train shape:",
                train_split[2].shape,
                "test_shape:",
                test_split[2].shape,
            )

            comparator, mean_losses, corrs = train_comparator(
                cfg, train_split, val_split, **self.training_args
            )
            full_pred = (
                get_model_pred(
                    comparator,
                    src=test_split[0],
                    ref=test_split[1],
                    mt=test_split[1],
                    use_gpu=comparator.use_gpu,
                )
                .cpu()
                .numpy()[:, 0]
            )
            dataset2all[data_label] = (comparator, mean_losses, corrs, full_pred)
        self.training_results[name] = dataset2all

        # Joining the predictions
        predictions = pd.Series(0, index=self.dataset.index)
        for k, (comparator, mean_losses, corrs, full_pred) in dataset2all.items():
            predictions.loc[self.dataset.dataset == k] = full_pred
        correlations = pd.DataFrame(
            {
                m: self.dataset.groupby("dataset").apply(
                    lambda d: scipy.stats.spearmanr(
                        predictions.loc[d.index], d[m]
                    ).correlation
                )
                for m in self.manner_names
            }
        )
        self.predictions[name] = predictions
        self.scores[name] = correlations

    def report(self):
        """Return Spearman correlations of predictions with human scores for each experiment, averaged over domains."""
        return pd.DataFrame({k: v.mean() for k, v in self.scores.items()})
