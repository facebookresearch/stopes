# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from collections import defaultdict

import pandas as pd
from sacrebleu.tokenizers.tokenizer_zh import TokenizerZh
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

TOKENIZER_ZH = TokenizerZh()


def percent_correct_pairs(y: tp.Iterable[float], x: tp.Iterable[float]) -> float:
    """Given the targets y and predictions x, compute the percentage of pairs with different y that are correctly ordered by x.
    This metric is a multi-class extension of ROC AUC.
    """
    n_correct, n_total = 0, 0
    for x1, y1 in zip(x, y):
        for x2, y2 in zip(x, y):
            if y1 >= y2:
                continue
            n_total += 1
            n_correct += x1 < x2
    return n_correct / max(n_total, 1)


def get_word2chars_mapping(sentence: str, tokenizer=None) -> tp.List[tp.List[int]]:
    """Tokenize the sentence and return the list of character ids for each word."""
    if tokenizer is None:
        words = TOKENIZER_ZH(sentence).split()
    else:
        words = tokenizer(sentence)
    word2chars = []
    end = 0
    for word in words:
        start = sentence.find(word, end)
        assert start != -1
        end = start + len(word)
        word2chars.append(list(range(start, end)))
    return word2chars


def fit_predict_proba(
    data: pd.DataFrame,
    target_name: str,
    features: tp.List[str],
    model: tp.Optional[ClassifierMixin] = None,
    group_name: str = "row_id",
    n_folds: int = 3,
) -> pd.Series:
    """Predict probabilities by logistic regression based on a single feature or a group of them.
    In the second case, groupwise cross-validation is applied.
    """
    # fitting the model
    if model is None:
        model = LogisticRegression(random_state=1, max_iter=1000)
    if isinstance(target_name, str):
        target = data[target_name]
    inputs = pd.get_dummies(data[features])

    if inputs.shape[1] == 1:
        # for a single feature, we don't do any cross-validation,
        # because it does not affect the order
        model.fit(inputs, target)
        preds = model.predict_proba(inputs)[:, 1]
    else:
        kfg = GroupKFold(n_folds)
        preds = cross_val_predict(
            model,
            inputs,
            target,
            cv=kfg,
            method="predict_proba",
            groups=data[group_name],
        )[:, 1]
    return pd.Series(preds, index=data.index)


def evaluate_token_level_features(
    tokens_dataset: pd.DataFrame,
    feature_names: tp.List[str],
    word_to_character_maps: tp.List[tp.List[tp.List[int]]],
    word_labels: tp.List[tp.List[int]],
) -> tp.Tuple[tp.List[tp.List[float]], tp.Dict[str, float]]:
    """Compute word-level detection scores and ROC AUCs
    for each language in the given dataset w.r.t. the provided labels."""
    tokens_dataset["score"] = fit_predict_proba(
        tokens_dataset, "token_label", feature_names
    )
    word_level_preds = []
    directions = defaultdict(list)
    for i, (group_label, g) in enumerate(tokens_dataset.groupby("row_id")):
        # mapping each character to token
        c2t = {i: -1 for w in word_to_character_maps[i] for i in w}
        for token_id, row in enumerate(g.itertuples()):
            for char_id in range(row.start, row.end):
                c2t[char_id] = token_id
        assert all(t >= 0 for t in c2t.values())
        # finding constituent tokens for each word
        w2t = [
            sorted(set([c2t[c] for c in chars])) for chars in word_to_character_maps[i]
        ]
        # finding predictions for each token
        t2p = g["score"].tolist()
        # extracting predictions for each word by aggregating the token-level ones
        w2p = [max(t2p[i] for i in tok_ids) for tok_ids in w2t]
        word_level_preds.append(w2p)
        # memorize the direction to compute aggregate stats
        directions[g.direction.iloc[0]].append(i)
    direction2auc = {
        direction: roc_auc_score(
            [label for i in indices for label in word_labels[i]],
            [pred for i in indices for pred in word_level_preds[i]],
        )
        for direction, indices in directions.items()
    }
    return word_level_preds, direction2auc
