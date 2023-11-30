# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import csv
import logging
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
from omegaconf import DictConfig
from tqdm.auto import trange

from stopes.core.utils import promote_config
from stopes.eval.local_prosody.annotate_utterances import SUPPORTED_SPEECH_UNIT_NAMES
from stopes.eval.local_prosody.emphasis_detection import evaluate_emphasis_alignment
from stopes.eval.local_prosody.utterance import Utterance
from stopes.eval.word_alignment.aligners.awesome_align_wrapper import (
    AwesomeAlignWrapper,
)
from stopes.eval.word_alignment.alignment_utils import TAlignment, alignments_to_text

TPauses = tp.List[
    tp.Tuple[int, float]
]  # list of (previous_word_id, pause_duration) tuples

logger = logging.getLogger(__name__)


def short_long_ratio(x: float, y: float) -> float:
    """Similarity of two non-negative numbers, as their shortest-to-longest ratio"""
    if x == y == 0:
        return 1
    return min(x, y) / max(x, y)


def get_noncross_ratio(
    i: int,
    j: int,
    strong_pairs: TAlignment,
    weak_pairs: tp.Optional[TAlignment] = None,
    weak_weight: float = 0.1,
) -> float:
    """
    For an edge that connects the pauses after words i and j,
    compute the weighted proportion of word alignment edges that do not cross it.
    """
    i_mid, j_mid = i + 0.5, j + 0.5
    num, den = 0.0, 0.0
    for p1, p2 in strong_pairs:
        den += 1
        num += int((p1 - i_mid) * (p2 - j_mid) > 0)
    if weak_pairs:
        for p1, p2 in weak_pairs:
            den += weak_weight
            num += int((p1 - i_mid) * (p2 - j_mid) > 0) * weak_weight
    if den == 0:
        return 1
    return num / den


def align_pauses(
    pauses_after_src_words: tp.List[float],
    pauses_after_tgt_words: tp.List[float],
    strong_alignments: TAlignment,
    weak_alignments: tp.Optional[TAlignment] = None,
    weak_weight: float = 0.1,
    duration_ratio_power: int = 1,
    noncrossing_ratio_power: int = 1,
) -> tp.Tuple[tp.Tuple[TPauses, TPauses], TAlignment, tp.List[float], tp.List[float]]:
    """
    Find the most aligned pairs of pauses after source and target words.
    Alignment quality for a pair of pauses is evaluated as a product of two measures:
    - duration ratio (shortest to longest of the two)
    - non-crossing ratio: proportion of word alingment edges that do not cross the pause alignment edge.
    Optionally, one can provide extra "weak" word alignments that are assinged a lower weight.
    Return several artifacts:
        - base_pauses: two lists of (prev_word_id, duration) tuples, for source and target
        - pause_alignment: set of (int, int) tuples, which are indices of aligned pauses
        - duration_scores: list of floats, matched duration ratio score for each pause
        - alignment_scores: list of floats, matched alignment score for each pause
    """
    pp_src = [(i, d) for i, d in enumerate(pauses_after_src_words) if d > 0]
    pp_tgt = [(i, d) for i, d in enumerate(pauses_after_tgt_words) if d > 0]
    base_pauses = (pp_src, pp_tgt)
    n, m = len(pp_src), len(pp_tgt)

    if not pp_src or not pp_tgt:
        return (
            base_pauses,
            set(),
            [0] * (n + m),
            [0] * (n + m),
        )

    dur_ratios = np.array(
        [[short_long_ratio(si, sj) for j, sj in pp_tgt] for i, si in pp_src]
    )
    nc_ratios = np.array(
        [
            [
                get_noncross_ratio(
                    i, j, strong_alignments, weak_alignments, weak_weight=weak_weight
                )
                for j, sj in pp_tgt
            ]
            for i, si in pp_src
        ]
    )

    sims = nc_ratios**noncrossing_ratio_power * dur_ratios**duration_ratio_power
    # find such exclusive (source_id, target_id) pairs that maximize the sum of corresponding similarities
    src_match_ids, tgt_match_ids = scipy.optimize.linear_sum_assignment(-sims)
    pause_pairs: TAlignment = {(i, j) for i, j in zip(src_match_ids, tgt_match_ids)}

    p2a_src = {pp_src[i][0]: (i, j) for i, j in pause_pairs}
    p2a_tgt = {pp_tgt[j][0]: (i, j) for i, j in pause_pairs}

    duration_scores = []
    alignment_scores = []
    for (pauses, p2a) in [(pp_src, p2a_src), (pp_tgt, p2a_tgt)]:
        for prev_word_id, duration in pauses:
            if prev_word_id not in p2a:
                duration_scores.append(0.0)
                alignment_scores.append(0.0)
            else:
                i, j = p2a[prev_word_id]
                duration_scores.append(dur_ratios[i, j])
                alignment_scores.append(nc_ratios[i, j])
    return base_pauses, pause_pairs, duration_scores, alignment_scores


@dataclass
class CompareUtterancesConfig:
    src_path: tp.Optional[Path] = None
    tgt_path: tp.Optional[Path] = None
    src_column: str = "utterance"
    tgt_column: str = "utterance"
    result_path: tp.Optional[Path] = None
    pause_level_result_path: tp.Optional[Path] = None
    pause_min_duration: float = 0.1
    weak_alignment_weight: float = 0.1
    evaluate_emphasis_alignment: bool = False
    verbose: bool = True


def compute_pause_alignment_metrics(
    cfg: CompareUtterancesConfig,
    src_utts: tp.Optional[tp.List[Utterance]] = None,
    tgt_utts: tp.Optional[tp.List[Utterance]] = None,
    word_aligner=None,
) -> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tp.Optional[pd.DataFrame]]:
    """
    Given two lists of utterances (or a config that points to files with them),
    align the pauses in each their pair and compute the metrics of this alignment's quality.
    Return two dataframes with scores: on the pause level and on the utterance level.
    """
    if cfg.src_path:
        src_df = pd.read_csv(cfg.src_path, sep="\t", quoting=csv.QUOTE_MINIMAL)
    else:
        src_df = None
    if cfg.tgt_path:
        tgt_df = pd.read_csv(cfg.tgt_path, sep="\t", quoting=csv.QUOTE_MINIMAL)
    else:
        tgt_df = None

    if src_utts is None:
        if src_df is not None and cfg.src_column:
            src_raw = src_df[cfg.src_column]
            src_utts = [Utterance.deserialize(x) for x in src_raw]
        else:
            raise ValueError(
                "Please provide either `src_utts` argument, or `src_path` and `src_column` in the config."
            )
    if tgt_utts is None:
        if tgt_df is not None and cfg.tgt_column:
            tgt_raw = tgt_df[cfg.tgt_column]
            tgt_utts = [Utterance.deserialize(x) for x in tgt_raw]
        else:
            raise ValueError(
                "Please provide either `tgt_utts` argument, or `tgt_path` and `tgt_column` in the config."
            )

    # Stage 1: run word alignment
    if word_aligner is None:
        word_aligner = AwesomeAlignWrapper()
    strong_alignments = []
    weak_alignments = []
    for idx in trange(len(src_utts)):
        sa, wa = word_aligner.force_align_single_pair(
            src_utts[idx].words, tgt_utts[idx].words
        )
        strong_alignments.append(sa)
        weak_alignments.append(wa)

    # putting the results of word alingment to a separate dataframe, to reuse it later
    wadf = pd.DataFrame(
        {
            "id_src": [u.id for u in src_utts],
            "id_tgt": [u.id for u in tgt_utts],
            "src_utterance": [u.serialize() for u in src_utts],
            "tgt_utterance": [u.serialize() for u in tgt_utts],
            "word_alignment": [
                alignments_to_text(sa, wa)
                for sa, wa in zip(strong_alignments, weak_alignments)
            ],
        }
    )
    if cfg.evaluate_emphasis_alignment:
        # TODO: warn if there is no emphasis on the source or target side
        nsrcs, ntgts, recalls, precisions = evaluate_emphasis_alignment(
            src_utts, tgt_utts, strong_alignments, weak_alignments
        )
        wadf["emph_nsrc"] = nsrcs
        wadf["emph_ntgt"] = ntgts
        wadf["emph_recall"] = recalls
        wadf["emph_precision"] = precisions
        if cfg.verbose:
            print("Emphasis alignment average results:")
            means = wadf.mean(numeric_only=True)
            means["emph_recall_weighted"] = (
                wadf.emph_nsrc * wadf.emph_recall
            ).sum() / wadf.emph_nsrc.sum()
            means["emph_precision_weighted"] = (
                wadf.emph_ntgt * wadf.emph_precision
            ).sum() / wadf.emph_ntgt.sum()
            means["emph_f1"] = f1(means["emph_recall"], means["emph_precision"])
            means["emph_f1_weighted"] = f1(
                means["emph_recall_weighted"], means["emph_precision_weighted"]
            )
            means.drop(["id_src", "id_tgt"], inplace=True, errors="ignore")
            print(means)

    # Stage 2: run pause alignment
    results = []
    for idx in trange(len(strong_alignments)):
        utt_src, utt_tgt = src_utts[idx], tgt_utts[idx]
        sa, wa = strong_alignments[idx], weak_alignments[idx]
        srcp, tgtp = (
            utt_src.get_pauses_after_words(min_duration=cfg.pause_min_duration)[:-1],
            utt_tgt.get_pauses_after_words(min_duration=cfg.pause_min_duration)[:-1],
        )

        base_pauses, pause_pairs, duration_scores, alignment_scores = align_pauses(
            srcp,
            tgtp,
            sa,
            wa,
            weak_weight=cfg.weak_alignment_weight,
        )
        src_pauses, tgt_pauses = base_pauses
        s2t = {i: j for i, j in pause_pairs}
        t2s = {j: i for i, j in pause_pairs}

        sides = ["src" for _ in base_pauses[0]] + ["tgt" for _ in base_pauses[1]]
        pause_ids = [i for i, _ in enumerate(src_pauses)] + [
            i for i, _ in enumerate(tgt_pauses)
        ]
        words_before = [w for w, d in src_pauses] + [w for w, d in tgt_pauses]
        aligned_to = [s2t.get(i, -1) for i, _ in enumerate(src_pauses)] + [
            t2s.get(i, -1) for i, _ in enumerate(tgt_pauses)
        ]

        # if there are no pauses, we insert empty items,
        # just to preserve the utterance id and facilitate aggregation
        utt_results = {
            "row_id": idx,
            "utterance_id": utt_src.id if utt_src.id is not None else idx,
            "side": sides or ["none"],
            "pause_id": pause_ids or [-1],
            "word_before": words_before or [-1],
            "aligned_to": aligned_to or [-1],
            "pause_duration": [dur for pp in base_pauses for _, dur in pp] or [0],
            "duration_score": duration_scores or [1],
            "alignment_score": alignment_scores or [1],
        }
        results.append(pd.DataFrame(utt_results))
    pause_level_results = pd.concat(results)

    # Stage 3: aggregate to the utterance level
    utterance_level_results = pause_level_results.groupby("row_id").apply(
        aggregate_pause_alignment_statistics
    )

    if cfg.verbose:
        print(
            "Pause alignment results: "
            "micro-averaged (by pause) and macro-averaged (by utterance -> avg):"
        )
        print(
            pd.DataFrame(
                {
                    "micro_avg": aggregate_pause_alignment_statistics(
                        pause_level_results
                    ),
                    "macro_avg": utterance_level_results.mean(),
                }
            )
        )

    if cfg.result_path:
        result_df = pd.concat(
            [wadf, utterance_level_results.reset_index(drop=True)], axis=1
        )
        result_df.to_csv(cfg.result_path, sep="\t", index=None, quoting=csv.QUOTE_ALL)
        logger.info(
            f"Word alignents and sentence-level scores saved to {cfg.result_path}"
        )

    if cfg.pause_level_result_path:
        pause_level_results.to_csv(
            cfg.pause_level_result_path, sep="\t", index=None, quoting=csv.QUOTE_ALL
        )
        logger.info(f"Pause-level scores saved to {cfg.pause_level_result_path}")

    # Stage 4: compare speech rate, if it is present
    speech_rate_corrs = None
    if src_df is not None and tgt_df is not None:
        unit2corrs = {}
        for unit_name in SUPPORTED_SPEECH_UNIT_NAMES:
            col = f"speech_rate_{unit_name}"
            if col in src_df and col in tgt_df:
                unit2corrs[col] = {
                    "pearson": scipy.stats.pearsonr(src_df[col], tgt_df[col]).statistic,
                    "spearman": scipy.stats.spearmanr(
                        src_df[col], tgt_df[col]
                    ).correlation,
                }
        if unit2corrs:
            speech_rate_corrs = pd.DataFrame(unit2corrs).T
            print("Speech rate source-target correlations:")
            print(speech_rate_corrs)

    return wadf, pause_level_results, utterance_level_results, speech_rate_corrs


def f1(x, y, eps=1e-10):
    return x * y * 2 / np.maximum(x + y, eps)


def aggregate_pause_alignment_statistics(df: pd.DataFrame):
    joint_score = df.duration_score * df.alignment_score
    w = df.pause_duration
    non_empty = w.sum() > 0
    return pd.Series(
        {
            "mean_duration_score": df.duration_score.mean(),
            "mean_alignment_score": df.alignment_score.mean(),
            "mean_joint_score": joint_score.mean(),
            "wmean_duration_score": (df.duration_score * w).sum() / w.sum()
            if non_empty
            else 1,
            "wmean_alignment_score": (df.alignment_score * w).sum() / w.sum()
            if non_empty
            else 1,
            "wmean_joint_score": (joint_score * w).sum() / w.sum() if non_empty else 1,
            "total_weight": w.sum(),
            "n_items": df.shape[0],
            "n_src_pauses": (df.side == "src").sum(),
            "n_tgt_pauses": (df.side == "tgt").sum(),
        }
    )


@hydra.main(version_base="1.1")
def main(config: DictConfig):
    typed_config: CompareUtterancesConfig = promote_config(
        config, CompareUtterancesConfig
    )
    if (
        not typed_config.src_path
        or not typed_config.tgt_path
        or not typed_config.result_path
    ):
        raise ValueError(
            "When running pause alignment calculation from CLI, please provide a src_path, tgt_path, and result_path."
        )
    compute_pause_alignment_metrics(cfg=typed_config)


if __name__ == "__main__":
    main()
