# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import itertools
import logging
import typing as tp
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import MISSING
from scipy.stats import logistic
from sonar.models.blaser.loader import load_blaser_model

from stopes.core.stopes_module import Requirements, StopesModule
from stopes.modules.speech.video_alignement.video_utils import (
    VideoShardPair,
    build_segment_intersection_connection,
    get_strict_monotonic_alignment,
    pyarrow_fixed_size_array_to_numpy,
)
from stopes.utils.math_utils import normalize_l2, pairwise_cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class LocalAlignmentConfig:
    dataset_root: Path = MISSING
    output_dir: Path = MISSING
    output_parquet_name: str = "aligned"

    language_pairs: tp.Optional[tp.List[tp.List[str]]] = None
    """
    Filtering option to restrict only the given language pairs
    """
    sibling_ids: tp.Optional[tp.List[str]] = None
    """
    Filtering option to restrict only the subset of sibling_ids
    """

    """
    Basic segments similarity is computed as a weighted average of speech and text similarity
    with the relative weights parametrized by `speech_sim_weight` and `text_sim_weight`
    """
    speech_sim_weight: tp.Optional[float] = 1.0
    text_sim_weight: tp.Optional[float] = 1.0

    small_segment_boost: tp.Optional[float] = 7.0
    """
    It will multiply the original similarity scores by `logistic(Standartized(segment_duration) / small_segment_boost)`.
    Thus,
    - Smaller positive values (typically 1,...,5) mean that the greedy alignment algorithm will prefer larger segments.
    - With `small_segment_boost` = inf, no preferences to the segment length is accorded.
    - Taking `small_segment_boost` as a small negative number (-1,...,-5) will give more weight to the smaller segments.
    """

    sentence_force: float = 0.1
    """
    Given preference (by modifying similarity scores) to the segments that are the sentences.
    """

    max_normalized_gap: float = 2
    """
    We estimate the Guassian parameters of gap distribution over "good" aligned segments,
    and based on those estimates we filter the outliers whose deviation > `max_normalized_gap` * sigma(gap)
    """

    min_duration_ratio: tp.Optional[float] = 0.5
    min_speech_sim: tp.Optional[float] = 0.5
    min_text_sim: tp.Optional[float] = 0.5
    min_blaser_score: tp.Optional[float] = 3
    """
    Before saving the aligned segments,
    one may want to filter them using the parametrizable thresholds as above.
    The concrete threshold values may be dataset dependent in general.
    """


class LocalAlignmentModule(StopesModule):
    """

    Example to run locally
    >>> my_config = LocalAlignmentConfig(
    ...                 dataset_root=".../segments",
    ...                 output_dir="...",
    ...                 language_pairs=[["eng", "fra"]],
    ...                 sibling_ids=["id1", "id2"],
    ...                 max_gap=3,
    ...                 min_duration_ratio=0.5,
    ...                 min_text_sim=0.35,
    ...                 min_speech_sim=0.6,
    ...                 speech_sim_weight=2.,
    ...                 text_sim_weight=1.,
    ...                 small_segment_boost=10.)
    >>> lam = LocalAlignmentModule(my_config)
    >>> shard = lam.array()[0]
    >>> lam.run(shard)

    Output is path to aligned segments parquet dataset with the following schema:

    partition keys:
        * "sibling_id": [string]
        * "lang1": [string] - language of the first segment
        * "lang2": [string] - language of the second segment
        * "ts": [string] - Alignment Pipeline launch timestamp
    other columns:
       * "blaser_score": [double] blazer qe score averaged over two directions
       * "speech_sim": [double] - cos-similarity of SONAR speech embeddings
       * "text_sim": [double] - cos-similarity of text embeddings
       * "start1": [double]
       * "end1": [double]
       * "audio_path1": [string]
       * "text1": [string]
       * "start2": [double]
       * "end2": [double]
       * "text2": [string]
       * "audio_path2": [string]
    """

    config: LocalAlignmentConfig
    shards: tp.List[VideoShardPair]

    def array(self):
        return self.shards

    def requirements(self) -> Requirements:
        return Requirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=10,
            timeout_min=1080,
        )

    def __init__(self, config: LocalAlignmentConfig = LocalAlignmentConfig()):
        super().__init__(config, LocalAlignmentConfig)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.shards = self._load_shards()
        logger.info(
            f"Generating {len(self.shards)}  (sibling_id, lang pair) for iteration"
        )

        self.launch_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def run(
        self, iteration_value: VideoShardPair = None, iteration_index: int = 0
    ) -> Path:
        shard: VideoShardPair = iteration_value or self.shards[0]
        table1, table2 = self.load_pair_dataset(shard)

        # Pure whisper segments are aligned via monotonic alignment algorithm (traceback).
        # This's used to estimate `lag_mean` and `lag_std`.
        whipser1 = table1.filter(pa.compute.equal(table1["source"], "whisper"))
        whipser2 = table2.filter(pa.compute.equal(table2["source"], "whisper"))

        lag_mean, lag_std = self.lag_auto_detection(whipser1, whipser2)
        self.lag_std: float = max(lag_std, 3)

        logger.info(f"Detecting lag_mean={lag_mean} and lag_std={lag_std}")

        path_sentence = self.generate_sentence_candidate(
            table1, table2, lag_mean, lag_std
        )

        return path_sentence

    def generate_sentence_candidate(
        self, table1: pa.Table, table2: pa.Table, lag_mean: float, lag_std: float
    ) -> Path:
        typical_time_window: float = max(
            max(lag_std, 1) * self.config.max_normalized_gap, 3.0
        )

        segments_alignment = self.get_sentence_greedy_alignment(
            table1,
            table2,
            time_window=typical_time_window,
            lag=lag_mean,
        )
        logger.info(f"Generating {len(segments_alignment)} sentence candidates")
        output_path = self.create_candidate_ds_and_dump(
            table1, table2, segments_alignment, lag_mean
        )
        return output_path

    def get_sentence_greedy_alignment(
        self, table1: pa.Table, table2: pa.Table, time_window: float, lag: float
    ) -> np.ndarray:
        sims = self.compute_similarity(table1, table2)
        sims *= self.get_time_mask(table1, table2, time_window, lag)
        sims *= self.get_duration_weight(table1, table2)
        sims *= self.get_text_punctuation_weight(
            table1, table2, max_range=self.config.sentence_force
        )

        # We want a least one of two aligned segments to be a sentence
        sentence_mask1 = pa.compute.not_equal(table1["source"], "sentence").to_numpy()
        sentence_mask2 = pa.compute.not_equal(table2["source"], "sentence").to_numpy()
        sims *= 1 - (
            np.expand_dims(sentence_mask1, 1) * np.expand_dims(sentence_mask2, 0)
        )

        speech_embeds1 = pyarrow_fixed_size_array_to_numpy(table1["speech_embeddings"])
        speech_embeds2 = pyarrow_fixed_size_array_to_numpy(table2["speech_embeddings"])
        #  blazer scores are computed only for pairs of segments with `sims > 0.01` (which relatively lower)
        #  to reduce computations time
        blazer_sim = self.apply_blaser_model(
            speech_embeds1, speech_embeds2, sim_mask=sims > 0.01
        )
        # TODO reajust text and speech similarity properly
        sims = 0.4 * blazer_sim + 0.6 * sims

        adj1 = build_segment_intersection_connection(
            table1["start"].to_numpy(), table1["end"].to_numpy()
        )
        adj2 = build_segment_intersection_connection(
            table2["start"].to_numpy(), table2["end"].to_numpy()
        )

        # Greedy selection algorithm
        # This version is not optimized and can be slow for large number (> 10k) segments.
        best_matches_ = []
        while True:
            i, j = np.unravel_index(sims.argmax(), sims.shape)
            sim_val = sims[i, j]
            if sim_val > 0:  # taking everybody, we'll truncate downstream
                best_matches_.append((i, j))
            else:
                break
            # Removing adjacent segments from the similarity
            sims[adj1[i].indices, :] = 0
            sims[:, adj2[j].indices] = 0

        best_matches = np.array(best_matches_, dtype=np.int32)
        best_matches = best_matches[best_matches[:, 0].argsort()]
        return best_matches

    def lag_auto_detection(
        self, table1: pa.Table, table2: pa.Table, top: int = 150
    ) -> tp.Tuple[float, float]:
        """
        Mean and STD estimation of lags over possibly "aligned" segments
        """
        sims = self.compute_similarity(table1, table2)
        # we assume that alignment of larger segments is more important
        sims *= self.get_duration_weight(table1, table2, small_segment_boost=8)
        segments_alignment = get_strict_monotonic_alignment(sims)

        bigest_sim_idx = np.argsort(
            sims[segments_alignment[:, 0], segments_alignment[:, 1]]
        )[::-1]
        segments_alignment = segments_alignment[bigest_sim_idx[:top]]

        seg1 = (
            table1.select(["start", "end"]).take(segments_alignment[:, 0]).to_pandas()
        )
        seg2 = (
            table2.select(["start", "end"]).take(segments_alignment[:, 1]).to_pandas()
        )
        mid1 = (seg1["start"] + seg1["end"]) / 2
        mid2 = (seg2["start"] + seg2["end"]) / 2
        delta = mid1 - mid2
        return delta.mean(), delta.std()

    @staticmethod
    def get_time_mask(
        table1: pa.Table, table2: pa.Table, time_window: float, lag: float
    ) -> np.ndarray:
        ss1, ss2 = np.expand_dims(table1["start"], 1), np.expand_dims(
            table2["start"], 0
        )
        ee1, ee2 = np.expand_dims(table1["end"], 1), np.expand_dims(table2["end"], 0)
        delta_ss, delta_ee = np.abs(ss1 - ss2 - lag), np.abs(ee1 - ee2 - lag)
        weight_ss = np.where(
            delta_ss > 3 * time_window, 0, np.exp(-delta_ss / (3 * time_window))
        )
        weight_ee = np.where(
            delta_ee > 3 * time_window, 0, np.exp(-delta_ee / (3 * time_window))
        )
        return np.sqrt(weight_ss * weight_ee)

    def filter_candiate_pairs(self, candidates: pd.DataFrame) -> pd.DataFrame:
        candidates = candidates.loc[
            candidates["gap"] <= self.config.max_normalized_gap * self.lag_std
        ]
        logger.info(f"Keeping {len(candidates)} candidates after max_gap filter")

        candidates = candidates.loc[
            candidates["duration_ratio"] >= self.config.min_duration_ratio
        ]
        logger.info(
            f"Keeping {len(candidates)} candidates after min_duration_ratio filter"
        )

        candidates = candidates.loc[
            candidates["speech_sim"] >= self.config.min_speech_sim
        ]
        logger.info(f"Keeping {len(candidates)} candidates after min_speech_sim filter")

        candidates = candidates.loc[
            candidates["blaser_score"] >= self.config.min_blaser_score
        ]
        logger.info(
            f"Keeping {len(candidates)} candidates after min_blaser_score filter"
        )

        candidates = candidates.loc[candidates["text_sim"] >= self.config.min_text_sim]
        logger.info(f"Keeping {len(candidates)} candidates after text_sim filter")

        candidates.reset_index(inplace=True)
        return candidates

    def get_duration_weight(
        self,
        table1: pa.Table,
        table2: pa.Table,
        small_segment_boost: tp.Optional[float] = None,
    ) -> np.ndarray:
        if small_segment_boost is None:
            small_segment_boost = self.config.small_segment_boost

        def _get_duration_weight(dd: np.ndarray) -> np.ndarray:
            norm_dd = (dd - dd.mean()) / (dd.std() + 0.0001)
            return logistic.cdf(norm_dd / small_segment_boost)

        duration_weight1 = np.expand_dims(
            table1["end"].to_numpy() - table1["start"].to_numpy(), 1
        )
        duration_weight2 = np.expand_dims(
            table2["end"].to_numpy() - table2["start"].to_numpy(), 0
        )
        duration_weight1, duration_weight2 = map(
            _get_duration_weight, (duration_weight1, duration_weight2)
        )
        return np.sqrt(duration_weight1 * duration_weight2)

    @staticmethod
    def get_text_punctuation_weight(
        table1: pa.Table,
        table2: pa.Table,
        good_stops: tp.List[str] = list(".。!！?？"),
        bad_stops: tp.List[str] = list(",:')"),
        max_range: float = 0.1,
    ) -> np.ndarray:
        text1 = table1["text"].to_pandas().str.strip()
        text2 = table2["text"].to_pandas().str.strip()
        end_pos1 = text1.apply(lambda s: any(s.endswith(x) for x in good_stops)).values
        end_pos2 = text2.apply(lambda s: any(s.endswith(x) for x in good_stops)).values
        end_neg1 = text1.apply(lambda s: any(s.endswith(x) for x in bad_stops)).values
        end_neg2 = text2.apply(lambda s: any(s.endswith(x) for x in bad_stops)).values

        start_pos2 = text2.apply(
            lambda s: s[0].isupper() if len(s) > 0 else False
        ).values
        start_pos1 = text1.apply(
            lambda s: s[0].isupper() if len(s) > 0 else False
        ).values

        weight1 = np.ones_like(end_pos1, dtype=np.float32)
        weight1[end_pos1] += max_range
        weight1[end_neg1] -= 2 * max_range
        weight1[start_pos1] += max_range * 0.5

        weight2 = np.ones_like(end_pos2, dtype=np.float32)
        weight2[end_pos2] += max_range
        weight2[end_neg2] -= 2 * max_range
        weight2[start_pos2] += max_range * 0.5

        return np.sqrt(np.expand_dims(weight1, 1) * np.expand_dims(weight2, 0))

    def compute_similarity(self, table1: pa.Table, table2: pa.Table) -> np.ndarray:
        text_embeds1 = pyarrow_fixed_size_array_to_numpy(table1["text_embeddings"])
        text_embeds2 = pyarrow_fixed_size_array_to_numpy(table2["text_embeddings"])
        sims_text = pairwise_cosine_similarity(text_embeds1, text_embeds2)
        sims_text[np.isnan(sims_text)] = 0

        speech_embeds1 = pyarrow_fixed_size_array_to_numpy(table1["speech_embeddings"])
        speech_embeds2 = pyarrow_fixed_size_array_to_numpy(table2["speech_embeddings"])
        sims_speech = pairwise_cosine_similarity(speech_embeds1, speech_embeds2)
        sims_speech[np.isnan(sims_speech)] = 0

        # compute average similarity
        weights = np.array([self.config.text_sim_weight, self.config.speech_sim_weight])
        weights /= weights.sum()
        sims = sims_text * weights[0] + sims_speech * weights[1]

        logger.info(f"Similarity shape {sims.shape}")
        return sims

    def load_pair_dataset(self, shard: VideoShardPair) -> tp.Tuple[pa.Table, pa.Table]:
        table1 = pq.read_table(
            self.config.dataset_root,
            filters=[
                ("sibling_id", "=", shard.sibling_id),
                ("lang", "=", shard.lang1),
                ("ts", "=", shard.ts1),
            ],
        )
        table2 = pq.read_table(
            self.config.dataset_root,
            filters=[
                ("sibling_id", "=", shard.sibling_id),
                ("lang", "=", shard.lang2),
                ("ts", "=", shard.ts2),
            ],
        )
        logger.info(
            f"Shard with two segments datasets of length {len(table1)} and {len(table2)}"
        )
        return (table1.sort_by("start"), table2.sort_by("start"))

    @staticmethod
    def duration_ratio(joint_seg, epsilon=0.5):
        d1 = joint_seg["end1"] - joint_seg["start1"]
        d2 = joint_seg["end2"] - joint_seg["start2"]
        min_ = np.minimum(d1, d2)
        max_ = np.maximum(d1, d2)
        return np.where(min_ <= 0, 0.0, (min_ + epsilon) / (max_ + epsilon))

    @staticmethod
    def create_aligned_dataset(
        tt1: pa.Table, tt2: pa.Table, segments_alignment: np.ndarray, lag: float
    ) -> pd.DataFrame:
        mono_columns = [
            "sibling_id",
            "start",
            "end",
            "lang",
            "audio_path",
            "text",
            # "sampled_wav",  # could be to heavy to keep both wavs here
        ]
        index1 = segments_alignment[:, 0]
        index2 = segments_alignment[:, 1]
        seg1 = tt1.select(mono_columns).take(index1).to_pandas()
        seg2 = tt2.select(mono_columns).take(index2).to_pandas()
        seg1.rename(mapper="{}1".format, axis="columns", inplace=True)
        seg2.rename(mapper="{}2".format, axis="columns", inplace=True)
        joint_seg = pd.concat([seg1, seg2], axis=1)
        joint_seg["index1"] = index1
        joint_seg["index2"] = index2

        joint_seg = joint_seg[
            sorted(joint_seg.columns, key=lambda x: x[-1])
        ]  # more readable order
        joint_seg["sibling_id"] = joint_seg["sibling_id1"]
        joint_seg.drop(columns=["sibling_id1", "sibling_id2"], inplace=True)

        joint_seg["lag"] = [lag] * len(joint_seg)
        joint_seg["gap"] = np.maximum(
            joint_seg["start1"] - joint_seg["lag"] - joint_seg["end2"],
            joint_seg["start2"] + joint_seg["lag"] - joint_seg["end1"],
        ).clip(0)

        joint_seg["duration_ratio"] = LocalAlignmentModule.duration_ratio(joint_seg)

        for col in ["text", "speech"]:
            emb_cn = f"{col}_embeddings"
            if emb_cn in tt1.column_names:
                embeds1 = pyarrow_fixed_size_array_to_numpy(
                    tt1.select([emb_cn]).take(index1)[emb_cn]
                )
                embeds2 = pyarrow_fixed_size_array_to_numpy(
                    tt2.select([emb_cn]).take(index2)[emb_cn]
                )
                joint_seg[f"{col}_sim"] = (
                    normalize_l2(embeds1) * normalize_l2(embeds2)
                ).sum(axis=1)

                if col == "speech":  # adding blazer scores
                    blaser_qe = LocalAlignmentModule.load_blaser_model()
                    embeds1_torch, embeds2_torch = map(
                        torch.from_numpy, (embeds1, embeds2)
                    )
                    with torch.inference_mode():
                        joint_seg["blaser_score"] = (
                            blaser_qe(src=embeds1_torch, mt=embeds2_torch)
                            + blaser_qe(src=embeds2_torch, mt=embeds1_torch)
                        ).numpy() / 2.0
        return joint_seg

    def create_candidate_ds_and_dump(
        self,
        table1: pa.Table,
        table2: pa.Table,
        segments_alignment: np.ndarray,
        lag: float,
    ) -> Path:
        candidates = self.create_aligned_dataset(
            table1, table2, segments_alignment, lag=lag
        )
        candidates = self.filter_candiate_pairs(candidates)

        candidates["ts"] = pd.Series(
            [self.launch_ts] * len(candidates), dtype="category"
        )
        # dump
        candidates = pa.Table.from_pandas(candidates)
        output_path = self.config.output_dir / f"{self.config.output_parquet_name}"
        pq.write_to_dataset(
            candidates,
            output_path,
            partition_cols=["sibling_id", "lang1", "lang2", "ts"],
        )
        return output_path

    def _load_shards(self) -> tp.List[VideoShardPair]:
        """
        From the partitioned keys of segmentation parquet dataset,
        we extract all possible pairs in the forme `(sibling_id, lang1, lang2)`.
        Note that segmentation dataset is supposed to be multilingual.

        We also perform the basic filtering from config.
        In a case of multiple available runs for a given (sibling_id, lang),
        we choose the most recent one based on the timestamp `ts`

        Returns:
            tp.List[VideoShardPair]:
        """
        columns = ["sibling_id", "lang", "ts"]
        segmentation_parts = pq.read_table(
            self.config.dataset_root, columns=columns
        ).to_pandas()
        parts_counts = segmentation_parts.groupby(columns[:2])["ts"].nunique()
        if (parts_counts > 1).any():
            logger.warning(
                "Found several segmentations for a given sibling_id/lang. Taking the latest"
            )

        unique_parts = segmentation_parts.sort_values(
            "ts", ascending=True
        ).drop_duplicates(columns[:2])
        unique_parts = unique_parts.reset_index()
        logger.info(
            f"Dataset contains {len(unique_parts)} unique sibling_id/lang segmentation"
        )

        if self.config.sibling_ids is not None:
            unique_parts = unique_parts.loc[
                unique_parts["sibling_id"].isin(self.config.sibling_ids)
            ]
            logger.info(
                f"Selecting {len(unique_parts)} unique sibling_id/lang by sibling_ids restriction"
            )

        unique_parts["lang_ts"] = list(zip(unique_parts["lang"], unique_parts["ts"]))
        sibling_id_gr = unique_parts.groupby("sibling_id")[["lang_ts"]].agg(list)
        # filter our entries without pairs
        sibling_id_gr = sibling_id_gr.loc[sibling_id_gr["lang_ts"].apply(len) > 1]

        # taking all pairs
        sibling_id_gr["lang_ts"] = sibling_id_gr["lang_ts"].apply(
            lambda x: list(itertools.combinations(x, r=2))
        )
        sibling_id_gr = sibling_id_gr.explode("lang_ts")
        if self.config.language_pairs:
            lang_pairs = set(tuple(sorted(xx)) for xx in self.config.language_pairs)
            sibling_id_gr["lang_pair"] = sibling_id_gr["lang_ts"].apply(
                lambda x: tuple(sorted((x[0][0], x[1][0])))
            )
            sibling_id_gr = sibling_id_gr.loc[
                sibling_id_gr["lang_pair"].isin(lang_pairs)
            ]
            sibling_id_gr.drop(columns="lang_pair", inplace=True)
            logger.info(
                f"Selecting {len(sibling_id_gr)} unique sibling_id/lang pair by restriction on language pair"
            )
        # unrolling to dataclass
        shards: tp.List[VideoShardPair] = []
        for sibling_id_, row in sibling_id_gr.iterrows():
            en1, en2 = row["lang_ts"]
            shard = VideoShardPair(
                sibling_id=sibling_id_,
                lang1=en1[0],
                ts1=en1[1],
                lang2=en2[0],
                ts2=en2[1],
            )
            shards.append(shard)
        return shards

    @staticmethod
    @lru_cache(maxsize=3)
    def load_blaser_model(name="blaser_2_0_qe"):
        return load_blaser_model(name).eval()

    @staticmethod
    def apply_blaser_model(
        emb1: np.ndarray, emb2: np.ndarray, sim_mask: np.ndarray, bs=10_000
    ) -> np.ndarray:
        """
        Batch Inference method for a Blazer model for non-zeros `sim_mask` segments pairs.

        Returns:
            np.ndarray: blazer score for each pairs segments of the same shape as `sim_mask`
        """
        blaser_qe = LocalAlignmentModule.load_blaser_model()

        i1, i2 = np.nonzero(sim_mask)
        logger.info(
            f"Computing Blaser scores over {len(i1)} items with batch size={bs}"
        )
        result_ = []
        with torch.inference_mode():
            for it in range(len(i1) // bs + int(len(i1) % bs != 0)):
                inp1, inp2 = map(
                    torch.from_numpy,
                    (
                        emb1[i1[it * bs : (it + 1) * bs]],
                        emb2[i2[it * bs : (it + 1) * bs]],
                    ),
                )
                result_.append(
                    (
                        blaser_qe(src=inp1, mt=inp2).numpy()
                        + blaser_qe(src=inp2, mt=inp1).numpy()
                    )
                    / 10.0
                )

        result = np.concatenate(result_)

        sim_blazer = np.zeros_like(sim_mask)
        result[np.isnan(result)] = 0
        sim_blazer[i1, i2] = result.reshape(-1)
        return sim_blazer
