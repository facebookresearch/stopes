# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import logging
import typing as tp
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from stopes.core import utils


# TODO: move this class to a shared utils that can be used in the mine_indices step
class Alignments:
    """
    Class encapsulating the scores, source and target indices (i.e. positions) of
    bitexts mined by the mine_indexes step and persisted to disk. Allows for all three
    arrays to be kept in sync.
    """

    def __init__(
        self,
        scores: np.ndarray,
        src_idx: np.ndarray,
        trg_idx: np.ndarray,
        bwd_pos: tp.List[int],
    ):
        self.scores: np.ndarray = scores
        self.src_idx: np.ndarray = src_idx
        # TODO: mine_indexes module saves this as "trg", but we are using
        # "tgt" everywhere else (especially in our config files)
        self.tgt_idx: np.ndarray = trg_idx
        self.src_texts, self.tgt_texts = None, None
        self.src_meta, self.tgt_meta = None, None
        self.bwd_pos = set(bwd_pos)

    @property
    def has_meta(self) -> bool:
        return bool(self.src_meta and self.tgt_meta)

    @staticmethod
    def from_npz(file: Path, score_min: float, score_max: float) -> "Alignments":
        # loading into memory the alignments arrays jointly saved into a single npz
        # file by the mine_indexes module
        als = Alignments(**np.load(file))

        # keeping only alignments within a window and sorting by score
        als._filter_and_sort(score_min, score_max)

        return als

    def _filter_and_sort(self, score_min: float, score_max: float):
        condition = (score_min <= self.scores) & (self.scores <= score_max)
        self.scores = self.scores[condition]
        self.src_idx = self.src_idx[condition]
        self.tgt_idx = self.tgt_idx[condition]

        # sorting per score
        sorting = np.argsort(self.scores)[::-1]
        self.scores = self.scores[sorting]
        self.src_idx = self.src_idx[sorting]
        self.tgt_idx = self.tgt_idx[sorting]
        new_bwd_pos = set()  # reorder bwd positions
        for new_pos, old_pos in enumerate(sorting):
            if old_pos in self.bwd_pos:
                new_bwd_pos.add(new_pos)
        self.bwd_pos = new_bwd_pos

    def _load_data(self, indices, data_files) -> tp.Dict[int, str]:
        ids = set(indices)
        mined_data = {}
        pos = 0

        # reading in data_files and taking lines identified during mining
        for file_name in data_files:
            with utils.open(file_name) as file:
                for line in file:
                    if pos in ids:
                        mined_data[pos] = line.rstrip("\n")
                    pos += 1

        return mined_data

    def _load_all_data(
        self,
        src_text_files: tp.List[Path],
        tgt_text_files: tp.List[Path],
        src_meta_files: tp.Optional[tp.List[Path]],
        tgt_meta_files: tp.Optional[tp.List[Path]],
    ):
        # loading in source and target texts
        self.src_texts = self._load_data(self.src_idx, src_text_files)
        self.tgt_texts = self._load_data(self.tgt_idx, tgt_text_files)

        # loading in source and target metadata, if it exists
        if src_meta_files and tgt_meta_files:
            self.src_meta = self._load_data(self.src_idx, src_meta_files)
            self.tgt_meta = self._load_data(self.tgt_idx, tgt_meta_files)

    def save_texts(
        self,
        src_text_files: tp.List[Path],
        tgt_text_files: tp.List[Path],
        src_meta_files: tp.Optional[tp.List[Path]],
        tgt_meta_files: tp.Optional[tp.List[Path]],
        texts_out_path: Path,
        meta_out_path: Path,
        dedup_bitexts: bool,
        logger: logging.Logger,
    ):
        # reading in texts (and metadata if there is any)
        self._load_all_data(
            src_text_files, tgt_text_files, src_meta_files, tgt_meta_files
        )

        # initializing output tsvs
        texts_tsv_cm = utils.open(texts_out_path, mode="wt")
        meta_tsv_cm = utils.open(meta_out_path, mode="wt")

        # keeping track of bitexts to deduplicate if needed
        seen = set()

        # iterating over all bitext indices, saving corresponding texts/meta
        with texts_tsv_cm as texts_tsv, meta_tsv_cm as meta_tsv:
            texts_w = csv.writer(texts_tsv, delimiter="\t")
            meta_w = csv.writer(meta_tsv, delimiter="\t")

            for pos_score, (score, i_src, i_tgt) in enumerate(
                zip(self.scores, self.src_idx, self.tgt_idx)
            ):
                # preventively checking if a given key is missing: technically
                # there might be a mistmatch between the faiss index and the texts
                if i_src not in self.src_texts or i_tgt not in self.tgt_texts:
                    logger.warning(
                        f"Could not find bitext ({i_src, i_tgt}) in text files"
                    )
                    continue

                # deduplicating bitexts if needed
                if dedup_bitexts:
                    bitext = self.src_texts[i_src] + "+" + self.tgt_texts[i_tgt]
                    if bitext in seen:
                        continue
                    else:
                        seen.add(bitext)

                # saving the corresponding bitexts along with their score
                texts_w.writerow([score, self.src_texts[i_src], self.tgt_texts[i_tgt]])

                # saving the corresponding metadata if it exists
                meta_w.writerow(
                    [
                        score,
                        self.src_meta[i_src] if self.src_meta else "",
                        self.tgt_meta[i_tgt] if self.tgt_meta else "",
                        "bwd" if pos_score in self.bwd_pos else "fwd",
                    ]
                )
