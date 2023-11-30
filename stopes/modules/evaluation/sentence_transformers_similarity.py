# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import logging
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from stopes.core.stopes_module import Requirements, StopesModule

logger = logging.getLogger(__name__)


@dataclass
class SentenceTransformersSimilarityConfig:
    input_glob: tp.Optional[str] = None
    input_files: tp.Optional[tp.List[Path]] = None
    model_name_or_path: str = "sentence-transformers/LaBSE"
    normalization: bool = (
        False  # LaBSE already normalizes the vectors; no need to redo it
    )
    src_column: str = "lang1_text"
    tgt_column: str = "lang2_text"
    result_column: str = "text_sim"
    tsv_quoting: int = 3
    batch_size: int = 16
    result_file_suffix: str = "labse_sim"


class SentenceTransformersSimilarityModule(StopesModule):
    """
    Compute LaBSE (or another sentence transformer) embedding similarity on the input bitext dataset,
    and save the result to a .tsv file next to the input file
    """

    def __init__(self, config: tp.Any):
        super().__init__(config, SentenceTransformersSimilarityConfig)
        self.config: SentenceTransformersSimilarityConfig
        if self.config.input_glob:
            self.input_files = [
                Path(filename) for filename in glob.glob(self.config.input_glob)
            ]
        elif self.config.input_files:
            self.input_files = self.config.input_files
        else:
            raise ValueError("Either input_glob or input_files must be specified")

        if SentenceTransformer is None:
            raise ImportError(
                "Please install the sentence_transformers package to run SentenceTransformersSimilarityModule"
            )

    def requirements(self) -> Requirements:
        return Requirements(gpus_per_node=1)

    def array(self):
        return self.input_files

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> tp.Any:
        input_file = iteration_value
        output_file = Path(f"{input_file}.{self.config.result_file_suffix}")

        logger.info(f"Loading data from: {input_file}")
        data = pd.read_csv(input_file, sep="\t", quoting=self.config.tsv_quoting)
        result = self.process_dataset(data)
        result.to_csv(
            output_file, sep="\t", index_label="idx", quoting=self.config.tsv_quoting
        )
        logger.info("Similarity computation finished:")
        logger.info(str(result.describe()))
        logger.info(f"Saving to: {output_file}")
        return output_file

    def process_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create a new dataset with three columns: source and target texts and their similarities"""
        src_texts = data[self.config.src_column].tolist()
        tgt_texts = data[self.config.tgt_column].tolist()

        model = SentenceTransformer(self.config.model_name_or_path)

        embs1 = model.encode(
            src_texts, show_progress_bar=True, batch_size=self.config.batch_size
        )
        embs2 = model.encode(
            tgt_texts, show_progress_bar=True, batch_size=self.config.batch_size
        )
        if self.config.normalization:
            embs1 = l2_normalize(embs1)
            embs2 = l2_normalize(embs2)

        result = pd.DataFrame(
            {
                self.config.src_column: src_texts,
                self.config.tgt_column: tgt_texts,
                self.config.result_column: (embs1 * embs2).sum(-1),
            }
        )
        return result

    def validate(
        self,
        output: tp.Any,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> bool:
        return output.exists()


def l2_normalize(data: np.ndarray, epsilon=1e-20) -> np.ndarray:
    return data / np.maximum((data**2).sum(-1, keepdims=True) ** 0.5, epsilon)
