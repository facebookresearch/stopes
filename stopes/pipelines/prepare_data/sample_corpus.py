# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stopes.core import utils
from stopes.core.launcher import Launcher
from stopes.core.stopes_module import Requirements, StopesModule
from stopes.pipelines.prepare_data.configs import SamplingConfig


@dataclass
class SampleCorpusConfig:
    sampling_config: SamplingConfig
    lang_files: tp.Dict[str, tp.List[str]]
    lang_counts_map: tp.Dict[str, int]
    corpus_file: Path


class SampleCorpus(StopesModule):
    def __init__(
        self,
        config: SampleCorpusConfig,
    ):
        super().__init__(config, SampleCorpusConfig)
        if not sum(config.lang_counts_map.values()):
            raise ValueError("Empty training data")

    def requirements(self) -> Requirements:
        return Requirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=1,
            timeout_min=2 * 24 * 60,
        )

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> Path:
        logger = logging.getLogger("stopes.prepare_data.sample_corpus")

        lang_counts_map = self.config.lang_counts_map
        sum_count = sum(lang_counts_map.values())
        lang_probs_map_smoothed = {
            lang: (count / sum_count)
            ** (1 / self.config.sampling_config.sampling_temperature)
            for lang, count in lang_counts_map.items()
        }
        sum_prob_smoothed = sum(lang_probs_map_smoothed.values())

        total_size = min(self.config.sampling_config.sampled_data_size, sum_count)
        lang_counts_sampled_map = {
            lang: int(total_size * (prob / sum_prob_smoothed))
            for lang, prob in lang_probs_map_smoothed.items()
        }
        logger.info(f"samples count per lang: {lang_counts_sampled_map}")

        output_lang_counts = defaultdict(int)
        np.random.seed(0)
        with utils.open(self.config.corpus_file, "wb") as f_out:
            for lang, corpora in self.config.lang_files.items():
                total_count = lang_counts_map[lang]
                sample_count = lang_counts_sampled_map[lang]
                indices = set(
                    np.random.choice(total_count, sample_count, replace=False)
                )
                num_processed_lines = 0
                for corpus in corpora:
                    with utils.open(corpus, "rb") as in_f:
                        for line in in_f:
                            if (
                                sample_count >= total_count
                                or num_processed_lines in indices
                            ):
                                f_out.write(line)
                                output_lang_counts[lang] += 1
                            num_processed_lines += 1
        logger.info(f"sampled corpus with counts per lang: {output_lang_counts}")
        return self.config.corpus_file

    def validate(
        self,
        output: tp.Any,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> bool:
        sampled_corpus_file = output
        return sampled_corpus_file.exists()

    def version(self):
        return "0.0"


async def sample_corpus(
    sampling_config: SamplingConfig,
    lang_files: tp.Dict[str, tp.List[str]],
    lang_counts_map: tp.Dict[str, int],
    launcher: Launcher,
    corpus_file: Path,
):
    sample_corpus_module = SampleCorpus(
        SampleCorpusConfig(
            sampling_config=sampling_config,
            lang_files=lang_files,
            lang_counts_map=lang_counts_map,
            corpus_file=corpus_file,
        )
    )
    sampled_corpus_file = await launcher.schedule(sample_corpus_module)
    return sampled_corpus_file
