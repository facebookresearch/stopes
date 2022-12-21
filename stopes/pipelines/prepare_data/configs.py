# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from dataclasses import dataclass, field
from enum import Enum

from omegaconf import MISSING

from stopes.modules.preprocess.train_spm import TrainSpmConfig, Vocab
from stopes.pipelines.filtering.dataset import Dataset


@dataclass
class CorporaConfig:
    train: tp.Dict[str, tp.Dict[str, Dataset]] = MISSING
    train_mining: tp.Dict[str, tp.Dict[str, Dataset]] = field(default_factory=dict)
    train_mmt_bt: tp.Dict[str, tp.Dict[str, Dataset]] = field(default_factory=dict)
    train_smt_bt: tp.Dict[str, tp.Dict[str, Dataset]] = field(default_factory=dict)
    valid: tp.Dict[str, tp.Dict[str, Dataset]] = field(default_factory=dict)
    test: tp.Dict[str, tp.Dict[str, Dataset]] = field(default_factory=dict)


@dataclass
class PreprocessingConfig:
    lowercase: bool
    normalize_punctuation: bool
    remove_non_printing_chars: bool
    deescape_special_chars: bool


@dataclass
class SamplingConfig:
    sampled_data_size: int
    sampling_temperature: float


@dataclass
class VocabParams:
    pretrained: tp.Optional[Vocab]
    sampling_config: tp.Optional[SamplingConfig]
    spm_config: tp.Optional[TrainSpmConfig]


@dataclass
class VocabConfig:
    src_vocab: VocabParams
    tgt_vocab: VocabParams
    use_joined_data: bool


class DedupType(Enum):
    src = "src"
    tgt = "tgt"
    both = "both"
    neither = "neither"


@dataclass
class DedupConfig:
    """
    dedup_type (DedupType): Dedup by src sentence or tgt sentence or both or neither.
        If neither, we only dedup by the sentence-pair.
    cross_fold (bool): Dedup across different folds together or not.
    """

    dedup_type: DedupType = DedupType.both
    cross_fold: bool = False


@dataclass
class ShardingConfig:
    """
    max_examples_per_shard (int): maximum number of sentences (from all languages) per shard
    smallest_shard (int): minimum number of sentences for each language per shard
    binarize_num_workers (int): number of workers to Multiproc binarize files
    """

    max_examples_per_shard: int = MISSING
    smallest_shard: int = 250000
    binarize_num_workers: int = 30


@dataclass
class PrepareDataConfig:
    corpora: CorporaConfig
    preprocessing: PreprocessingConfig
    vocab: VocabConfig
    dedup: DedupConfig
    sharding: ShardingConfig
    launcher: tp.Dict[str, tp.Any]
    output_dir: str = MISSING
