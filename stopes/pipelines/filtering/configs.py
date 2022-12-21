#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from hydra.core.config_store import ConfigStore


@dataclass
class ExecutorConfig:
    log_folder: str = "executor_logs"
    cluster: str = "local"
    slurm_partition: Optional[str] = None


@dataclass
class LaserFilterConfig:
    _target_: str = "stopes.pipelines.filtering.filters.LaserFilter"
    threshold: float = 1.06


@dataclass
class LengthFilterConfig:
    _target_: str = "stopes.pipelines.filtering.filters.LengthFilter"
    min_len: Optional[int] = 1
    max_len: Optional[int] = 1050
    max_len_ratio: Optional[float] = 9.0
    min_src_unique_ratio: Optional[float] = None


@dataclass
class LidFilterConfig:
    _target_: str = "stopes.pipelines.filtering.filters.LidFilter"
    model_path: str = "/large_experiments/seamless/nllb/mmt/lidruns/lid_models/2022-02-18_ft_model.bin"
    excluded_corpora: Optional[List[str]] = None
    excluded_languages: Optional[List[str]] = None
    default_threshold: float = 0.0
    thresholds: Dict[str, float] = field(default_factory=dict)


@dataclass
class ToxicityFilterConfig:
    _target_: str = "stopes.pipelines.filtering.filters.ToxicityFilter"
    twl_path_template: str = (
        "/large_experiments/seamless/nllb/mmt/data/toxicity/{lang}_twl.txt"
    )
    eng_porn_twl_path: Optional[
        str
    ] = "/large_experiments/seamless/nllb/mmt/data/toxicity/eng_twl_short_porn.txt"
    max_toxicity: Optional[int] = None
    max_toxicity_difference: Optional[int] = 2


@dataclass
class DedupFilterConfig:
    _target_: str = "stopes.pipelines.filtering.filters.DedupFilter"
    dedup_pairs: bool = False
    max_source_dedup: Optional[int] = None
    max_target_dedup: Optional[int] = None


@dataclass
class GroupFilterConfig:
    # one (and only one) of these should be set, the other should be None
    included_corpora: Optional[List[str]] = None
    excluded_corpora: Optional[List[str]] = None

    normalize_punctuation: bool = True
    normalize_unicode: bool = False

    laser_filter: Optional[LaserFilterConfig] = None
    length_filter: LengthFilterConfig = LengthFilterConfig()
    lid_filter: Optional[LidFilterConfig] = None
    toxicity_filter: Optional[ToxicityFilterConfig] = None
    dedup_filter: DedupFilterConfig = DedupFilterConfig()


@dataclass
class FilterConfig:
    data_conf_dir: str
    output_dir: str
    executor: ExecutorConfig
    directions: List[str]
    train_primary: Optional[GroupFilterConfig]
    train_mined: Optional[GroupFilterConfig]
    train_bt: Optional[GroupFilterConfig]


def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=FilterConfig)

    # Primary
    cs.store(name="default", group="train_primary/laser_filter", node=LaserFilterConfig)
    cs.store(
        name="default", group="train_primary/length_filter", node=LengthFilterConfig
    )
    cs.store(
        name="default", group="train_primary/toxicity_filter", node=ToxicityFilterConfig
    )
    cs.store(name="default", group="train_primary/lid_filter", node=LidFilterConfig)
    cs.store(name="default", group="train_primary/dedup_filter", node=DedupFilterConfig)

    # Mined
    cs.store(name="default", group="train_mined/laser_filter", node=LaserFilterConfig)
    cs.store(name="default", group="train_mined/length_filter", node=LengthFilterConfig)
    cs.store(
        name="default", group="train_mined/toxicity_filter", node=ToxicityFilterConfig
    )
    cs.store(name="default", group="train_mined/lid_filter", node=LidFilterConfig)
    cs.store(name="default", group="train_mined/dedup_filter", node=DedupFilterConfig)

    # Backtranslation
    cs.store(name="default", group="train_bt/laser_filter", node=LaserFilterConfig)
    cs.store(name="default", group="train_bt/length_filter", node=LengthFilterConfig)
    cs.store(
        name="default", group="train_bt/toxicity_filter", node=ToxicityFilterConfig
    )
    cs.store(name="default", group="train_bt/lid_filter", node=LidFilterConfig)
    cs.store(name="default", group="train_bt/dedup_filter", node=DedupFilterConfig)
