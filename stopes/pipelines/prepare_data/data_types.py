# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Dict, List, Optional

from omegaconf import MISSING


@dataclass
class ParallelDataset:
    """A configuration class of parallel dataset paths in translation

    Features:
        source (str): file path of source language corpora
        target (Str): file path of target language corpora
        data_tag (Optional[str]): optional string to tag source sentences from specific corpora
    """

    source: str = MISSING
    target: str = MISSING
    num_lines: Optional[int] = None
    is_gzip: Optional[bool] = False
    data_tag: Optional[str] = None


@dataclass
class CorporaMap:
    """A configuration class of corpora sources & corresponding corpora paths in one direction

    Features:
        values (Dict[str, ParallelDataset]): dictionary mapping from translation direction to corresponding corpora
    """

    values: Dict[str, ParallelDataset] = MISSING


@dataclass
class BuiltVocab:
    model_file: str = MISSING
    vocab_file: str = MISSING
    dict_file: Optional[str] = None


@dataclass
class VocabBuildParams:
    """A configuration class of parameters of training spm model

    Features:
        vocab_size (int): vocabulary size
        used_joned_data (bool): whether to use joined source & target data for training or train separately
        model_type (str): segmentation algorithms ('bep', 'unigram')
        sampled_data_size (int): maximum number of sentences for training
        sampling_temperature (float): temperature parameter of temperature sampling
        character_coverage (int): amount of characters covered by the model
        shuffle_input_sentence (bool): random shuffle or take the first <SIZE> lines
    """

    vocab_size: int = MISSING
    use_joined_data: bool = True
    model_type: str = "bpe"
    sampled_data_size: int = 10000000
    sampling_temperature: float = 1.0
    character_coverage: float = 0.99995
    shuffle_input_sentence: bool = True
    random_seed: int = 0


@dataclass
class VocabConfig:
    pretrained: Optional[BuiltVocab] = None
    vocab_build_params: Optional[VocabBuildParams] = VocabBuildParams()


@dataclass
class BinarizationConfig:
    """A configuration class of binarization

    Features:
        max_examples_per_shard (int): maximum number of sentences (from all languages) per shard
        smallest_shard (int): minimum number of sentences for each language in each shard
        binarize_workers (int): number of workers to use in fairseq-preprocess
    """

    max_examples_per_shard: int = MISSING
    smallest_shard: int = 250000
    binarize_workers: int = 60
    random_seed: int = 0


@dataclass
class MosesConfig:
    """A configuration class of Moses preprocessing

    Features:
        script_directory (str): location of moses scripts
        lowercase (bool): whether to lowercase
        normalize_punctuation (bool): whether to normalize punctuation
        remove_non_printing_chars (bool): whether to remove non-printing chars
        deescape_special_chars (bool): whether to deescape special chars
    """

    script_directory: str = MISSING
    lowercase: bool = False
    normalize_punctuation: bool = True
    remove_non_printing_chars: bool = False
    deescape_special_chars: bool = False


@dataclass
class PreprocessingConfig:
    """A configuation class of preprocessing

    Features:
        sample_size (int): how many lines to sample (default: all)
        max_tokens (int): filter out lines containing greater than max tokens (default: none)
        moses_config (MosesConfig): moses-specific configuration
        preprocess_source (bool): whether to apply preprocessing to the source
        preprocess_target (bool): whether to apply preprocessing to the target
        tag_data (bool): whether to prepend special tags to source sentences in some corpora
    """

    sample_size: Optional[int] = None
    max_tokens: Optional[int] = None
    moses_config: Optional[MosesConfig] = MosesConfig()
    preprocess_source: Optional[bool] = True
    preprocess_target: Optional[bool] = True
    tag_data: bool = True


@dataclass
class ExecutorConfig:
    log_folder: str = "executor_logs"
    cluster: str = "local"
    slurm_partition: Optional[str] = None


@dataclass
class DataConfig:
    """A configuration class of data

    Features:
        train_corpora (Dict[str, CorporaMap): training corpora configurations for clean bitext data
        train_mining_corpora (Dict[str, CorporaMap): training corpora configurations for mining data
        train_smt_bt_corpora (Dict[str, CorporaMap): training corpora configurations for smt BT data
        train_mmt_bt_corpora (Dict[str, CorporaMap): training corpora configurations for mmt BT data
        valid_corpora (Dict[str, CorporaMap): validation corpora configurations
        test_corpora (Dict[str, CorporaMap): testing corpora configurations
        vocab_config (VocabConfig): vocabulary configurations
        binarization_config (BinarizationConfig): binarization configurations
    """

    train_corpora: Dict[str, CorporaMap] = MISSING
    train_mining_corpora: Optional[Dict[str, CorporaMap]] = None
    train_smt_bt_corpora: Optional[Dict[str, CorporaMap]] = None
    train_mmt_bt_corpora: Optional[Dict[str, CorporaMap]] = None
    valid_corpora: Optional[Dict[str, CorporaMap]] = None
    test_corpora: Optional[Dict[str, CorporaMap]] = None
    source_vocab_config: VocabConfig = VocabConfig()
    target_vocab_config: VocabConfig = VocabConfig()
    binarization_config: BinarizationConfig = BinarizationConfig()
    executor_config: ExecutorConfig = ExecutorConfig()
    preprocessing_config: PreprocessingConfig = PreprocessingConfig()
