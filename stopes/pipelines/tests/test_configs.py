# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import typing as tp
from pathlib import Path

import hydra
import omegaconf
import pytest

import stopes.core
import stopes.modules.bitext.indexing.train_faiss_index_module
import stopes.modules.evaluation.generate_multi_bleu_detok_module
import stopes.modules.nmt_bitext_eval_utils
import stopes.modules.preprocess
import stopes.modules.train_fairseq_module
from stopes.modules.preprocess import TrainSpmConfig

STOPES = Path(__file__).parents[2]
CONF = STOPES / "pipelines" / "bitext" / "conf"

VALIDATED_CONFS: tp.Set[Path] = set()


def load_conf(file: Path, overrides: tp.Tuple[str, ...] = ()):
    assert file.exists(), f"{file} doesn't exist"
    file = file.relative_to(STOPES)
    conf_parent_dir, conf_name = str(file).split("/conf/", 1)
    conf_name = conf_name.replace(".yaml", "")

    with hydra.initialize(
        version_base=None,
        # hydra is using path relative to the Python file calling init.
        config_path="../../" + conf_parent_dir + "/conf",
        job_name=f"test_configs_{conf_name}",
    ):
        cfg = hydra.compose(
            config_name=conf_name,
            overrides=("hydra/job_logging=none",) + overrides,
        )

    # TODO: do we always want to un-nest the config ?
    parts = conf_name.split("/")
    for part in parts[:-1]:
        cfg = getattr(cfg, part)
    return cfg


def validate_prepare_data_configs(conf_dir: Path):
    from stopes.pipelines.prepare_data.configs import (
        DedupConfig,
        PrepareDataConfig,
        PreprocessingConfig,
        ShardingConfig,
        VocabConfig,
        VocabParams,
    )

    validate_conf(conf_dir / "prepare_data.yaml", PrepareDataConfig)
    validate_conf(conf_dir / "dedup" / "both.yaml", DedupConfig)
    validate_conf(conf_dir / "dedup" / "neither.yaml", DedupConfig)
    validate_conf(conf_dir / "preprocessing" / "default.yaml", PreprocessingConfig)
    validate_conf(conf_dir / "sharding" / "default.yaml", ShardingConfig)
    validate_conf(conf_dir / "vocab" / "default.yaml", VocabConfig)
    validate_conf(conf_dir / "vocab" / "src_vocab" / "default.yaml", VocabParams)
    validate_conf(conf_dir / "vocab" / "tgt_vocab" / "default.yaml", VocabParams)


def validate_conf(conf_file: Path, conf_cls: type, *overrides: str):
    """Validates a config file against a dataclass type."""
    cfg = load_conf(conf_file, overrides)
    cfg = stopes.core.utils.promote_config(cfg, conf_cls)
    omegaconf.OmegaConf.resolve(cfg)
    assert "$" not in str(cfg)
    VALIDATED_CONFS.add(conf_file)


def validate_nested_conf(conf_file: Path, conf_cls: type, *overrides: str):
    """Validates a 'nested' config file against a dataclass type.

    Nested config files are yaml file with a top level config.
    """
    cfg = load_conf(conf_file, overrides).config
    cfg = stopes.core.utils.promote_config(cfg, conf_cls)
    omegaconf.OmegaConf.resolve(cfg)
    assert "$" not in str(cfg)
    VALIDATED_CONFS.add(conf_file)


def instantiate_conf(conf_file: Path, *overrides: str):
    """Instantiates a class from a config file.

    This is typically harder because most modules will check
    whether some files specified in the config actually exist on disk.
    """
    cfg = load_conf(conf_file, overrides)
    module = stopes.core.StopesModule.build(cfg)
    assert "$" not in str(module.config)
    assert omegaconf.OmegaConf.missing_keys(module.config) == set()
    assert "???" not in str(module.config)
    module.requirements()
    VALIDATED_CONFS.add(conf_file)
    return module


def test_configs(tmp_path):
    validate_nested_conf(
        CONF / "train_index" / "train_faiss.yaml",
        stopes.modules.bitext.indexing.train_faiss_index_module.TrainFAISSIndexConfig,
    )
    validate_conf(
        CONF / "binarize" / "standard_conf.yaml",
        stopes.modules.preprocess.line_processor.LineProcessorConfig,
    )
    validate_conf(
        CONF / "moses" / "standard_conf.yaml",
        stopes.modules.preprocess.moses_cli_module.MosesPreprocessConfig,
    )
    validate_conf(
        CONF / "moses_filter" / "standard_conf.yaml",
        stopes.modules.nmt_bitext_eval_utils.MosesFilterConfig,
    )
    validate_conf(
        STOPES / "pipelines" / "speech" / "conf" / "compute_laser_embeddings.yaml",
        stopes.modules.preprocess.LaserEmbeddingConfig,
    )
    validate_prepare_data_configs(STOPES / "pipelines" / "prepare_data" / "conf")

    from stopes.pipelines.translate.translation_pipeline import TranslationConfig

    validate_conf(
        STOPES / "pipelines" / "translate" / "conf" / "example.yaml",
        TranslationConfig,
    )

    instantiate_conf(
        CONF / "train_spm" / "standard_conf.yaml",
        f"train_spm.output_dir={tmp_path}",
        f"train_spm.train_data_file={tmp_path}/eng.txt",
    )
    instantiate_conf(
        CONF / "train_fairseq" / "nmt.yaml",
        f"train_fairseq.params.task.data={tmp_path}",
        f"train_fairseq.params.task.source_lang=eng",
        f"train_fairseq.params.task.target_lang=fra",
        f"train_fairseq.output_dir={tmp_path}",
    )
    validate_conf(
        CONF / "eval" / "generate_multi_bleu_detok.yaml",
        stopes.modules.evaluation.generate_multi_bleu_detok_module.GenerateMultiBleuDetokConfig,
    )

    (tmp_path / "eng.dev").write_text("")
    (tmp_path / "checkpoint.pt").write_bytes(b"")
    with pytest.warns(DeprecationWarning):
        # TODO: simplify generate config
        instantiate_conf(
            CONF / "generate/standard_conf.yaml",
            f"generate.config.model={tmp_path}/checkpoint.pt",
            f"generate.config.src_text_file={tmp_path}/eng.dev",
            "generate.config.src_lang=eng",
            "generate.config.tgt_lang=ast",
            f"generate.config.output_dir={tmp_path}",
            "generate.config.beam_search.beam=1",
            f"+generate.config.arg_overrides.data={tmp_path}",
        )

    # Look for all configs, skip the one in 'tests' folder,
    # since they are probably already tested with some other test.
    all_configs = normalize_path(
        f
        for f in STOPES.glob("**/conf/**/*.yaml")
        if (
            ("_test.yaml" not in f.name)
            and ("tests" not in f.parts)
            and ("fb_reqs" not in f.parts)
            and not any([p.startswith("fb_") for p in f.parts])
        )
    )
    validated_configs = normalize_path(VALIDATED_CONFS)
    unvalidated = all_configs - validated_configs
    # If you're seeing this error it means you either:
    # * added a new config file, please add a test for it.
    # * added a new config test, please decrement the counter
    assert len(unvalidated) == 53


def normalize_path(files: tp.Iterable[Path]) -> tp.Set[Path]:
    return {f.resolve().relative_to(STOPES) for f in files}


def test_promote_adds_default_values():
    partial = omegaconf.DictConfig(
        {"output_dir": "/tmp", "train_data_file": "/tmp/input.txt", "vocab_size": 10}
    )
    default = TrainSpmConfig()
    promoted = stopes.core.utils.promote_config(partial, TrainSpmConfig)

    for key in dir(default):
        if key.startswith("__"):
            continue
        if key not in dir(partial):
            assert getattr(promoted, key) == getattr(default, key)
        else:
            assert getattr(promoted, key) == getattr(partial, key)

    assert promoted.vocab_size == 10
    assert promoted.vocab_size != default.vocab_size
    assert promoted.training_lines == default.training_lines


def test_promote_detects_bad_config():
    partial = omegaconf.DictConfig({"wrong_key": "/tmp", "vocab_size": 10})
    with pytest.raises(omegaconf.errors.ConfigKeyError):
        stopes.core.utils.promote_config(partial, TrainSpmConfig)


@dataclasses.dataclass
class MyModuleConf:
    field: str


class MyModule(stopes.core.StopesModule):
    def run(self):
        pass

    def requirements(self):
        pass


class MyTypedModule(MyModule):
    def __init__(self, config: MyModuleConf):
        super().__init__(config, MyModuleConf)


@pytest.mark.parametrize("typed", [False, True])
def test_flat_conf(typed: bool):
    with hydra.initialize(version_base=None, config_path="conf"):
        nested_cfg = hydra.compose(config_name="nested")

    with hydra.initialize(version_base=None, config_path="conf"):
        flat_cfg = hydra.compose(config_name="flat")

    assert nested_cfg.field == "hello"
    assert flat_cfg.field == "hello"
    assert nested_cfg.my_module.config.field == "hello_world"
    assert flat_cfg.my_module.field == "hello_world"

    if typed:
        typed_module = "stopes.pipelines.tests.test_configs.MyTypedModule"
        nested_cfg.my_module._target_ = typed_module
        flat_cfg.my_module._target_ = typed_module

    with pytest.warns(DeprecationWarning, match="Nested configs"):
        nested = stopes.core.StopesModule.build(nested_cfg.my_module)
    flat = stopes.core.StopesModule.build(flat_cfg.my_module)

    if typed:
        assert type(nested) is MyTypedModule
        assert type(flat) is MyTypedModule
    else:
        assert type(nested) is MyModule
        assert type(flat) is MyModule

    assert nested.config.field == "hello_world"
    assert flat.config.field == "hello_world"
