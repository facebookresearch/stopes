# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import builtins
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
import stopes.modules.speech.audio_zip
import stopes.modules.speech.denoise
import stopes.modules.speech.segment_dataset
import stopes.modules.speech.shas_segment_audio
import stopes.modules.speech.speech_units
import stopes.modules.speech.speechbrain_lid
import stopes.modules.speech.vad_segment_audio
import stopes.modules.speech.vad_trim_audio
import stopes.modules.speech.whisper
import stopes.modules.train_fairseq_module
from stopes.eval.local_prosody.unity2_forced_aligner_f2 import (
    UnitY2F2ForcedAlignerConfig,
)
from stopes.eval.vocal_style_similarity.vocal_style_sim_module import (
    VocalStyleSimilarityConfig,
)
from stopes.modules.evaluation.compare_audio_module import CompareAudiosConfig
from stopes.modules.evaluation.sentence_transformers_similarity import (
    SentenceTransformersSimilarityConfig,
)
from stopes.modules.preprocess import TrainSpmConfig, wav2vec_laser_speech_encoder
from stopes.pipelines.bitext.added_toxicity_mining import (
    AddedToxicityMiningPipelineConfig,
)
from stopes.pipelines.m4t_eval.config_def import M4TEvalConfig
from stopes.pipelines.speech.whisper_asr import WhisperPipelineConfig

STOPES = Path(__file__).parents[2]
CONF = STOPES / "pipelines" / "bitext" / "conf"


@pytest.fixture()
def openned_files(monkeypatch: pytest.MonkeyPatch) -> tp.Iterator[tp.Set[Path]]:
    """Exposes all files openned since the beginning of the current unit test."""
    openned = set()
    builtins_open = builtins.open

    def open(file: str, mode: str = "r", **kwargs: tp.Any) -> tp.IO[tp.Any]:
        if "r" in mode:
            openned.add(Path(file))
        return builtins_open(file, mode=mode, **kwargs)

    monkeypatch.setattr(builtins, "open", open)
    yield openned


def load_conf(
    file: Path, overrides: tp.Tuple[str, ...] = (), unnest: bool = True
) -> tp.Any:
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
            # hydra should accept an Iterable[str], not just List[str]
            overrides=("hydra/job_logging=none",) + overrides,  # type: ignore[arg-type]
        )

    if not unnest:
        return cfg

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


def validate_nested_conf(conf_file: Path, conf_cls: type, *overrides: str):
    """Validates a 'nested' config file against a dataclass type.

    Nested config files are yaml file with a top level config.
    """
    cfg = load_conf(conf_file, overrides).config
    cfg = stopes.core.utils.promote_config(cfg, conf_cls)
    omegaconf.OmegaConf.resolve(cfg)
    assert "$" not in str(cfg)


def validate_line_processor_conf(conf_file: Path, conf_cls: type, *overrides: str):
    """Validates a config file against a dataclass type."""
    cfg = load_conf(conf_file, overrides)
    cfg = stopes.core.utils.promote_config(cfg.line_processor.config, conf_cls)
    omegaconf.OmegaConf.resolve(cfg)
    assert "$" not in str(cfg)


def instantiate_module(conf_file: Path, *overrides: str) -> stopes.core.StopesModule:
    """Instantiates a stopes Module from a config file.

    This imitates the `python -m stopes.module` CLI behavior.

    This is typically harder because some modules will check
    whether some files specified in the config actually exist on disk.
    """
    assert CONF.parts == conf_file.parts[:-2], (
        "For now instantiate_module only works with config file inside "
        f"'pipelines/bitext/conf'. Received {conf_file}"
    )
    key, conf_name = conf_file.parts[-2:]
    launch_conf = CONF / "launch_conf.yaml"
    cfg = load_conf(launch_conf, (f"+{key}={conf_name}",) + overrides, unnest=False)
    cfg = getattr(cfg, key)

    module = stopes.core.StopesModule.build(cfg)
    assert "$" not in str(module.config)
    assert omegaconf.OmegaConf.missing_keys(module.config) == set()
    assert "???" not in str(module.config)
    module.requirements()
    return module


def test_configs(tmp_path: Path, openned_files: tp.Set[Path]) -> None:
    validate_embed_text_configs()

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
    validate_line_processor_conf(
        CONF / "speech_preproc" / "speechbrain_lid.yaml",
        stopes.modules.speech.speechbrain_lid.SpeechbrainLIDConfig,
    )
    validate_conf(
        CONF / "speech_preproc" / "vad.yaml",
        stopes.modules.speech.vad_segment_audio.VADSegmentAudioConfig,
    )
    validate_conf(
        CONF / "speech_preproc" / "vad_trim.yaml",
        stopes.modules.speech.vad_trim_audio.VADTrimAudioConfig,
    )
    validate_conf(
        CONF / "speech_preproc" / "shas.yaml",
        stopes.modules.speech.shas_segment_audio.SHASSegmentAudioConfig,
    )
    validate_conf(
        CONF / "speech_units" / "base.yaml",
        stopes.modules.speech.speech_units.SpeechUnitsConfig,
    )
    validate_conf(
        CONF / "speech_transcription" / "whisper.yaml",
        stopes.modules.speech.whisper.WhisperConfig,
    )
    from stopes.pipelines.speech.wav2vec_asr import ASRPipelineConfig

    validate_conf(
        STOPES / "pipelines" / "speech" / "conf" / "wav2vec_asr.yaml", ASRPipelineConfig
    )
    validate_conf(
        CONF / "speech_preproc" / "segment_dataset.yaml",
        stopes.modules.speech.segment_dataset.SegmentDATASETConfig,
    )
    validate_conf(
        STOPES
        / "pipelines"
        / "speech_laser_embeddings"
        / "conf"
        / "speech_laser_embeddings.yaml",
        wav2vec_laser_speech_encoder.LaserEmbeddingConfig,
    )
    from stopes.pipelines.speech_kmeans.speech_kmeans import SpeechKMeansMainConfig

    validate_conf(
        STOPES / "pipelines" / "speech_kmeans" / "conf" / "speech_kmeans.yaml",
        SpeechKMeansMainConfig,
    )
    validate_prepare_data_configs(STOPES / "pipelines" / "prepare_data" / "conf")
    validate_embed_text_configs()

    from stopes.pipelines.translate.translation_pipeline import TranslationConfig

    validate_conf(
        STOPES / "pipelines" / "translate" / "conf" / "example.yaml",
        TranslationConfig,
    )
    validate_conf(
        STOPES / "pipelines" / "m4t_eval" / "conf" / "m4t_eval.yaml",
        M4TEvalConfig,
    )

    instantiate_module(
        CONF / "train_spm" / "standard_conf.yaml",
        f"train_spm.output_dir={tmp_path}",
        f"train_spm.train_data_file={tmp_path}/eng.txt",
    )
    instantiate_module(
        CONF / "train_fairseq" / "nmt.yaml",
        f"train_fairseq.params.task.data={tmp_path}",
        "train_fairseq.params.task.source_lang=eng",
        "train_fairseq.params.task.target_lang=fra",
        f"train_fairseq.output_dir={tmp_path}",
    )
    validate_conf(
        CONF / "eval" / "generate_multi_bleu_detok.yaml",
        stopes.modules.evaluation.generate_multi_bleu_detok_module.GenerateMultiBleuDetokConfig,
    )
    validate_conf(
        STOPES / "pipelines" / "bitext" / "conf" / "denoise" / "denoise.yaml",
        stopes.modules.speech.denoise.DenoiserConfig,
    )

    validate_conf(
        STOPES / "pipelines" / "bitext" / "conf" / "audio_zip" / "base.yaml",
        stopes.modules.speech.audio_zip.AudioZipConfig,
    )
    validate_conf(
        STOPES / "pipelines" / "bitext" / "conf" / "compare_audios" / "base.yaml",
        CompareAudiosConfig,
    )
    validate_conf(
        STOPES
        / "pipelines"
        / "bitext"
        / "conf"
        / "compare_audios"
        / "AutoPCP_multilingual_v2.yaml",
        CompareAudiosConfig,
    )
    validate_conf(
        STOPES / "pipelines" / "speech" / "conf" / "uromanization.yaml",
        stopes.modules.preprocess.uromanize_cli_module.UromanPreprocessConfig,
    )
    validate_conf(
        STOPES / "pipelines" / "speech" / "conf" / "whisper_asr.yaml",
        WhisperPipelineConfig,
    )
    validate_conf(
        STOPES / "pipelines" / "bitext" / "conf" / "mine_added_toxicity.yaml",
        AddedToxicityMiningPipelineConfig,
    )
    validate_nested_conf(
        STOPES
        / "pipelines"
        / "bitext"
        / "conf"
        / "forced_alignment"
        / "fairseq2_nar_t2u_aligner.yaml",
        UnitY2F2ForcedAlignerConfig,
    )
    validate_conf(
        STOPES
        / "pipelines"
        / "bitext"
        / "conf"
        / "vocal_style_similarity"
        / "base.yaml",
        VocalStyleSimilarityConfig,
    )
    validate_conf(
        STOPES
        / "pipelines"
        / "bitext"
        / "conf"
        / "sentence_transformers_similarity"
        / "base.yaml",
        SentenceTransformersSimilarityConfig,
    )

    for hub_conf in (CONF / "speech_tokenizer").glob("*.yaml"):
        hub_conf_str = str(hub_conf.resolve())
        if "encodec" in hub_conf_str:
            from stopes.speech.encodec import EncodecConfig

            validate_conf(
                hub_conf,
                EncodecConfig,
                "speech_tokenizer.target_sample_rate=50",
                "speech_tokenizer.quantizer_bins=1024",
            )
        else:
            from stopes.speech.tokenizers import SpeechTokenizerConfig

            validate_conf(
                hub_conf,
                SpeechTokenizerConfig,
                "speech_tokenizer.lang=en",
                "speech_tokenizer.feature_layer=35",
            )

    (tmp_path / "eng.dev").write_text("")
    (tmp_path / "checkpoint.pt").write_bytes(b"")
    with pytest.warns(DeprecationWarning):
        # TODO: simplify generate config
        instantiate_module(
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
    all_configs = normalize_paths(
        f
        for f in STOPES.glob("**/conf/**/*.yaml")
        if (
            ("_test.yaml" not in f.name)
            and ("tests" not in f.parts)
            and ("fb_reqs" not in f.parts)
            and not any([p.startswith("fb_") for p in f.parts])
        )
    )

    validated_configs = normalize_paths(f for f in openned_files if f.suffix == ".yaml")
    unvalidated = all_configs - validated_configs
    print(f"Found following {len(unvalidated)} unvalidated config files:")
    for file in sorted(unvalidated):
        print(file)
    # If you're seeing this error it means you either:
    # * added a new config file, please add a test for it.
    # * added a new config test, please decrement the counter
    assert len(unvalidated) == 46


def normalize_paths(files: tp.Iterable[Path]) -> tp.Set[Path]:
    return {f.resolve().relative_to(STOPES) for f in files}


def validate_embed_text_configs() -> None:
    EMBED = CONF / "embed_text"
    # Do we need all those equivalent way of launching text embedding pipeline ?
    _embed_text(EMBED / "laser2.yaml", "+model_dir=/laser2/", "+vocab_dir=/laser2/")
    _embed_text(
        EMBED / "encode.yaml",
        "+model_dir=/laser2/",
        "+vocab_dir=/laser2/",
        "+embed_text/encoder=laser2_encoder",
    )

    _embed_text(EMBED / "laser3.yaml")
    _embed_text(EMBED / "encode.yaml", "+embed_text/encoder=laser3_encoder")
    _embed_text(EMBED / "preproc_and_encode.yaml", "+embed_text/encoder=laser3_encoder")

    _embed_text(EMBED / "hf_labse.yaml")
    _embed_text(EMBED / "hf_roberta_large.yaml")
    _embed_text(EMBED / "huggingface.yaml", "+name=LaBSE")
    _embed_text(EMBED / "huggingface.yaml", "+name=all-roberta-large-v1")
    _embed_text(
        EMBED / "encode.yaml",
        "+embed_text/encoder=hf_encoder",
        "embed_text.encoder._name=LaBSE",
    )
    _embed_text(EMBED / "sonar.yaml", "+name=text_sonar_basic_encoder")
    _embed_text(
        EMBED / "encode.yaml",
        "+embed_text/encoder=sonar_encoder",
        "embed_text.encoder._name=text_sonar_basic_encoder",
    )


def _embed_text(main_conf: Path, *overrides: str) -> None:
    assert main_conf.exists(), f"Invalid test: {main_conf} doesn't exist"
    instantiate_module(
        main_conf,
        "embed_text.lang=ukr",
        "embed_text.shards=[/tmp/embed_text/input.tsv]",
        "embed_text.output_dir=/tmp/embed_text/",
        *overrides,
    )


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
