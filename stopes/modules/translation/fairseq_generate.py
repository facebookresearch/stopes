# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import base64
import collections
import dataclasses
import functools
import io
import json
import logging
import pickle
import subprocess
import typing as tp
from pathlib import Path

import fairseq.checkpoint_utils
import fairseq.sequence_generator
import sentencepiece
import torch
import tqdm
from omegaconf import DictConfig

import stopes.core
from stopes.core import utils
from stopes.utils.cache import GenerationalCache

log = logging.getLogger("stopes.fairseq_generate")


class Sample(tp.NamedTuple):
    sentence: str
    src: int
    tgt: int


SampleOrStr = tp.Union[Sample, str]


@dataclasses.dataclass
class BeamSearchConfig:
    beam: int = 5
    max_len_a: int = 0
    max_len_b: int = 200
    min_len: int = 1
    unnormalized: bool = False
    lenpen: float = 1.0
    unkpen: float = 0.0
    temperature: float = 1.0
    match_source_len: bool = False
    no_repeat_ngram_size: int = 0
    sampling: bool = False
    sampling_topk: int = -1
    sampling_topp: float = -1.0


@dataclasses.dataclass
class FairseqGenerateConfig:
    model: str
    src_text_file: str = ""
    src_lang: str = ""  # Should be same lang as src_text_file, otherwise use file_list for multiple src_files/src-tgt_langs
    tgt_lang: str = "eng"
    spm: str = ""
    output_dir: tp.Optional[Path] = None
    # Should be formatted as: List[(src_file, src_lang, tgt_lang)]
    file_list: tp.List[tp.Any] = dataclasses.field(default_factory=list)
    preserve_filenames: bool = False
    lang_code_mapping: tp.Dict = dataclasses.field(default_factory=dict)
    beam_search: BeamSearchConfig = BeamSearchConfig()
    use_gpu: bool = True
    fp16: bool = True
    batch_size: int = 32
    torchscript: bool = False
    encoder_langtok: str = "src"
    arg_overrides: tp.Dict = dataclasses.field(default_factory=dict)
    cache_size: int = 1_000_000
    max_sentence_len: int = 256


class FairseqGenerateModule(stopes.core.StopesModule):
    def __init__(self, config: FairseqGenerateConfig, output_dir: Path = None):
        super().__init__(config, FairseqGenerateConfig)
        assert Path(config.model).exists(), f"Fairseq model {config.model} not found"
        assert Path(
            self.config.spm
        ).exists(), f"SentencePiece model {config.spm} not found"
        if self.config.file_list:
            for file in self.config.file_list:
                assert Path(
                    file[0]
                ).exists(), f"Source file {file} from 'file_list' not found"
        else:
            assert Path(
                self.config.src_text_file
            ).exists(), f"Source file {self.config.src_text_file} not found"
            assert (
                self.config.src_lang
            ), f"Unspecified input language for file {self.config.src_text_file}"
        self.output_dir = Path(
            output_dir
            or self.config.output_dir
            or Path(self.config.src_text_file).parent
        )
        self.output_dir.mkdir(exist_ok=True)

    def name(self) -> str:
        if self.config.file_list:
            return "fairseq_generate"
        else:
            return f"fairseq_generate_{self.config.src_lang}-{self.config.tgt_lang}"

    def array(self) -> tp.Optional[tp.List[tp.Any]]:
        if self.config.file_list:
            return self.config.file_list

        return [(self.config.src_text_file, self.config.src_lang, self.config.tgt_lang)]

    def requirements(self) -> stopes.core.Requirements:
        return stopes.core.Requirements(
            nodes=1,
            mem_gb=20,
            tasks_per_node=1,
            gpus_per_node=1 if self.config.use_gpu else 0,
            cpus_per_task=4,
            timeout_min=60 * 10,
        )

    def run(
        self, iteration_value: tp.Optional[tp.Any] = None, iteration_index: int = 0
    ):
        assert self.output_dir is not None
        if iteration_value is not None:
            src_file, src_lang, tgt_lang = iteration_value
        else:
            src_file = self.config.src_text_file
            src_lang = self.config.src_lang
            tgt_lang = self.config.tgt_lang
        if self.config.preserve_filenames:
            name = Path(src_file).name
            output_file = self.output_dir / f"{src_lang}-{tgt_lang}.{name}"
        else:
            output_file = self.output_dir / f"{src_lang}-{tgt_lang}.gen"
        translator = FairseqGenerator(self.config, src_lang=src_lang, tgt_lang=tgt_lang)
        with stopes.core.utils.open(output_file, mode="w") as o:
            with stopes.core.utils.open(src_file) as f:
                for translation in translator.translate(f.readlines()):
                    print(translation, file=o)


class FairseqGenerator:
    """
    Contains all the state for running the generation: model weights, spm, cache, ...
    This part doesn't depend on stopes.
    """

    def __init__(
        self, config: FairseqGenerateConfig, src_lang: str = None, tgt_lang: str = None
    ) -> None:
        """
        Load Fairseq model and SentencePiece model.
        """
        self.config = config
        self.cache: GenerationalCache[SampleOrStr, str] = GenerationalCache(
            max_entry=config.cache_size
        )
        (
            [model],
            model_cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [config.model], arg_overrides=config.arg_overrides
        )
        log.info(f"Loaded fairseq model from {config.model}")
        log.info(f"Model was trained on {model_cfg.task.data}")
        self.task = task
        tgt_lang = tgt_lang or config.tgt_lang
        self.vocab = resolve_task_vocab(task, tgt_lang)

        assert self.config.encoder_langtok in ("src", "tgt")
        if self.config.encoder_langtok == "src":
            encoder_lang = src_lang or config.src_lang
        else:
            encoder_lang = tgt_lang
        self.default_src_token = self.lang_token(encoder_lang) if encoder_lang else None
        self.default_tgt_token = self.lang_token(tgt_lang)

        self.spm = sentencepiece.SentencePieceProcessor()
        spm_path = config.spm or resolve_spm_path(model_cfg)
        self.spm.Load(spm_path)
        log.info(f"Loaded spm from {spm_path}")

        self.device = "cuda:0" if self.config.use_gpu else "cpu"
        self.model = model.eval().to(self.device)
        if config.use_gpu and config.fp16:
            model.half()
        if config.torchscript:
            model = torch.jit.script(model)

        if model_cfg.task.target_lang is None:
            model_cfg.task.target_lang = tgt_lang
        else:
            assert model_cfg.task.target_lang == tgt_lang, (
                "Model was trained with target lang "
                f"{model_cfg.task.target_lang} != {tgt_lang}"
            )

        self.sequence_generator = task.build_generator([self.model], config.beam_search)
        self._spm_vocab_shortcut: tp.Optional[bool] = None

    def lang_token(self, lang: str) -> int:
        if self.config.lang_code_mapping:
            lang = self.config.lang_code_mapping.get(lang, lang)

        token = self.vocab.index(f"__{lang}__")
        assert token != self.vocab.unk(), f"Unknown language '{lang}'"
        return token

    def tokenize(self, line: str) -> list:
        # For some models it seems that pieces and vocab dict actually match.
        if self._spm_vocab_shortcut is None:
            words = self.spm.EncodeAsPieces(line.strip())
            tokens = [self.vocab.index(word) for word in words]
            spm_tokens = self.spm.EncodeAsIds(line.strip())
            shortcut_works = tokens == [t + 1 for t in spm_tokens]
            if shortcut_works:
                log.warning(
                    f"It seems that vocab from {self.config.model} matches pieces from {self.config.spm}. Will use only SPM going forward."
                )
            else:
                log.warning(
                    f"It seems that vocab from {self.config.model} does NOT match pieces from {self.config.spm}. Will use SPM + vocab going forward."
                )
            self._spm_vocab_shortcut = shortcut_works

        elif self._spm_vocab_shortcut:
            spm_tokens = self.spm.EncodeAsIds(line.strip())
            tokens = [t + 1 for t in spm_tokens]
        else:
            words = self.spm.EncodeAsPieces(line.strip())
            tokens = [self.vocab.index(word) for word in words]

        return tokens

    def preprocess_one(self, sample: tp.Union[str, Sample]) -> dict:
        """
        preprocess data into a format that the model can do inference on
        """
        if isinstance(sample, str):
            line = sample
            src_tok = self.default_src_token
            tgt_tok = self.default_tgt_token
        else:
            line, src_tok, tgt_tok = sample.sentence, sample.src, sample.tgt
        tokens = self.tokenize(line.rstrip("\n"))
        # Prevent OOM.
        # TODO: we should try to translate the different parts and collate afterwards
        tokens = tokens[: self.config.max_sentence_len]

        return {
            "src_tokens": [src_tok] + tokens + [self.vocab.eos()],
            "src_length": len(tokens) + 1,
            "tgt_token": tgt_tok,
        }

    def preprocess(self, samples) -> dict:
        samples = [self.preprocess_one(s) for s in samples]
        prefix_tokens = torch.tensor([[s["tgt_token"]] for s in samples])
        src_lengths = torch.tensor([s["src_length"] for s in samples])
        src_tokens = fairseq.data.data_utils.collate_tokens(
            [torch.tensor(s["src_tokens"]) for s in samples],
            self.vocab.pad(),
            self.vocab.eos(),
        )
        return {
            "nsentences": len(samples),
            "ntokens": src_lengths.sum().item(),
            "net_input": {
                "src_tokens": src_tokens.to(self.device),
                "src_lengths": src_lengths.to(self.device),
            },
            "prefix_tokens": prefix_tokens.to(self.device),
        }

    def strip_pad(self, sentence):
        assert sentence.ndim == 1
        return sentence[sentence.ne(self.vocab.pad())]

    @torch.no_grad()
    def inference(self, sentences: tp.List[SampleOrStr]) -> tp.List[str]:
        sample = self.preprocess(sentences)
        generated = self.task.inference_step(
            self.sequence_generator, [self.model], sample
        )
        # `generate` returns a list of samples
        # with several hypothesis per sample
        # and a dict per hypothesis.
        # We also need to strip the language token.
        # We also need to restore the original ordering of the samples
        hypothesis = [hypos[0]["tokens"][1:] for hypos in generated]

        translations = [
            self.vocab.string(self.strip_pad(sentence), "sentencepiece")
            for sentence in hypothesis
        ]
        assert len(sentences) == len(translations)
        return translations

    def translate(self, it: tp.Iterable[SampleOrStr]) -> tp.Iterator[str]:
        yield from self.cache.map_batches(self.inference, it, self.config.batch_size)


def resolve_task_vocab(task, tgt_lang: str):
    if hasattr(task, "tgt_dict"):
        return task.tgt_dict
    if hasattr(task, "dicts"):
        assert (
            tgt_lang in task.dicts
        ), f"This model doesn't know lang {tgt_lang}, chose from {list(task.dicts.keys())}"
        return task.dicts[tgt_lang]

    raise NotImplementedError(f"fairseq_generate.py doesn't handle {type(task)}")


def resolve_spm_path(model_cfg) -> str:
    """Given a Fairseq config will look around for the spm model used to binarize the data"""
    log.warn("SPM not passed explictly will try to infer it automatically")
    data_bin_shard0 = Path(model_cfg.task.data)
    vocab_bin = data_bin_shard0.parent.parent / "vocab_bin"
    log.info(
        f"Will look into {vocab_bin}, because model was trained on {data_bin_shard0}"
    )
    assert vocab_bin.exists(), f"{vocab_bin} not found !"
    spm_paths = list(vocab_bin.glob("*.model"))
    assert spm_paths, f"No spm model found at {vocab_bin}"
    assert len(spm_paths) == 1, f"Too many spm models found at {vocab_bin}"
    return str(spm_paths[0])
