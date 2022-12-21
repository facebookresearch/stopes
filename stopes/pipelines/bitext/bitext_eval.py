# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import contextlib
import dataclasses
import itertools
import logging
import os
import re
import typing as tp
from pathlib import Path

import fairseq.dataclass.configs
import hydra
import hydra.core.config_store
import omegaconf
from omegaconf import MISSING, DictConfig, OmegaConf
from submitit import Job

import stopes.core
import stopes.modules.preprocess as preprocess
from stopes.core import utils
from stopes.modules import train_fairseq_module

logger = logging.getLogger("bitext_eval")


@dataclasses.dataclass
class BitextEvalConfig:
    launcher: DictConfig
    bitext: str
    test_data_dir: str
    valid_data_dir: str
    output_dir: str
    langs: tp.List[str]
    spm: preprocess.TrainSpmConfig
    train_fairseq: fairseq.dataclass.configs.FairseqConfig


class BitextEvalPipeline(stopes.core.StopesModule):
    def __init__(
        self,
        config: BitextEvalConfig,
    ):
        assert len(config.langs) == 2
        # Make the cache more useful
        config.langs = sorted(config.langs)
        super().__init__(config)
        self.langpair = "-".join(config.langs)

        self.launcher = hydra.utils.instantiate(config.launcher)

        self.output_dir = Path(self.config.output_dir).resolve()
        self.output_dir.mkdir(exist_ok=True)

    async def train_spm_and_binarize(self, lang: str) -> Job[Path]:
        bitext = Path(self.config.bitext)
        assert bitext.exists()
        # TODO: handle reading from tsv files
        lang_file = bitext.with_suffix(f".{lang}")

        with utils.clone_config(self.config.spm.config) as spm_config:
            spm_config.train_data_file = str(lang_file)
            spm_config.output_dir = str(self.output_dir / "spm")

        train_spm_module = preprocess.TrainSpmModule(spm_config)
        spm_vocab = await self.launcher.schedule(train_spm_module)
        spm, spm_dict = spm_vocab.model_file, spm_vocab.dict_file

        train_bin_ = self.binarize(
            lang_file, spm, spm_dict, f"train.{self.langpair}.{lang}"
        )
        valid_bin_ = self.binarize(
            Path(self.config.valid_data_dir % lang),
            spm,
            spm_dict,
            f"valid.{self.langpair}.{lang}",
        )
        test_bin_ = self.binarize(
            Path(self.config.test_data_dir % lang),
            spm,
            spm_dict,
            f"test.{self.langpair}.{lang}",
        )
        train_bin, _, _ = await asyncio.gather(train_bin_, valid_bin_, test_bin_)
        # TODO: this should be moved to train_nmt_and_eval
        # TODO: this can have collision, find a way to avoid them
        utils.symlink(train_bin.parent / f"dict.{lang}.txt", spm_dict)
        return train_bin

    async def binarize(
        self, lang_file: Path, spm: Path, spm_dict: Path, split: str
    ) -> Path:
        assert lang_file.exists()
        bin_dir = self.output_dir / "binarized"
        encoder = omegaconf.OmegaConf.create(
            {
                "_target_": f"{preprocess.FairSeqBinarizerEncoder.__module__}.FairSeqBinarizerEncoder",
                "spm_model_path": str(spm),
                "vocab_file_path": str(spm_dict),
                "dataset_impl": "mmap",
            }
        )
        encode_conf = preprocess.LineProcessorConfig(
            line_processor=encoder,
            output_dir=str(bin_dir),
            outfile_prefix=split,
            shards=[lang_file],
        )
        encode_module = preprocess.LineProcessorModule(encode_conf)
        encoded = await self.launcher.schedule(encode_module)
        return encoded[0]

    async def train_nmt_and_eval(
        self, src: str, src_bin: Path, tgt: str, tgt_bin: Path
    ) -> Path:
        data = src_bin.parent
        assert tgt_bin.parent == data
        # Removes ".000" from file names
        for split, ext, lang in itertools.product(
            ("train", "valid", "test"), (".bin", ".idx"), (src, tgt)
        ):
            prefix = f"{split}.{src}-{tgt}.{lang}"
            reversed = f"{split}.{tgt}-{src}.{lang}"

            input_file = data / (reversed + ".000" + ext)
            if not input_file.exists():
                input_file = data / (prefix + ".000" + ext)
            utils.symlink(data / (prefix + ext), input_file)

        with utils.clone_config(self.config.train_fairseq.config) as nmt_config:
            # Create a unique directory since Fairseq automatically resume existing
            # checkpoints.
            outdir = self.output_dir / "nmt" / f"checkpoints_{self.sha_key()}"
            nmt_config.output_dir = str(outdir.parent)
            nmt_config.data = {}
            nmt_config.params.task.source_lang = src
            nmt_config.params.task.target_lang = tgt
            nmt_config.params.task.data = str(src_bin.parent)
            nmt_config.params.checkpoint.save_dir = str(outdir)

        nmt_module = train_fairseq_module.TrainFairseqModule(nmt_config)
        return await self.launcher.schedule(nmt_module)

    def run(self):
        loop = asyncio.get_event_loop()
        if self.config.launcher.cluster == "debug":
            loop.set_debug(True)
        loop.run_until_complete(self.arun())

    async def arun(self):
        logger.info(f"output: {self.output_dir}")
        logger.info(f"working dir: {os.getcwd()}")
        l0, l1 = self.config.langs
        bin_l0, bin_l1 = await asyncio.gather(
            *[self.train_spm_and_binarize(l) for l in (l0, l1)]
        )

        model_l0_l1, model_l1_l0 = await asyncio.gather(
            self.train_nmt_and_eval(l0, bin_l0, l1, bin_l1),
            self.train_nmt_and_eval(l1, bin_l1, l0, bin_l0),
        )

    @classmethod
    def version(cls):
        return "0.1"


@hydra.main(config_path="conf", config_name="bitext_eval")
def main(config: BitextEvalConfig) -> None:
    pipeline = BitextEvalPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
