# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import os
import typing as tp
from pathlib import Path

import hydra
from seamless_communication.cli.m4t.evaluate.evaluate import main as evaluate_main

from stopes.core import utils
from stopes.core.stopes_module import Requirements, StopesModule
from stopes.pipelines.m4t_eval.config_def import (
    M4TEvalConfig,
    M4TEvalJob,
    M4TEvalModuleConfig,
)

logger = logging.getLogger("m4t_eval")


class M4TEval(StopesModule):
    def __init__(
        self,
        config: M4TEvalModuleConfig,
    ):
        super().__init__(config, M4TEvalModuleConfig)
        self.config: M4TEvalModuleConfig

    def array(self):
        return self.config.eval_jobs

    def requirements(self):
        return Requirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=1,
            cpus_per_task=2,
            timeout_min=24 * 60,
            constraint="volta32gb",
        )

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        assert iteration_value is not None, "iteration value is null"
        src_lang, tgt_lang = iteration_value.src_lang, iteration_value.tgt_lang
        dataset_name = f"{self.config.dataset_split}_{src_lang}-{tgt_lang}"
        dataset_path = Path(self.config.data_dir) / f"{dataset_name}.tsv"
        self.logger = logging.getLogger("stopes.m4t_eval")
        self.logger.info(f"Running eval on {dataset_name}.tsv")
        utils.ensure_dir(self.config.output_dir)
        args = {
            "data_file": dataset_path,
            "task": self.config.task,
            "tgt_lang": tgt_lang,
            "output_path": Path(self.config.output_dir),
            "src_lang": src_lang,
            "audio_root_dir": self.config.audio_root_dir,
        }
        kwargs: tp.Dict[str, tp.Any] = dict(self.config.kwargs)
        args.update(kwargs)
        self.logger.info(f"args for m4t_evaluate script - {args}")
        evaluate_main(args)
        return


async def m4t_eval(config: M4TEvalConfig) -> None:
    logger.info(config)
    utils.ensure_dir(config.output_dir)
    launcher = hydra.utils.instantiate(config.launcher)
    eval_jobs = []
    for lang_dir in config.lang_dirs:
        src_lang, tgt_lang = lang_dir.split("-")
        eval_jobs.append(M4TEvalJob(src_lang=src_lang, tgt_lang=tgt_lang))
        dataset_name = f"{config.dataset_split}_{src_lang}-{tgt_lang}"
        dataset_path = Path(config.data_dir) / f"{dataset_name}.tsv"
        if not os.path.exists(dataset_path):
            raise Exception(f"Please make sure the dataset_path {dataset_path} exists")
    eval_module = M4TEval(
        M4TEvalModuleConfig(
            output_dir=config.output_dir,
            data_dir=config.data_dir,
            dataset_split=config.dataset_split,
            task=config.task,
            audio_root_dir=config.audio_root_dir,
            kwargs=config.kwargs,
            eval_jobs=eval_jobs,
        )
    )
    await launcher.schedule(eval_module)


@hydra.main(config_path="conf", config_name="m4t_eval", version_base="1.1")
def main(config: M4TEvalConfig) -> None:
    asyncio.run(m4t_eval(config))


if __name__ == "__main__":
    main()
