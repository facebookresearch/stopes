# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import math
import os
import typing as tp
from dataclasses import dataclass
from pathlib import Path

from omegaconf.omegaconf import MISSING, OmegaConf

from stopes.core.launcher import Launcher
from stopes.core.stopes_module import DistributedRequirements, StopesModule
from stopes.core.utils import ensure_dir
from stopes.utils.data_utils import DataConfig

if tp.TYPE_CHECKING:
    from fairseq.dataclass.configs import FairseqConfig

logger = logging.getLogger("stopes.train_fairseq")


@dataclass
class TrainFairseqConfig:
    params: "FairseqConfig" = MISSING
    data: DataConfig = MISSING
    # number of gpus to use for training the model
    num_gpus: int = 1
    # number of gpus per node in your cluster
    num_gpus_per_node: int = 8
    output_dir: str = "train.${data.data_version}/checkpoints"


class TrainFairseqModule(StopesModule):
    def __init__(
        self,
        config: TrainFairseqConfig = TrainFairseqConfig(),
    ):
        super().__init__(config)
        ensure_dir(self.config.output_dir)

    def requirements(self):
        num_gpus = self.config.num_gpus
        num_gpus_per_node = self.config.num_gpus_per_node
        assert (
            num_gpus < num_gpus_per_node or num_gpus % num_gpus_per_node == 0
        ), f"Can't split {num_gpus} across several nodes"
        n_nodes = math.ceil(num_gpus / num_gpus_per_node)
        num_gpus_per_node = min(num_gpus, num_gpus_per_node)
        return DistributedRequirements(
            nodes=n_nodes,
            mem_gb=num_gpus_per_node * 50,
            tasks_per_node=1,
            gpus_per_node=num_gpus_per_node,
            cpus_per_task=4,
            timeout_min=24 * 60,
        )

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
        launcher: tp.Optional[Launcher] = None,
    ) -> Path:
        from fairseq.dataclass.configs import FairseqConfig
        from fairseq_cli.hydra_train import hydra_main

        fairseq_cfg = OmegaConf.merge(FairseqConfig(), self.config.params)
        fairseq_cfg.common.reset_logging = True
        if not hasattr(self.config.params.checkpoint, "save_dir"):
            fairseq_cfg.checkpoint.save_dir = os.path.abspath(
                os.path.join(
                    self.config.output_dir,
                    self.name(),
                    f"{self.config.params.task.source_lang}-{self.config.params.task.target_lang}",
                )
            )
        ensure_dir(fairseq_cfg.checkpoint.save_dir)

        num_gpus = self.config.num_gpus
        if num_gpus > 1:
            fairseq_cfg.distributed_training.distributed_port = 9218
            fairseq_cfg.distributed_training.distributed_world_size = num_gpus

        best_val = hydra_main(fairseq_cfg)
        logger.info(f"Finished training model, got a score of {best_val}")

        best_checkpoint_path = (
            Path(self.config.params.checkpoint.save_dir)
            / f"checkpoint_best.pt"  # based on naming scheme of train fairseq
        ).resolve()

        return best_checkpoint_path

    @classmethod
    def version(cls):
        return "0.2"

    def name(self):
        return f"train_fairseq_{self.config.params.task._name}"

    def comment(self):
        return f"Training Fairseq task: {self.config.params.task._name}."
