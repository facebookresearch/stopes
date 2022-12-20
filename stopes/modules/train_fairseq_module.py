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

import submitit
from fairseq.dataclass.configs import FairseqConfig
from omegaconf.omegaconf import MISSING, OmegaConf

from stopes.core.stopes_module import Requirements, StopesModule
from stopes.utils.data_utils import DataConfig

log = logging.getLogger("stopes.train_fairseq")


@dataclass
class TrainFairseqConfig:
    params: FairseqConfig = MISSING
    # number of gpus to use for training the model
    num_gpus: int = 1
    # number of gpus per node in your cluster
    num_gpus_per_node: int = 8
    output_dir: str = "train_fairseq"
    timeout_min: int = 3 * 24 * 60


class TrainFairseqModule(StopesModule, submitit.helpers.Checkpointable):
    def __init__(
        self,
        config: TrainFairseqConfig,
    ):
        super().__init__(config, TrainFairseqConfig)
        self.outdir = Path(self.config.output_dir).resolve()
        self.outdir.mkdir(exist_ok=True)
        if not hasattr(config.params.checkpoint, "save_dir"):
            self.checkpoint_dir = (
                self.outdir
                / self.name()
                / f"{config.params.task.source_lang}-{config.params.task.target_lang}"
            )
        else:
            self.checkpoint_dir = Path(config.params.checkpoint.save_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

    def requirements(self):
        num_gpus = self.config.num_gpus
        num_gpus_per_node = self.config.num_gpus_per_node
        assert (
            num_gpus < num_gpus_per_node or num_gpus % num_gpus_per_node == 0
        ), f"Can't split {num_gpus} across several nodes"
        n_nodes = math.ceil(num_gpus / num_gpus_per_node)
        num_gpus_per_node = min(num_gpus, num_gpus_per_node)
        return Requirements(
            nodes=n_nodes,
            mem_gb=num_gpus_per_node * 50,
            tasks_per_node=num_gpus_per_node,
            gpus_per_node=num_gpus_per_node,
            cpus_per_task=4,
            timeout_min=self.config.timeout_min,
        )

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> Path:
        import fairseq.distributed.utils
        import fairseq_cli.train
        import torch
        from fairseq.dataclass.configs import FairseqConfig

        env = submitit.helpers.TorchDistributedEnvironment()
        env.export()
        fairseq_cfg = OmegaConf.merge(FairseqConfig(), self.config.params)
        fairseq_cfg.common.reset_logging = True
        fairseq_cfg.checkpoint.save_dir = str(self.checkpoint_dir)

        # Even though we won't be calling fairseq.distributed_init,
        # set up the config because other parts of the code are reading it.
        # Also it's less confusing is someone look at this in fairseq logs.
        distributed_cfg = fairseq_cfg.distributed_training
        distributed_cfg.distributed_world_size = env.world_size
        distributed_cfg.distributed_num_procs = 0
        distributed_cfg.distributed_rank = env.rank
        distributed_cfg.distributed_init_method = "env://"
        distributed_cfg.distributed_port = env.master_port
        distributed_cfg.device_id = env.local_rank
        distributed_cfg.distributed_no_spawn = True
        log.info(
            f"Setting up distributed training with: {fairseq_cfg.distributed_training}"
        )

        if env.world_size > 1:
            # Directly call torch.distributed.init_process_group ourselves
            # the logic in fairseq.distributed_init is quite complex
            torch.cuda.set_device(distributed_cfg.device_id)
            torch.distributed.init_process_group(
                backend=distributed_cfg.distributed_backend,
                init_method=distributed_cfg.distributed_init_method,
            )
            assert env.rank == torch.distributed.get_rank()
            assert env.world_size == torch.distributed.get_world_size()
            tensor = torch.zeros(env.world_size).cuda()
            tensor[env.rank] = env.rank
            torch.distributed.all_reduce(tensor)
            result = list(tensor)
            log.info(
                f"Successfully run all-reduce across {env.world_size} nodes: {result}"
            )
            assert result == list(range(env.world_size))

        best_val = fairseq_cli.train.main(fairseq_cfg)
        log.info(f"Finished training model, got a score of {best_val}")
        if env.world_size > 1:
            log.info(f"Waiting for all other nodes")
            # Wait for all process
            torch.distributed.barrier(torch.distributed.new_group())

        log.info(f"Finished training model, got a score of {best_val}")
        # based on naming scheme of train fairseq
        best_checkpoint_path = self.checkpoint_dir / f"checkpoint_best.pt"
        return best_checkpoint_path.resolve()

    @classmethod
    def version(cls):
        return "0.2"

    def name(self):
        if isinstance(self.config.params.task, str):
            return self.config.params.task
        elif getattr(self.config.params.task, "_name", False):
            if self.config.params.task._name == "translation":
                src_lang = self.config.params.task.source_lang
                tgt_lang = self.config.params.task.target_lang
                return f"NMT:{src_lang}-{tgt_lang}"
            return f"train_fairseq_{self.config.params.task._name}"
        else:
            return f"train_fairseq_{self.config.params.task.task}"
