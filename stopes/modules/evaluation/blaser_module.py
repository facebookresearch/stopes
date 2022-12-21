# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import typing as tp

from stopes.core.stopes_module import Requirements, StopesModule
from stopes.eval.blaser.score import BlaserEvalConfig
from stopes.eval.blaser.score import run as blaser_score


class BlaserEvalModule(StopesModule):
    def __init__(self, config: tp.Any):
        super().__init__(config, BlaserEvalConfig)
        self.config: BlaserEvalConfig
        self.config.output_dir.mkdir(exist_ok=True, parents=True)

    def requirements(self) -> Requirements:
        return Requirements(
            gpus_per_node=1 if self.config.use_gpu else 0,
        )

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> tp.Any:
        return blaser_score(self.config)

    def validate(
        self,
        output: tp.Any,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> bool:
        _corr, score_file = output
        return score_file.exists()
