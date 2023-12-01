# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import logging
import typing as tp
from pathlib import Path

from stopes.core import utils
from stopes.core.stopes_module import Requirements, StopesModule

logger = logging.getLogger("segment_and_lid")


class SegmentLIDModule(StopesModule):
    """
    Does both segmentation and lid using those two modules.

    python -m stopes.modules \
        +speech_preproc=segment_and_lid \
        speech_preproc.shards="thai_example.mp3" \
        speech_preproc.segment_model=silero_vad.jit \
        speech_preproc.lid_model=epaca_100/best \
        speech_preproc.output_dir=test-lid-thai
    """

    def __init__(self, config):
        super().__init__(config)
        self.shards = self.config.shards
        if isinstance(self.shards, str):
            # it's a glob instead of a list of files
            self.shards = glob.glob(self.shards)
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def requirements(self) -> Requirements:
        # TODO: take the most expensive setting between segmentation and lid.
        reqs = self.config.lid.requirements
        if not isinstance(reqs, Requirements):
            # perform conversion if needed
            return Requirements(**reqs)
        return reqs

    def array(self) -> tp.List[tp.List[Path]]:
        return utils.make_duration_batches(
            self.shards, self.config.max_duration_in_seconds
        )

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        segment_module = StopesModule.build(
            self.config.segment,
            shards=[str(iteration_value)],
        )
        segmented_file, nb_lines = segment_module.run(
            iteration_value=str(iteration_value),
            iteration_index=iteration_index,
        )

        lid_module = StopesModule.build(
            self.config.lid,
            shards=[str(segmented_file)],
        )
        return lid_module.run(
            iteration_value=str(segmented_file),
            iteration_index=iteration_index,
        )

    def name(self) -> str:
        return f"segment_and_lid.{len(self.shards)}"
