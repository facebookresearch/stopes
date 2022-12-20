# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import shlex
import shutil
import subprocess
import typing as tp
from pathlib import Path

import submitit

from stopes.core import stopes_module
from stopes.core.utils import bash_pipefail
from stopes.pipelines.monolingual.utils import slurm_tmp_maybe
from stopes.pipelines.monolingual.utils.sort import build_sort_command

logger = logging.getLogger("dedup_module")


class DedupeWithMergeSort(stopes_module.StopesModule):
    """
    this assumes that the input files in config.shards are already sorted
    """

    def requirements(self) -> stopes_module.Requirements:
        return stopes_module.Requirements(
            nodes=1,
            mem_gb=getattr(self.config, "mem_gb", 1),
            tasks_per_node=1,
            cpus_per_task=getattr(self.config, "num_cpu", 40),
            gpus_per_node=0,
            timeout_min=getattr(self.config, "timeout_min", 14400),
        )

    def array(self) -> tp.Optional[tp.List[tp.Any]]:
        do_local_dedup = getattr(self.config, "do_local_dedup", False)
        print(f"Doing local dedup first: {do_local_dedup}")
        return self.config.shards if do_local_dedup else None

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> tp.Any:
        tmp_dir = slurm_tmp_maybe(Path(self.config.tmp_dir))
        stm = Path(self.config.output_file).stem
        out_dir = Path(self.config.output_file).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        if iteration_value:
            # we are deduping a single file locally
            nb_shards = len(self.config.shards)
            print(
                f"Locally deduping {iteration_value}, {iteration_index + 1}th shard out"
                f" of {nb_shards}"
            )
            files_to_process = [Path(iteration_value)]
            is_merge = False
            tmp_dir = tmp_dir / f"{stm}_local_{iteration_index}"
            output_file = out_dir / f"{Path(iteration_value).stem}_deduped"
            if getattr(self.config, "compress", True):
                output_file = output_file.with_suffix(".xz")
            else:
                output_file = output_file.with_suffix(".txt")

            completion_message = (
                f"Done deduping {iteration_value}, {iteration_index + 1}th shard out"
                f" of {nb_shards}"
            )

        else:
            # we are now deduping globally across all shards previously deduped
            print(f"Globally deduping now")
            files_to_process = [Path(f) for f in self.config.shards]
            is_merge = True
            tmp_dir = tmp_dir / stm
            output_file = Path(self.config.output_file)

            completion_message = "Done deduping globally"

        tmp_dir.mkdir(parents=True, exist_ok=True)

        if self.config.process_locally:
            for f in files_to_process:
                local_file = tmp_dir / f.name
                shutil.copyfile(str(f), str(local_file))

        # unpack sorted shards
        # unxzipped = []
        # for file in self.config.shards:
        #     f = Path(file)
        #     unxz = tmp_dir / f.stem
        #     subprocess.run(
        #         " ".join([open_file_cmd(str(file)), ">", shlex.quote(str(unxz))]),
        #         shell=True,
        #         check=True,
        #     )
        #     unxzipped.append(unxz)

        sort_cmd = build_sort_command(
            files=files_to_process,
            is_merge=is_merge,
            num_cpu=self.config.num_cpu,
            tmp_dir=tmp_dir,
            field_def=self.config.field_def,
        )
        # merge them
        if getattr(self.config, "compress", True):
            full_cmd = bash_pipefail(
                sort_cmd,
                " ".join(["xz", ">", shlex.quote(str(output_file))]),
            )
        else:
            full_cmd = bash_pipefail(
                " ".join([sort_cmd, ">", shlex.quote(str(output_file))])
            )

        subprocess.run(
            full_cmd,
            shell=True,
            check=True,
        )

        # try:
        #     for file in unxzipped:
        #         file.unlink()
        # except Exception as e:
        #     logger.warning("couldn't remove temp files", exc_info=e)

        print(completion_message)
        return output_file

    def checkpoint(
        self, *args: tp.Any, **kwargs: tp.Any
    ) -> submitit.helpers.DelayedSubmission:
        return submitit.helpers.DelayedSubmission(
            self, *args, **kwargs
        )  # submits to requeuing

    def name(self):
        return f"dedup.{Path(self.config.output_file).stem}"
