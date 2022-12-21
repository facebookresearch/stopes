# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import shlex
import subprocess
import typing as tp
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from omegaconf.omegaconf import MISSING

from stopes.core import utils
from stopes.core.launcher import Launcher
from stopes.core.stopes_module import Requirements, StopesModule
from stopes.modules.preprocess.moses_cli_module import get_moses_commands
from stopes.pipelines.filtering.dataset import Dataset
from stopes.pipelines.prepare_data.configs import PreprocessingConfig


@dataclass
class RetrieveDataJob:
    datasets: tp.List[Dataset]


@dataclass
class RetrieveDataConfig:
    output_dir: Path
    preprocessing_config: PreprocessingConfig
    retrieve_data_jobs: tp.List[RetrieveDataJob] = MISSING


class RetrieveData(StopesModule):
    def __init__(
        self,
        config: RetrieveDataConfig,
    ):
        super().__init__(config, RetrieveDataConfig)

    def array(self):
        return self.config.retrieve_data_jobs

    def requirements(self) -> Requirements:
        return Requirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=1,
            timeout_min=24 * 60,
        )

    def _run_command(self, command: str, outfile: str):
        self.logger.info(f"Running command: ${command}")
        try:
            subprocess.run(
                f"{command} > {shlex.quote(outfile)}",
                shell=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            if Path(outfile).is_file():
                Path(outfile).unlink()
            self.logger.error(
                f"ERROR during encoding of {outfile}. Deleted corrupted output file.",
                exc_info=e,
            )
            raise e

    def _concatenate_files(
        self,
        datasets: tp.List[Dataset],
        data_type: str,
        lang: tp.Optional[str] = None,
        preprocessing: bool = False,
    ) -> str:
        """
        data_type (str): Valid options: src, tgt, metadata
        """
        lang_dir, fold = datasets[0].lang_dir, datasets[0].fold
        cat_files = [
            utils.open_file_cmd(getattr(dataset, data_type)) for dataset in datasets
        ]
        cmds = ["(" + " && ".join(cat_files) + ")"]
        if lang and preprocessing:
            cmds.extend(get_moses_commands(self.config.preprocessing_config, lang))

        outfile_ext = "metadata" if data_type == "metadata" else lang
        outfile = str(self.config.output_dir / f"{fold}.{lang_dir}.{outfile_ext}")
        cmd = utils.bash_pipefail(*cmds)
        self._run_command(cmd, outfile)
        return outfile

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> Dataset:
        assert iteration_value is not None, "iteration value is null"
        self.logger = logging.getLogger("stopes.prepare_data.retrieve_data")

        retrieve_data_job: RetrieveDataJob = iteration_value
        datasets = retrieve_data_job.datasets
        lang_dir, fold = datasets[0].lang_dir, datasets[0].fold
        src, tgt = lang_dir.split("-")
        self.logger.info(f"Retrieving data for {fold}: {lang_dir}")

        src_file = self._concatenate_files(
            datasets, "src", lang=src, preprocessing=True
        )
        tgt_file = self._concatenate_files(
            datasets, "tgt", lang=tgt, preprocessing=True
        )
        metadata_file = None
        if all(
            dataset.metadata is not None and Path(dataset.metadata).exists()
            for dataset in datasets
        ):
            metadata_file = self._concatenate_files(datasets, "metadata")

        out_dataset = Dataset(
            src=src_file,
            tgt=tgt_file,
            metadata=metadata_file,
            lang_dir=lang_dir,
            fold=fold,
        )
        return out_dataset

    def validate(
        self,
        output: tp.Any,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> bool:
        dataset = output
        return Path(dataset.src).exists() and Path(dataset.tgt).exists()

    def version(self):
        return "0.0"


async def retrieve_data(
    datasets: tp.List[Dataset],
    preprocessing_config: PreprocessingConfig,
    launcher: Launcher,
    output_dir: Path,
) -> tp.List[Dataset]:

    datasets_per_fold_lang_dir = defaultdict(list)
    for dataset in datasets:
        datasets_per_fold_lang_dir[(dataset.fold, dataset.lang_dir)].append(dataset)

    # Launch one retrieve data per (fold, lang_dir)
    retrieve_data_jobs: tp.List[RetrieveDataJob] = []
    for _, datasets in datasets_per_fold_lang_dir.items():
        retrieve_data_jobs.append(RetrieveDataJob(datasets=datasets))

    retrieve_data_module = RetrieveData(
        RetrieveDataConfig(
            output_dir=output_dir,
            preprocessing_config=preprocessing_config,
            retrieve_data_jobs=retrieve_data_jobs,
        )
    )
    retrieved_datasets = await launcher.schedule(retrieve_data_module)
    return retrieved_datasets
