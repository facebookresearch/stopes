# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import logging
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from omegaconf import MISSING
from tqdm import tqdm
from ust_common.lib.manifests import (
    audio_tsv_from_files,
    save_to_tsv,
    zip_from_audio_manifest,
    zip_from_file_stream,
)

from stopes.core import Requirements, StopesModule, utils

logger = logging.getLogger("audio_preprocess_utils")


def merge_tsvs_if_sharded(tsvs, output_manifest: Path) -> Path:
    if (len(tsvs) > 1) or (Path(tsvs[0]).resolve() != output_manifest.resolve()):
        tsv_df = pd.concat([pd.read_csv(tsv, sep="\t", quoting=3) for tsv in tsvs])
        tsv_df.to_csv(
            output_manifest,
            sep="\t",
            quoting=3,
            index=None,
        )
        logger.info(
            f"Merge {len(tsvs)} shards into {output_manifest} with {len(tsv_df)} rows"
        )
    return output_manifest


@dataclass
class MergeTSVsConfig:
    tsvs: List[str] = MISSING
    output_manifest: str = MISSING


class MergeTSVsModule(StopesModule):
    def __init__(
        self,
        config: MergeTSVsConfig = MergeTSVsConfig(),
        validate_config: bool = False,
    ):
        super().__init__(
            config,
            # TODO: always validate that config is a LineProcessorConfig
            # This is not possible currently because several config files add extra args
            # to make it easier to type the config
            config_class=MergeTSVsConfig if validate_config else None,
        )

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> tp.Any:
        return merge_tsvs_if_sharded(
            self.config.tsvs, Path(self.config.output_manifest)
        )

    def requirements(self) -> Requirements:
        """
        return a set of Requirements for your module, like num of gpus etc.
        """
        return Requirements()

    @staticmethod
    def version() -> str:
        return "0.1"


def compile_feat(
    input_manifest: Path,
    output_manifest: Path,
    feat_folder: Path,
    feat_column_name: str,
    nshards: int = 1,
):
    logger.info(
        f"compile feat, input_manifest = {input_manifest}, output_manifest = {output_manifest}"
    )
    input_df = pd.read_csv(input_manifest, sep="\t", quoting=3)

    def zipfilepath_and_byteio(df):
        df_ids = set(df["id"].tolist())
        with tqdm() as pbar:
            for ns in range(nshards):
                feat_path = feat_folder / f"{feat_column_name}_{ns}_{nshards}.pt"
                single_feat = torch.load(feat_path)
                for key, value in single_feat.items():
                    if key not in df_ids:
                        logger.info(f"{key} not found. Skip!")
                        continue
                    if value is None:
                        logger.info(
                            f"No feat available for {key} in {feat_path}. Skip!"
                        )
                        continue
                    zipfile_name = f"{key}.npy"
                    byteio = io.BytesIO()
                    np.save(byteio, value)
                    byteio.seek(0)
                    pbar.update(1)
                    yield (zipfile_name, byteio)

    zip_manifest_df = zip_from_file_stream(
        # zipfilepath_and_byteio(input_df, feats),
        zipfilepath_and_byteio(input_df),
        output_folder=feat_folder,
        zip_name=f"{input_manifest.stem}_{feat_column_name}.zip",
        output_manifest_name=None,
        output_column_name=feat_column_name,
        no_progress_bar=False,
        return_df=True,
        is_audio=False,
    )
    logger.info("Start saving to tsv ...")
    save_to_tsv(
        output_manifest.as_posix(),
        #        (
        #            feat_folder / f"{input_manifest.stem}_{feat_column_name}.tsv"
        #        ).as_posix(),
        zip_manifest_df[[feat_column_name]],
        index=True,
        index_label="id",
    )


@dataclass
class CompileFeatConfig:
    input_manifest: str = MISSING
    output_manifest: str = MISSING
    feat_folder: str = MISSING
    feat_column_name: str = MISSING
    nshards: int = 1


class CompileFeatModule(StopesModule):
    def __init__(
        self,
        config: CompileFeatConfig = CompileFeatConfig(),
        validate_config: bool = False,
    ):
        super().__init__(
            config,
            # TODO: always validate that config is a LineProcessorConfig
            # This is not possible currently because several config files add extra args
            # to make it easier to type the config
            config_class=CompileFeatConfig if validate_config else None,
        )

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> tp.Any:
        return compile_feat(
            Path(self.config.input_manifest),
            Path(self.config.output_manifest),
            Path(self.config.feat_folder),
            self.config.feat_column_name,
            self.config.nshards,
        )

    def requirements(self) -> Requirements:
        """
        return a set of Requirements for your module, like num of gpus etc.
        """
        return Requirements()

    @staticmethod
    def version() -> str:
        return "0.1"
