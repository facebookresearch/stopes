# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import logging
import typing as tp
from abc import abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

import hydra
from omegaconf import MISSING

import stopes.core
from stopes.core import Requirements, StopesModule, utils

logger = logging.getLogger("Audio_segmentation")

import argparse
import hashlib
import os
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import fairseq
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fairseq.data.audio.audio_utils import get_features_or_waveform
from tqdm import tqdm
from ust_common.lib.manifests import audio_tsv_from_files, zip_from_audio_manifest

from .audio import (
    apply_silero_vad_audio_list,
    apply_vad_audio_list,
    convert_float_to_pcm_list,
    convert_pcm_to_float_list,
    convert_to_mono_audio,
    init_silero_vad,
    init_vad,
    load_audio,
    multiprocess_map,
    pipeline,
)
from .audio import pipeline_func_decorator as pcd
from .audio import slice_audio, write_audios


def stable_hash(val):
    return int(hashlib.md5(str(val).encode()).hexdigest(), 16)


import tempfile


def get_local_temporary_folder():
    # if it's slurm array
    if "SLURM_JOB_ID" in os.environ:
        folder = f"/scratch/slurm_tmpdir/{os.environ['SLURM_JOB_ID']}"
        if "SLURM_ARRAY_TASK_ID" in os.environ:
            folder = f"{folder}/{os.environ['SLURM_ARRAY_TASK_ID']}"
    else:
        folder = f"/raid/{os.getlogin()}/__tmp_audio_segmentation"
    os.makedirs(folder, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=folder)


@dataclass
class AudioSegmentationConfig:
    input_manifest: str = MISSING
    output_folder: str = MISSING
    nshards: int = 1
    audio_column_name: str = "src_audio"
    audio_files_root_dirpath: str = MISSING
    output_audio_extension: str = MISSING
    target_sample_rate: int = MISSING
    vad_agg: int = MISSING
    vad_silero_prob: float = 0.0
    save_abs_audio_path: bool = False
    keep_only_audio_column: bool = False
    output_audio_column_name: str = MISSING
    verify_audio_duration: bool = False
    no_multiprocess: bool = False
    save_to_zip: bool = False
    requirements: Requirements = Requirements(
        nodes=1,
        tasks_per_node=1,
        gpus_per_node=0,
        cpus_per_task=10,
        timeout_min=240,
    )
    custom_name: str = ""


from multiprocessing import cpu_count


class AudioSegmentationModule(StopesModule):
    def __init__(
        self,
        config: AudioSegmentationConfig = AudioSegmentationConfig(),
        validate_config: bool = False,
    ):
        super().__init__(
            config,
            # TODO: always validate that config is a LineProcessorConfig
            # This is not possible currently because several config files add extra args
            # to make it easier to type the config
            config_class=AudioSegmentationConfig if validate_config else None,
        )
        # we do basic checkpointing with submitit Checkpointable which will store the state of this
        # callable. The basic idea here is to remember the last line processed
        # self.processed_lines = processed_lines
        Path(config.output_folder).mkdir(exist_ok=True, parents=True)

    def array(self) -> tp.List[str]:
        return [str(i) for i in range(self.config.nshards)]

    def requirements(self) -> Requirements:
        reqs = self.config.requirements
        if not isinstance(reqs, Requirements):
            # Performe conversion if needed
            return Requirements(**reqs)
        return reqs

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> tp.Any:
        shard_id = 0
        if iteration_value is not None:
            shard_id = int(iteration_value)

        # HACK: if save to zip, we will dump intermediate file to local for faster file access
        #  and then zip and move back to final target output folder
        if self.config.save_to_zip:
            tmp_folder = get_local_temporary_folder()
            output_folder = Path(tmp_folder.name)
        else:
            output_folder = Path(self.config.output_folder)

        os.makedirs(output_folder / "audio", exist_ok=True)

        logger.info(f"Reading {self.config.input_manifest}")

        df = pd.read_csv(self.config.input_manifest, sep="\t", quoting=3)
        logger.info(
            f"Loaded {self.config.input_manifest} with {len(df)} rows, columsn {df.columns}"
        )

        # get from specific shard based on audio file
        if self.config.nshards > 1:
            assert shard_id is not None
            assert shard_id >= 0 and shard_id < self.config.nshards
            logger.info(
                f"# shards ({self.config.nshards}) > 1, start taking shard {shard_id} (0-indexed) based on audio file path"
            )
            df = df[
                (
                    df[self.config.audio_column_name].apply(stable_hash)
                    % self.config.nshards
                )
                == shard_id
            ].sort_values(by=[self.config.audio_column_name])
            logger.info(f"Shard {shard_id} has {len(df)} rows")

        # Use src_ofst_and_len to decide if we want to slice
        # format is start_time_in_sec:duration_in_sec
        is_need_slice = "src_ofst_and_len" in df
        items = []
        # compose the parameters for multi-processing for pipeline_functions below
        logger.info(f"Preparing job data ... ")
        uniq_audios = []
        audios = df[self.config.audio_column_name].to_numpy()
        for i, audio in enumerate(audios):
            if i == 0 or audio != audios[i - 1]:
                uniq_audios.append(audio)
        # uniq_audios = df[args.audio_column_name].unique()
        last_idx = 0
        curr_idx = 0
        for src_audio in tqdm(uniq_audios):
            # TODO: this step is slow, improve it by building a set
            assert (
                src_audio == df[self.config.audio_column_name].iloc[curr_idx]
            ), f"{src_audio} {df[self.config.audio_column_name].iloc[curr_idx]} {curr_idx}"
            while (curr_idx < len(df)) and (
                df[self.config.audio_column_name].iloc[curr_idx] == src_audio
            ):
                curr_idx += 1
            _df = df.iloc[last_idx:curr_idx]
            last_idx = curr_idx

            ids = _df["id"].tolist()

            src_audio = Path(src_audio)
            audio_extension = self.config.output_audio_extension or src_audio.suffix[1:]

            if is_need_slice:
                src_ofst_and_lens = (
                    _df["src_ofst_and_len"].apply(lambda x: x.split(":")).tolist()
                )
            else:
                src_ofst_and_lens = None
            item = {
                "load_audio": Path(self.config.input_manifest).parent / src_audio,
                "convert_to_mono_audio": self.config.target_sample_rate,
                "slice_audio": [
                    (float(src), float(dur)) for src, dur in src_ofst_and_lens
                ]
                if src_ofst_and_lens is not None
                else None,
                "write_audios": [
                    output_folder / "audio" / f"{id}.{audio_extension}" for id in ids
                ],
            }
            items.append(item)

        pipeline_functions = [
            load_audio,
            convert_to_mono_audio,
            slice_audio,
        ]
        if self.config.vad_agg is not None:
            init_vad(self.config.vad_agg)
            pipeline_functions.extend(
                [
                    convert_float_to_pcm_list,
                    apply_vad_audio_list,
                    convert_pcm_to_float_list,
                ]
            )
        pipeline_functions.append(
            write_audios,
        )

        # multiprocess_map(
        #    items,
        #    partial(pipeline, funcs=pipeline_functions),
        # )

        logger.info(f"Processing audios ({len(items)} audios) ...")
        if self.config.no_multiprocess:
            logger.warning(f"No multiprocess due to no_multiprocess is set to True.")
            for item in tqdm(items):
                pipeline(item, pipeline_functions)
        else:
            # Follow the # cpu resource allocated by slurm if this is a slurm job
            n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))
            logger.info(f"Start multiprocess with {n_workers}")
            multiprocess_map(
                items,
                partial(pipeline, funcs=pipeline_functions),
                n_workers=n_workers,
            )

        # update output audio files in the output manifest and check
        output_audio_column_name = (
            self.config.output_audio_column_name or self.config.audio_column_name
        )

        if self.config.save_abs_audio_path:
            audio_path_resolver = lambda x: (
                output_folder.resolve() / "audio" / f"{x}.{audio_extension}"
            ).as_posix()
        else:
            audio_path_resolver = lambda x: f"audio/{x}.{audio_extension}"

        df[output_audio_column_name] = df["id"].map(audio_path_resolver)
        for src_audio in df[output_audio_column_name]:
            path = output_folder / src_audio
            assert os.path.exists(path), f"{path} does not exists"

        if self.config.verify_audio_duration:
            print("Computing post-VAD durations ")
            df["VAD_durations"] = df[output_audio_column_name].apply(
                lambda x: librosa.get_duration(filename=x)
            )
            print(
                "These audio are likely to not have any speech, dropping it from final manifest:\n"
            )
            degenerate_items = df[df["VAD_durations"] < 0.01][
                self.config.audio_column_name
            ].index
            print("\n".join(degenerate_items.tolist()))
            df.drop(index=degenerate_items, inplace=True)

        if self.config.keep_only_audio_column:
            df = df[[output_audio_column_name]]

        # prepare for output tsv and zip
        # output_name = Path(self.config.input_manifest).stem
        output_name = "audio"
        if self.config.nshards > 1:
            output_name = output_name + "_" + str(shard_id)

        # HACK: if save to zip, we will dump intermediate file to local, zip and move back to final target output folder
        if self.config.save_to_zip:
            zip_df = audio_tsv_from_files(
                output_folder / "audio",
                audio_extension=self.config.output_audio_extension,
                audio_column="audio",
            )
            logger.info(
                f"saving zip to {self.config.output_folder}/{output_name + '.zip'}"
            )
            zip_manifest_df = zip_from_audio_manifest(
                input_manifest=zip_df,
                output_folder=Path(self.config.output_folder),
                audio_column_name="audio",
                zip_name=output_name + ".zip",
                output_manifest_name=output_name + ".zip.tsv",
                remove_audio=False,
                return_df=True,
                is_audio=True,
            )
            assert zip_manifest_df is not None
            df = df.merge(
                zip_manifest_df.reset_index()[["id", "audio", "length"]].rename(
                    columns={"audio": "__audio"}
                ),
                on="id",
                how="inner",
            )
            df[self.config.audio_column_name] = df["__audio"]
            df = df.drop(["__audio"], axis=1)
            # clean up temp folder
            tmp_folder.cleanup()

        output_manifest = Path(self.config.output_folder) / (output_name + ".tsv")
        df.to_csv(output_manifest, sep="\t", quoting=3, index=None)

        logger.info(f"output manifest to {output_manifest} with {len(df)} rows")
        return output_manifest

    @staticmethod
    def version() -> str:
        return "0.1"
