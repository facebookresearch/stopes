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

logger = logging.getLogger("f0extraction")

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fairseq.data.audio.audio_utils import get_features_or_waveform
from librosa.util import normalize
from scipy.signal import medfilt
from tqdm import tqdm

F0_FRAME_LENGTH = 0.020  # sec
F0_FRAME_SPACE = 0.005  # sec


def get_f0(filename, rate=16000, type="yaapt"):
    audio = get_features_or_waveform(filename, need_waveform=True, use_sample_rate=rate)
    assert audio.ndim == 1

    to_pad = int(F0_FRAME_LENGTH * rate) // 2

    audio = normalize(audio) * 0.95
    audio = np.pad(audio, (to_pad, to_pad), "constant", constant_values=0)

    if type == "yaapt":
        try:
            import amfm_decompy.basic_tools as basic
            import amfm_decompy.pYAAPT as pYAAPT
        except ImportError:
            raise ImportError("Please install amfm_decompy: pip install AMFM-decompy.")

        audio = basic.SignalObj(audio, rate)
        f0 = pYAAPT.yaapt(
            audio,
            frame_length=F0_FRAME_LENGTH * 1000,
            frame_space=F0_FRAME_SPACE * 1000,
            nccf_thresh1=0.25,
            tda_frame_length=25.0,
        )
        f0 = f0.samp_values

    elif type == "dio":
        try:
            import pyworld
        except ImportError:
            raise ImportError("Please install PyWORLD: pip install pyworld")

        audio = audio.astype(np.float64)
        f0, f0_time = pyworld.dio(audio, rate, frame_period=F0_FRAME_SPACE * 1000)
        f0 = pyworld.stonemask(audio, f0, f0_time, rate)
        f0 = medfilt(f0, kernel_size=3)
    else:
        raise TypeError(f'type should be either "yaapt" or "dio". Current type={type}')

    return f0


# copied from https://github.com/facebookresearch/fairseq/blob/main/examples/textless_nlp/pgslm/data_utils.py
class Stat:
    def __init__(self, keep_raw=False):
        self.x = 0.0
        self.x2 = 0.0
        self.z = 0.0  # z = logx
        self.z2 = 0.0
        self.n = 0.0
        self.u = 0.0
        self.keep_raw = keep_raw
        self.raw = []

    def update(self, new_x):
        new_z = new_x.log()

        self.x += new_x.sum()
        self.x2 += (new_x**2).sum()
        self.z += new_z.sum()
        self.z2 += (new_z**2).sum()
        self.n += len(new_x)
        self.u += 1

        if self.keep_raw:
            self.raw.append(new_x)

    @property
    def mean(self):
        return self.x / self.n

    @property
    def std(self):
        return (self.x2 / self.n - self.mean**2) ** 0.5

    @property
    def mean_log(self):
        return self.z / self.n

    @property
    def std_log(self):
        return (self.z2 / self.n - self.mean_log**2) ** 0.5

    @property
    def n_frms(self):
        return self.n

    @property
    def n_utts(self):
        return self.u

    @property
    def raw_data(self):
        assert self.keep_raw, "does not support storing raw data!"
        return torch.cat(self.raw)


class F0Stat(Stat):
    def update(self, new_x):
        # assume unvoiced frames are 0 and consider only voiced frames
        if new_x is not None:
            super().update(new_x[new_x != 0])


def process_one(path, sr, type="yaapt"):
    """
    Args:
        path: audio file path
        sr: sampling rate
    """
    try:
        # YAAPT throws errors in some rare cases
        f0 = get_f0(path, sr, type=type)
        f0 = torch.from_numpy(f0.astype(np.float32))
    except Exception as e:
        print(
            f"WARNING: error when processing {path}. set f0 to None. original error message:\n{e}"
        )
        f0 = None
    return f0


@dataclass
class F0ExtractionConfig:
    input_manifest: str = MISSING
    output_folder: str = MISSING
    audio_column_name: str = "audio"
    f0_extractor: str = "dio"  # yaapt
    audio_root_dirpath: str = MISSING
    sampling_rate: int = 16000
    nshards: int = 1

    #    line_processor: tp.Any = MISSING
    #    output_dir: str = MISSING
    #    outfile_prefix: str = "embed"
    #    outfile_postfix: str = ""
    #    # shards is either a list of files or a glob string
    #    # if only hydra allowed, the right type would be tp.Union[str, tp.List[str]]
    #    shards: tp.Any = MISSING
    #    buffer_size: int = 10_000
    requirements: Requirements = Requirements(
        nodes=1,
        tasks_per_node=1,
        gpus_per_node=0,
        cpus_per_task=10,
        timeout_min=240,
    )
    custom_name: str = ""


#    parser.add_argument("--nshards", type=int, default=1)
#    parser.add_argument("--rank", type=int, default=1)
#    parser.add_argument("--sampling_rate", type=int, default=16000)


class F0ExtractionModule(StopesModule):
    def __init__(
        self,
        config: F0ExtractionConfig = F0ExtractionConfig(),
        # processed_lines: int = 0,
        validate_config: bool = False,
    ):
        super().__init__(
            config,
            # TODO: always validate that config is a LineProcessorConfig
            # This is not possible currently because several config files add extra args
            # to make it easier to type the config
            config_class=F0ExtractionConfig if validate_config else None,
        )
        # we do basic checkpointing with submitit Checkpointable which will store the state of this
        # callable. The basic idea here is to remember the last line processed
        # self.processed_lines = processed_lines
        Path(config.output_folder).mkdir(exist_ok=True)

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
        shard_id = None
        if iteration_value is not None:
            shard_id = int(iteration_value)

        df = pd.read_csv(self.config.input_manifest, sep="\t", quoting=3)
        num_audios = len(df["id"])

        # shard
        assert (
            self.config.nshards <= num_audios and self.config.nshards > 0
        ), "--nshards should be in [1, #audios]"
        assert shard_id is not None and shard_id >= 0 and shard_id < self.config.nshards

        shard_size = num_audios / self.config.nshards
        s = int(round((shard_id) * shard_size))
        e = int(round((shard_id + 1) * shard_size))

        # process
        f0_data = {}
        for i, audio_path in enumerate(tqdm(df[self.config.audio_column_name][s:e])):
            if self.config.audio_root_dirpath:
                audio_path = f"{self.config.audio_root_dirpath}/{audio_path}"
            sample_id = df["id"][s + i]
            f0 = process_one(
                audio_path, self.config.sampling_rate, self.config.f0_extractor
            )
            f0_data[sample_id] = f0
        print(f"finished processing {len(f0_data)} utterances ({s}-{e})")

        f0_path = (
            Path(self.config.output_folder)
            / f"f0_{self.config.f0_extractor}"
            / f"f0_{shard_id}_{self.config.nshards}.pt"
        )
        os.makedirs(f0_path.parent, exist_ok=True)
        torch.save(f0_data, f0_path)
        print(f"saved to {f0_path}")

        return f0_path

    @staticmethod
    def version() -> str:
        return "0.1"


#    def name(self) -> str:
#        return getattr(self.config, "custom_name", None) or "_".join(
#            [
#                "LineProc",
#                self.config.line_processor._target_.split(".")[-1],
#                self.sha_key(),
#            ]
#        )

# def checkpoint(
#     self, *args: tp.Any, **kwargs: tp.Any
# ) -> submitit.core.utils.DelayedSubmission:
#     """Resubmits the same module with the same arguments"""
#     return submitit.core.utils.DelayedSubmission(
#         LineProcessorModule(
#             config=self.config, processed_lines=self.processed_lines
#         ),
#         *args,
#         **kwargs,
#     )
