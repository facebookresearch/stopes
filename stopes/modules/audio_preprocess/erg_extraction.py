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

logger = logging.getLogger("erg_extraction")

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fairseq.data.audio.audio_utils import get_features_or_waveform
from tqdm import tqdm

ERG_FRAME_SPACE = 0.005  # sec
ERG_FRAME_LENGTH = 0.035  # sec


def get_energy(filename, rate=16000):
    hop_size = int(rate * ERG_FRAME_SPACE)
    win_len = int(rate * ERG_FRAME_LENGTH)

    # load audio
    audio = get_features_or_waveform(filename, need_waveform=True, use_sample_rate=rate)
    audio = torch.from_numpy(audio).view(1, 1, -1)

    # compute energy
    audio_frm = F.unfold(
        audio.unsqueeze(-1),
        kernel_size=(win_len, 1),
        stride=(hop_size, 1),
    ).transpose(1, 2)
    energy = torch.norm(audio_frm, dim=-1).squeeze(0)

    # convert to decibel scale
    energy = 20 * torch.log10(energy + 1e-7)
    return energy.cpu().numpy()


class ErgStat:
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
        if new_x is None:
            return
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
    def n_frms(self):
        return self.n

    @property
    def n_utts(self):
        return self.u

    @property
    def raw_data(self):
        assert self.keep_raw, "does not support storing raw data!"
        return torch.cat(self.raw)


def process_one(path, sr):
    """
    Args:
        path: audio file path
        sr: sampling rate
    """
    try:
        erg = get_energy(path, sr)
    except Exception as e:
        print(
            f"WARNING: error when processing {path}. set erg to None. original error message:\n{e}"
        )
        erg = None
    return erg


@dataclass
class ErgExtractionConfig:
    input_manifest: str = MISSING
    output_folder: str = MISSING
    audio_column_name: str = "audio"
    f0_extractor: str = "dio"  # yaapt
    audio_root_dirpath: str = MISSING
    sampling_rate: int = 16000
    nshards: int = 1


class ErgExtractionModule(StopesModule):
    def __init__(
        self,
        config: ErgExtractionConfig = ErgExtractionConfig(),
        # processed_lines: int = 0,
        validate_config: bool = False,
    ):
        super().__init__(
            config,
            # TODO: always validate that config is a LineProcessorConfig
            # This is not possible currently because several config files add extra args
            # to make it easier to type the config
            config_class=ErgExtractionConfig if validate_config else None,
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
        erg_data = {}
        for i, audio_path in enumerate(tqdm(df[self.config.audio_column_name][s:e])):
            sample_id = df["id"][s + i]
            erg = process_one(audio_path, self.config.sampling_rate)
            erg_data[sample_id] = erg
        print(f"finished processing {len(erg_data)} utterances ({s}-{e})")

        erg_path = (
            Path(self.config.output_folder)
            / "erg"
            / f"erg_{shard_id}_{self.config.nshards}.pt"
        )
        os.makedirs(erg_path.parent, exist_ok=True)
        torch.save(erg_data, erg_path)
        print(f"saved to {erg_path}")
        return erg_path

    @staticmethod
    def version() -> str:
        return "0.1"
