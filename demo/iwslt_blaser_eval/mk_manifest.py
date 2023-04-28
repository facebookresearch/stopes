#! /bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import math
import multiprocessing
import typing as tp
import wave
from pathlib import Path

import func_argparse


def audio_len(file: Path) -> tp.Tuple[Path, int]:
    with contextlib.closing(wave.open(str(file), "r")) as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = int(math.ceil(frames * 1000 / rate))
        return (file, duration)


def manifest(input_dir: Path, num_cpu: int = 4):
    print(str(input_dir.absolute()))
    with multiprocessing.Pool(processes=num_cpu) as pool:
        for fname, duration in pool.imap(audio_len, input_dir.glob("*.wav")):
            print(fname.name, duration, sep="\t")


if __name__ == "__main__":
    func_argparse.main()
