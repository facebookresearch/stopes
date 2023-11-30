# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import func_argparse
import soundfile
import xxhash
from tqdm import tqdm

from stopes.core.utils import open as stopes_open
from stopes.modules.speech.speech_units import parallel_audio_read


def extract(
    in_tsv_file: Path,
    out_dir: Path,
    column_offset: int,
    num_cpus: int = 10,
    skip_header: bool = True,
    num_lines: tp.Optional[int] = None,
):
    """
    Sometimes you are faced with a tsv manifest, but all you want is a few wavs to listen to or use somewhere else.
    Use this script to just extract the wavs.

    Arguments:
        in_tsv_file: the tsv file to read the manifest from.
        out_dir: the directory where to export the wavs.
        column_offset: the offset of the column to read in the tsv (starts at 0).
        num_cpus: how many cpus to use to read the audio.
        skip_header: is there a header in the tsv to skip.
        num_lines: the maximum number of lines to read (inclusive).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with stopes_open(in_tsv_file) as reader:
        if skip_header:
            next(reader)
        line_cnt = 0
        for line, audio in tqdm(
            parallel_audio_read(
                reader,
                column_offset,
                gpu=False,
                fp16=False,
                num_process=num_cpus,
            ),
            unit="segment",
        ):
            if num_lines is not None and line_cnt > num_lines:
                return
            filename = f"{xxhash.xxh3_64_intdigest(line)}_{column_offset}.wav"
            outfile = out_dir / filename
            soundfile.write(
                outfile,
                audio.transpose(),
                16000,
                "PCM_24",
                format="wav",
            )
            line_cnt += 1


if __name__ == "__main__":
    func_argparse.single_main(extract)
