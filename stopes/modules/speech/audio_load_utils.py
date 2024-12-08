# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import functools
import logging
import typing as tp
from typing import Tuple

import numpy as np
import torch

from stopes.modules.speech import utils as speech_utils

log = logging.getLogger(__name__)


@contextlib.contextmanager
def spawn_mp(num_processses: int):
    from torch import multiprocessing

    start_method = multiprocessing.get_start_method()
    multiprocessing.set_start_method("spawn", force=True)
    yield multiprocessing.Pool(num_processses)

    multiprocessing.set_start_method(start_method)


def parallel_audio_read(
    lines: tp.Iterator[str],
    column_offset: int,
    gpu: bool = False,
    fp16: bool = False,
    num_process: tp.Optional[int] = 4,
    chunksize: int = 16,
    sampling_factor: int = 16,
    collapse_channels: bool = False,
    **kwargs,
) -> tp.Iterator[Tuple[str, np.ndarray]]:
    """
    Load audio from a given manifest file, using several processes.
    For each input line,
        * extract audio corresponding to one column
        * optionally move the audio tensor to gpu
        * yield the full line as well as the loaded audio as a numpy array
    """
    with contextlib.ExitStack() as stack:
        if num_process is None or num_process > 1:
            num_process = num_process or 4  # default number of processes
            pool = stack.enter_context(spawn_mp(num_process))
            # chunksize=16, to send segments from the same file to the same worker.
            # Note this optimization only work if the input file is sorted per segment,
            # which in the case of mining only work for the source segments.
            pool_imap = functools.partial(pool.imap, chunksize=chunksize)
        else:
            pool_imap = map  # type: ignore

        _load = functools.partial(
            load_audio,
            column_offset,
            gpu,
            fp16,
            sampling_factor=sampling_factor,
            as_numpy=True,
            # pytorch modifies the multiprocessing behaviour to optimize serializing Tensors.
            # This causes non-deterministic "No space left on device" errors when we return a Tensor.
            # In order to avoid that, we are using numpy arrays as return values.
            collapse_channels=collapse_channels,
            **kwargs,
        )
        yield from pool_imap(_load, lines)


def load_audio(
    column_offset: int,
    gpu: bool,
    fp16: bool,
    line: str,
    sampling_factor: int = 16,
    as_numpy: bool = False,
    collapse_channels: bool = False,
    **kwargs,
) -> tp.Tuple[str, tp.Union[torch.Tensor, np.ndarray]]:
    """
    Load audio from a TSV-line where column at `column_offset` contains audio info

    Args:
        column_offset: which column in `line` that contains the audio information
        gpu: do the resampling in GPU
        fp16: do the resampling with fp16 to save memory (with less accuracy)
        line: input data represented as tab-separated columns
        sampling_factor: unit of sample rate (1 factor = 1 kHz)
    """
    input_line = line.rstrip("\n").split("\t")
    audio_meta = speech_utils.parse_audio(
        input_line[column_offset], sampling_factor=sampling_factor
    )
    if isinstance(audio_meta, speech_utils.Audio):
        # mp3 files need to be fully read before we can extract a segment.
        # But segments are sorted so we should hit several time the same file.
        wav = speech_utils.read_audio(
            audio_meta.path,
            audio_meta.sampling_factor * 1000,
        )
        if len(wav.shape) > 1:
            wav = wav[:, audio_meta.start : audio_meta.end]
        else:
            wav = wav[audio_meta.start : audio_meta.end]
    elif isinstance(audio_meta, speech_utils.AudioBytes):
        wav = audio_meta.load()
    elif isinstance(audio_meta, speech_utils.Text):
        wav = speech_utils.read_audio(
            audio_meta.content,
            sampling_factor * 1000,
            kwargs.get("start_frame", None),
            kwargs.get("end_frame", None),
        )
    if gpu and torch.cuda.is_available():
        wav = wav.cuda()
        if fp16:
            wav = wav.half()
    if collapse_channels and len(wav.shape) > 1:
        wav = wav.mean(0)
    if as_numpy:
        wav = wav.cpu().numpy()
    return (line, wav)
