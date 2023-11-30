# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
from tqdm.contrib.concurrent import process_map

try:
    import webrtcvad
except ImportError:
    raise ImportError("Please install py-webrtcvad: pip install webrtcvad")
import os

from .webrtc_vad import apply_vad as _apply_vad
from .webrtc_vad import frame_generator, vad_collector


def multiprocess_map(
    a_list: list,
    func: Callable,
    n_workers: Optional[int] = None,
    chunksize: int = 1,
    desc=None,
):
    if n_workers is None:
        n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))
    n_workers = min(n_workers, cpu_count())
    return process_map(
        func, a_list, max_workers=n_workers, chunksize=chunksize, desc=desc
    )


def _convert_to_16bit_mono_audio(
    in_path_and_out_path_and_sample_rate_and_overwrite: Tuple[Path, Path, int, bool]
) -> None:
    (
        in_path,
        out_path,
        tgt_sample_rate,
        overwrite,
    ) = in_path_and_out_path_and_sample_rate_and_overwrite
    """
    Default SoX backend via TorchAudio
    """
    if out_path.exists() and not overwrite:
        return

    wav, sr = torchaudio.load(in_path.as_posix())
    effects = []
    if tgt_sample_rate != sr:
        effects.append(["rate", str(tgt_sample_rate)])
    if wav.shape[0] > 1:  # multi-channel
        effects.append(["channels", "1"])
    if len(effects) > 0:
        wav, sr = torchaudio.sox_effects.apply_effects_tensor(wav, sr, effects)
    torchaudio.save(out_path.as_posix(), wav, sr, bits_per_sample=16)


def convert_to_16bit_mono_audio(
    in_and_out_paths: List[Tuple[Path, Path]],
    target_sample_rate: int = 16_000,
    overwrite: bool = False,
) -> None:
    items = [(i, o, target_sample_rate, overwrite) for i, o in in_and_out_paths]
    multiprocess_map(items, _convert_to_16bit_mono_audio)


from functools import wraps

# (audio tensor, sample_rate)
AudioType = NewType("AudioType", Tuple[torch.Tensor, int])
ItemType = NewType("ItemType", Dict)


def pipeline(item, funcs: List[Callable]) -> None:
    return_value = None
    for func in funcs:
        return_value = func(return_value, item)


def pipeline_func_decorator(func_config_name, expand_list=False) -> Callable:
    """
    Decorate the functions that sending into pipeline for multi-process.
    """

    def wrapper(func):
        @wraps(func)
        def inner(audio_and_sr: Union[AudioType, List[AudioType]], item: ItemType):
            func_config = item.get(func_config_name, None)
            if not expand_list:
                return func(audio_and_sr, func_config)
            assert isinstance(
                audio_and_sr, list
            ), f"expand_list is True, but input of {func.__name__} is not list {audio_and_sr}"
            if not isinstance(func_config, list):
                return [
                    func(audio_and_sr_item, func_config)
                    for audio_and_sr_item in audio_and_sr
                ]
            else:
                assert len(audio_and_sr) == len(
                    func_config
                ), f"input list size ({len(audio_and_sr)}) of {func.__name__} does not equal to input params ({len(func_config)})"
                return [
                    func(audio_and_sr_item, func_config_item)
                    for audio_and_sr_item, func_config_item in zip(
                        audio_and_sr, func_config
                    )
                ]

        return inner

    return wrapper


@pipeline_func_decorator("convert_to_mono_audio")
def convert_to_mono_audio(
    audio_and_sr: AudioType,
    tgt_sample_rate: Union[None, int],
) -> Tuple[torch.Tensor, int]:
    """
    Default SoX backend via TorchAudio
    """
    wav, sr = audio_and_sr
    effects = []
    if tgt_sample_rate is not None and tgt_sample_rate != sr:
        effects.append(["rate", str(tgt_sample_rate)])
    if wav.shape[0] > 1:  # multi-channel
        effects.append(["channels", "1"])
    if len(effects) > 0:
        wav, sr = torchaudio.sox_effects.apply_effects_tensor(wav, sr, effects)
    return (wav, sr)


@pipeline_func_decorator("slice_audio")
def slice_audio(
    audio_and_sr: AudioType,
    start_and_duration_secs: Optional[List[Tuple[float, float]]],
) -> List[Tuple[torch.Tensor, int]]:
    if start_and_duration_secs is None:
        return [audio_and_sr]
    audio, sr = audio_and_sr
    results = []
    for start_sec, duration_sec in start_and_duration_secs:
        start_frame = int(start_sec * sr)
        end_frame = int((start_sec + duration_sec) * sr)
        results.append((audio[:, start_frame:end_frame], sr))
    return results


@pipeline_func_decorator("load_audio")
def load_audio(_none: Any, in_path: Path) -> AudioType:
    return torchaudio.load(in_path.as_posix())


@pipeline_func_decorator("write_audio")
def write_audio(
    audio_and_sr: AudioType,
    out_path: Path,
    bits_per_sample=16,
) -> None:
    audio, sr = audio_and_sr
    torchaudio.save(out_path.as_posix(), audio, sr, bits_per_sample=bits_per_sample)


@pipeline_func_decorator("write_audios", expand_list=True)
def write_audios(
    audio_and_sr: AudioType,
    out_path: Path,
    bits_per_sample=16,
) -> None:
    audio, sr = audio_and_sr
    if out_path.suffix in [".ogg"]:
        torchaudio.save(out_path.as_posix(), audio, sr)
    else:
        torchaudio.save(out_path.as_posix(), audio, sr, bits_per_sample=bits_per_sample)


_vad = None


def init_vad(agg: int = 3) -> None:
    global _vad
    # build vad object
    _vad = webrtcvad.Vad(agg)


def init_silero_vad(threshold: float) -> None:
    torch.set_num_threads(1)
    import onnxruntime

    opts = onnxruntime.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    global _vad
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=True,
        onnx=True,
    )
    (get_speech_timestamps, _, _, _, collect_chunks) = utils

    _vad = model, get_speech_timestamps, collect_chunks, threshold


@pipeline_func_decorator("apply_vad_audio_list", expand_list=True)
def apply_vad_audio_list(audio_and_sr: AudioType, _none: Any) -> AudioType:
    global _vad
    audio, sr = audio_and_sr
    return _apply_vad(_vad, audio, sr)


@torch.inference_mode()
@pipeline_func_decorator("apply_silero_vad_audio_list", expand_list=True)
def apply_silero_vad_audio_list(
    audio_and_sr: AudioType, _none: Any
) -> Tuple[torch.Tensor, int]:
    global _vad
    audio, sr = audio_and_sr
    model, get_speech_timestamps, collect_chunks, threshold = _vad  # type: ignore
    for thr in torch.arange(threshold, 0.0, -0.02):
        speech_tss = get_speech_timestamps(
            audio, model, threshold=thr, sampling_rate=sr
        )
        if len(speech_tss) > 0:
            break

    # if VAD failed, we return None here and treat that audio as silence only
    if len(speech_tss) > 0:
        vad_audio = collect_chunks(speech_tss, audio.view(-1)).unsqueeze(0)
    else:
        vad_audio = torch.tensor([[0.0]], device="cpu")

    return (vad_audio, sr)


@pipeline_func_decorator("convert_float_to_pcm_list", expand_list=True)
def convert_float_to_pcm_list(
    audio_and_sr: AudioType, _none: Any, bits_per_sample=16
) -> Tuple[torch.Tensor, int]:
    audio, sr = audio_and_sr
    return (float_to_byte(audio), sr)


@pipeline_func_decorator("convert_pcm_to_float_list", expand_list=True)
def convert_pcm_to_float_list(
    audio_and_sr: AudioType, _none: Any, bits_per_sample=16
) -> Tuple[torch.Tensor, int]:
    audio, sr = audio_and_sr
    return (float_to_tensor(byte_to_float(audio)), sr)
    # return (audio.type(dtype=torch.float) / (2 ** bits_per_sample), sr)


# Borrow from https://gist.github.com/HudsonHuang/fbdf8e9af7993fe2a91620d3fb86a182
def float_to_byte(sig):
    # float32 -> int16(PCM_16) -> byte
    return float2pcm(sig, dtype="int16").tobytes()


def byte_to_float(byte):
    # byte -> int16(PCM_16) -> float32
    return pcm2float(np.frombuffer(byte, dtype=np.int16), dtype="float32")


def float_to_tensor(sig):
    # float32 -> torch.float32
    return torch.Tensor(np.expand_dims(sig, 0))


def pcm2float(sig, dtype="float32"):
    """Convert PCM signal to floating point with a range from -1 to 1.
    Use dtype='float32' for single precision.
    Parameters
    ----------
    sig : array_like
        Input array, must have integral type.
    dtype : data type, optional
        Desired (floating point) data type.
    Returns
    -------
    numpy.ndarray
        Normalized floating point data.
    See Also
    --------
    float2pcm, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind not in "iu":
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != "f":
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max


def float2pcm(sig, dtype="int16"):
    """Convert floating point signal with a range from -1 to 1 to PCM.
    Any signal values outside the interval [-1.0, 1.0) are clipped.
    No dithering is used.
    Note that there are different possibilities for scaling floating
    point numbers to PCM numbers, this function implements just one of
    them.  For an overview of alternatives see
    http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html
    Parameters
    ----------
    sig : array_like
        Input array, must have floating point type.
    dtype : data type, optional
        Desired (integer) data type.
    Returns
    -------
    numpy.ndarray
        Integer data, scaled and clipped to the range of the given
        *dtype*.
    See Also
    --------
    pcm2float, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != "f":
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in "iu":
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


def change_sample_rate(audio_and_sr: Tuple, tgt_sr: int) -> Tuple[torch.Tensor, int]:
    wav, sr = audio_and_sr
    if tgt_sr != sr:
        effects = [["rate", str(tgt_sr)]]
        wav, sr = torchaudio.sox_effects.apply_effects_tensor(wav, sr, effects)
    return (wav, sr)
