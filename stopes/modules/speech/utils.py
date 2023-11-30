# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Different helper functions to interface with fairseq
#

import functools
import io
import logging
import typing as tp
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import librosa
import torch
import torchaudio

if tp.TYPE_CHECKING:
    import torch

log = logging.getLogger("stopes.speech.utils")


@functools.lru_cache(10)
def warn_once(msg: str) -> None:
    """Prevents flooding stderr with the same repeated error message."""
    log.warning(msg)


class IntersectMethods(Enum):
    """Method to compute the overlap of two segments:
    - FRACTION: uses overlap time / max(samples duration)
    - IOU: intersection over union of the two samples
    """

    FRACTION = 0
    IOU = 1


class Text(tp.NamedTuple):
    content: str

    def __str__(self) -> str:
        return self.content


class Audio(tp.NamedTuple):
    """
    Audio file is assumed to be sampled at 16kHz. The default
    is to assume that timestamps are given in milliseconds.
    Should the input be given in wav frames, the sampling factor should
    be set to 16.

    Args:
        the path of the audio where segment is found
        start: start frame
        end: end frame
        sampling_factor: Factor of sampling rate / 1000
        sep: separator, used to represent the audio segment in string format
    """

    path: str
    start: int
    end: int
    sampling_factor: int = 16
    sep: str = "|"

    @property
    def duration(self) -> float:
        """Returns the duration of a segment in ms"""
        return float(self.end - self.start) / float(self.sampling_factor)

    def audio_segment_to_string(self) -> str:
        # TODO: does it make sense to preserve the format ?
        # Instead should we always convert to our preferred format ?
        start = self.start if self.sep == " " else self.start // self.sampling_factor
        end = self.end if self.sep == " " else self.end // self.sampling_factor
        return self.sep.join(
            [
                self.path,
                str(start),
                str(end),
                str(self.sampling_factor),
            ]
        )

    def __str__(self) -> str:
        # method to be coherent with text,
        # but less descriptive
        return self.audio_segment_to_string()

    def load(self, average_channels: bool = False) -> "torch.Tensor":
        num_frames = self.end - self.start if self.end > 0 else None
        audio, sample_rate = torchaudio.load(self.path, self.start, num_frames)
        if sample_rate != self.sampling_factor * 1000:
            warn_once(
                f"Audio has sample rate {sample_rate}. "
                f"Resample to {self.sampling_factor * 1000}"
            )
            audio = torchaudio.functional.resample(
                audio, sample_rate, self.sampling_factor * 1000
            )
        if average_channels and audio.ndim > 1 and audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        return audio  # type: ignore[no-any-return]


# TODO: there should only be one "Audio" class
# that provide minimal API: "load", "to_str", "duration"
class AudioBytes(tp.NamedTuple):
    path: str
    byte_offset: int
    length: int
    # Note: with this data format, the sample rate isn't encoded in the string
    # it has to be specified by user.
    sample_rate: int

    def audio_segment_to_string(self) -> str:
        return ":".join((self.path, str(self.byte_offset), str(self.length)))

    def __str__(self) -> str:
        return self.audio_segment_to_string()

    def load(self, average_channels: bool = False) -> "torch.Tensor":
        with open(self.path, "rb") as f:
            f.seek(self.byte_offset)
            audio_bytes = f.read(self.length)

        if len(audio_bytes) != self.length:
            raise RuntimeError(
                f"Expected to read {self.length} bytes from {self.path} at offest"
                f" {self.byte_offset}, only read {len(audio_bytes)}"
            )

        wav, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
        if sample_rate != self.sample_rate:
            warn_once(
                f"Audio has sample rate {sample_rate}. Resample to {self.sample_rate}"
            )
            wav = torchaudio.functional.resample(wav, sample_rate, self.sample_rate)
        if average_channels and wav.ndim > 1 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav  # type: ignore[no-any-return]


@dataclass
class MiningLineResult:
    score: float
    src: tp.Union[Text, Audio, AudioBytes]
    tgt: tp.Union[Text, Audio, AudioBytes]


@dataclass
class LineResult:
    columns: tp.List[tp.Union[float, Text, Audio, AudioBytes]]


def parse_audio(
    audio: str,
    sampling_factor: tp.Optional[int] = None,
) -> tp.Union[Audio, AudioBytes]:
    """
    Expected audio formats (by priority):
        1. full audio file: <file>
        2. offset slice in bytes: <file>:<offset>:<bytes length>
        3. time slice in ms: <file>|<ts_start_ms>|<ts_end_ms>|<sampling_factor,optional>
        4. frame slice: <file> <frame_start> <frame_end> <to_be_ignored,optional>
    """
    if Path(audio).exists():
        audio_info = torchaudio.info(audio)
        if sampling_factor is None:
            sampling_factor = audio_info.sample_rate / 1000
        return Audio(audio, 0, audio_info.num_frames, sampling_factor)

    if audio.count(":") == 2:
        # Fairseq data format: file + byte offset + byte length
        path, byte_offset, length = audio.split(":")
        if sampling_factor is None:
            raise ValueError(
                "No default sampling factor provided, and not read from file"
            )
        return AudioBytes(path, int(byte_offset), int(length), sampling_factor * 1000)

    if audio.count("|") in (2, 3):
        try:
            path, start_ms, end_ms, *_optional_sampling_factor = audio.split("|")
            if _optional_sampling_factor:
                sampling_factor = int(_optional_sampling_factor[0])
            elif sampling_factor is None:
                # Not using a ValueError, since here we are confident about the parsing,
                # but the sampling rate isn't specified.
                raise RuntimeError(
                    "No default sampling factor provided, and not read from file"
                )
            else:
                warn_once("Sampling factor not present in file, using provided value.")

            start = int(start_ms) * sampling_factor  # type: ignore[assignment]
            end = int(end_ms) * sampling_factor  # type: ignore[assignment]
            return Audio(
                path=path,
                start=start,
                end=end,
                sampling_factor=sampling_factor,
                sep="|",
            )
        except ValueError:
            pass

    if audio.count(" ") in (2, 3):
        try:
            path, frame_start, frame_end, *_ = audio.split(" ")
            warn_once("Sampling factor is assumed to be 16 for space-split text")
            start, end = int(frame_start), int(frame_end)
            return Audio(path=path, start=start, end=end, sampling_factor=16, sep=" ")
        except ValueError:
            pass

    raise ValueError(f"Failed to parse audio segment from {audio}")


def parse_audio_or_text(
    annotation: str, sampling_factor: tp.Optional[int] = None
) -> tp.Union[Text, Audio, AudioBytes]:
    try:
        return parse_audio(annotation, sampling_factor)
    except ValueError:
        return Text(content=annotation)


def auto_parse(
    annotation: str, sampling_factor: tp.Optional[int] = None
) -> tp.Union[float, Text, Audio, AudioBytes]:
    try:
        return parse_audio(annotation, sampling_factor)
    except ValueError:
        pass

    try:
        return float(annotation)
    except ValueError:
        return Text(content=annotation)


def split_mining_line(
    line: str, sampling_factor: tp.Optional[int] = None
) -> MiningLineResult:
    score, src, tgt = line.split("\t")
    return MiningLineResult(
        score=float(score),
        src=parse_audio_or_text(src, sampling_factor=sampling_factor),
        tgt=parse_audio_or_text(tgt, sampling_factor=sampling_factor),
    )


def auto_parse_line(line: str, sampling_factor: tp.Optional[int] = None) -> LineResult:
    columns = line.split("\t")
    return LineResult(
        columns=[
            auto_parse(column, sampling_factor)
            if i == 0
            else parse_audio_or_text(column, sampling_factor)
            for (i, column) in enumerate(columns)
        ]
    )


def compute_overlap(
    segment1: Audio,
    segment2: Audio,
    method: IntersectMethods = IntersectMethods.FRACTION,
) -> float:
    overlap = min(segment1.end, segment2.end) - max(segment1.start, segment2.start)
    if overlap <= 0:
        return 0
    # convert to ms to be coherent with duration
    overlap_ms = overlap / float(segment1.sampling_factor)
    duration1, duration2 = segment1.duration, segment2.duration

    if method == IntersectMethods.FRACTION:
        # returns the fraction that is the least significant
        # among the two durations
        return overlap_ms / max(duration1, duration2)
    elif method == IntersectMethods.IOU:
        total_time = duration1 + duration2
        union_time = total_time - overlap_ms
        return overlap_ms / union_time
    else:
        raise ValueError("unknown overlap method calculation")


def parse_audio_deprecated(
    line: str,
) -> tp.Tuple[str, tp.Optional[int], tp.Optional[int], tp.Optional[int]]:
    # TODO: Replace calls to `parse_audio_deprecated` to `parse_audio`
    # `parse_audio` also needs to handle 'file f_start f_end'
    line_words = iter(line.split(" "))
    infile = next(line_words)
    f_start = None
    f_end = None
    batch_no = None
    try:
        f_start = int(next(line_words))
        f_end = int(next(line_words))
        batch_no = int(next(line_words))
    except StopIteration:
        pass

    if f_start is not None:
        if f_end is None:
            raise ValueError("Invalid format.")
        elif f_end != -1 and f_end < f_start:
            raise ValueError("End frame value must be greater than start frame.")

    return (infile, f_start, f_end, batch_no)


@functools.lru_cache(10)
def read_audio(
    path: str,
    sampling_rate: int = 16000,
    start_frame: tp.Optional[int] = None,
    end_frame: tp.Optional[int] = None,
) -> "torch.Tensor":
    """
    Load an audio file from disk, resample to the target rate, and return a 1d tensor.
    """
    # TODO: this logic overlaps with Audio.load; we might want to merge them.
    if path.endswith(".mp3"):
        torchaudio.set_audio_backend("sox_io")
    if start_frame is not None and end_frame is not None and start_frame < end_frame:
        # if end is unknown, start_frame = end_frame = 0, thus the last condition
        wav, sr = torchaudio.load(
            path, frame_offset=start_frame, num_frames=end_frame - start_frame
        )
    else:
        if any(map(path.endswith, (".webm", ".flac", ".m4a"))):
            wav, sr = librosa.load(path, sr=sampling_rate)
            wav = torch.from_numpy(wav).unsqueeze(0)
        else:
            wav, sr = torchaudio.load(path)

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != sampling_rate:
        warn_once(f"Audio has sample rate {sr}. Resample to {sampling_rate}")
        transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)
        wav = transform(wav)
        sr = sampling_rate

    return wav.squeeze(0)  # type: ignore[no-any-return]
