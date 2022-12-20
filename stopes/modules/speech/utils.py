# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import typing as tp
import warnings
from dataclasses import dataclass
from enum import Enum


class IntersectMethods(Enum):
    """Method to compute the overlap of two segments:
    - fraction: uses overlap time / max(samples duration)
    - iou: intersection over union of the two samples
    """

    fraction = 0
    iou = 1


class Text:
    def __init__(self, content: str):
        self.content = content
        self.kind = "text"

    def __str__(self):
        return self.content

    def __eq__(self, other):
        if not isinstance(other, Text):
            return NotImplemented
        return self.content == other.content


class Audio:
    def __init__(
        self,
        path: str,
        start: int,
        end: int,
        sampling_factor: int = 16,
        sep="|",
    ):
        """
        Audio file is assumed to be sampled at 16kHz. The default
        is to assume that timestamps are given in milliseconds.
        Should the input be given in wav frames, the sampling factor should
        be set to 16.

        path (str): path to file
        start (int): beginning of the segment
        end (int): end of the segment
        sampling_factor (int): sampling factor to ease
            the conversion between wav frames and ms
        sep (str): sep used in the file,
            used to export to string
        """

        self.path = path
        self.kind = "audio"
        self.start = start
        self.end = end
        self.sampling_factor = sampling_factor
        self.duration = None
        self.sep = sep

    def get_duration(self) -> float:
        """Returns the duration of a segment in ms"""
        if self.duration is None:
            duration = float(self.end - self.start) / float(self.sampling_factor)
            # memorize value for future calls
            self.duration = duration
        return self.duration

    def __eq__(self, other):
        if not isinstance(other, Audio):
            return NotImplemented
        return all(
            [
                self.path == other.path,
                self.start == other.start,
                self.end == other.end,
                self.sampling_factor == other.sampling_factor,
            ]
        )

    def audio_segment_to_string(self) -> str:
        string = self.sep.join(
            [self.path, str(self.start), str(self.end), str(self.sampling_factor)]
        )
        return string

    def __str__(self) -> str:
        # method to be coherent with text,
        # but less descriptive
        return self.audio_segment_to_string()


@dataclass
class MiningLineResult:
    score: float
    src: tp.Union[Text, Audio]
    tgt: tp.Union[Text, Audio]


def extract_info(
    annotation: str, sampling_factor: tp.Optional[int] = None, raise_warning=False
) -> tp.Union[Text, Audio]:
    # Expected audio formats
    # format 1: <file> <ts_start> <ts_end> <to_be_ignored> -> wav cycle timestamps
    # format 2: <file>|<ts_start_ms>|<ts_end_ms>|<sampling_factor,optional> -> ms timestamps
    # TODO: update this methods once output formats from speech mining are unified
    if annotation.count("|") == 2:
        path, start, end = annotation.split("|")
        start, end = int(start), int(end)
        if sampling_factor is not None:
            if raise_warning:
                warnings.warn(
                    Warning(
                        "Sampling factor not present in file, using provided value."
                    )
                )
            return Audio(
                path=path,
                start=int(start),
                end=int(end),
                sampling_factor=sampling_factor,
                sep="|",
            )
        else:
            raise ValueError(
                "No default sampling factor provided, and not read from file"
            )
    elif annotation.count("|") == 3:
        path, start, end, sampling_factor_read = annotation.split("|")
        start, end, sampling_factor = int(start), int(end), int(sampling_factor_read)
        return Audio(
            path=path,
            start=int(start),
            end=int(end),
            sampling_factor=sampling_factor,
            sep="|",
        )
    try:
        path, start, end, _ = annotation.split(" ")
        start, end = int(start), int(end)
        if raise_warning:
            warnings.warn(
                Warning("Sampling factor is assumed to be 16 for space-split text")
            )
        return Audio(
            path=path, start=int(start), end=int(end), sampling_factor=16, sep=" "
        )
    except ValueError:
        return Text(content=annotation)


def split_line(
    line: str,
    sampling_factor: tp.Optional[int] = None,
    raise_warning: bool = False,
) -> MiningLineResult:
    score, src, tgt = line.split("\t")
    return MiningLineResult(
        score=float(score),
        src=extract_info(
            src, sampling_factor=sampling_factor, raise_warning=raise_warning
        ),
        tgt=extract_info(
            tgt, sampling_factor=sampling_factor, raise_warning=raise_warning
        ),
    )


def compute_overlap(
    segment1: Audio,
    segment2: Audio,
    method: IntersectMethods = IntersectMethods.fraction,
):
    overlap = min(segment1.end, segment2.end) - max(segment1.start, segment2.start)
    if overlap <= 0:
        return 0
    # convert to ms to be coherent with duration
    overlap = overlap / float(segment1.sampling_factor)
    duration1, duration2 = segment1.get_duration(), segment2.get_duration()

    if method == IntersectMethods.fraction:
        # returns the fraction that is the least significant
        # among the two durations
        return overlap / max(duration1, duration2)
    elif method == IntersectMethods.iou:
        total_time = duration1 + duration2
        union_time = total_time - overlap
        return overlap / union_time
    else:
        raise ValueError("unknown overlap method calculation")
