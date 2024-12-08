# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import typing as tp
from pathlib import Path


def _safe_readline(fd) -> str:
    pos = fd.tell()
    while True:
        try:
            return fd.readline()
        except UnicodeDecodeError:
            pos -= 1
            fd.seek(pos)  # search where this character begins


def find_offsets(filename: tp.Union[str, Path], num_chunks: int) -> tp.List[int]:
    """
    given a file and a number of chuncks, find the offsets in the file
    to be able to chunk around full lines.
    """
    with open(filename, "r", encoding="utf-8") as f:
        size = os.fstat(f.fileno()).st_size
        chunk_size = size // num_chunks
        offsets = [0 for _ in range(num_chunks + 1)]
        for i in range(1, num_chunks):
            f.seek(chunk_size * i)
            _safe_readline(f)
            offsets[i] = f.tell()
        offsets[-1] = size
        return offsets


def find_offsets_of_lines(
    filename: tp.Union[str, Path], num_chunks: int, nrows: int
) -> tp.List[int]:
    """
    Find the offsets of a text file that makes a total of `num_chunks` roughly equal-size chunks.
    Here only the first `nrows` lines are read. This function should be used when `nrows` is
    relatively small compared to the size of `filename`.
    To find offsets of the entire file, please use `stopes.utils.file_chunker_utils.find_offsets()`
    """
    offsets = []
    r = nrows % num_chunks
    chunk_size = nrows // num_chunks
    with open(filename, "r", encoding="utf-8") as f:
        # Each of the r first chunks has one more line than the rest num_chunks - r
        size = chunk_size + 1
        for _ in range(r):
            offsets.append(f.tell())
            [f.readline() for _ in range(size)]

        for _ in range(0, num_chunks - r):
            offsets.append(f.tell())
            [f.readline() for _ in range(chunk_size)]

        offsets.append(f.tell())

    return offsets


def find_line_numbers(
    filename: tp.Union[str, Path], start_offsets: tp.List[int]
) -> tp.List[int]:
    """
    given a file and a number of start byte offsets, return the numbers of the
    lines that would correspond to the beginning of the chunk, i.e. would
    read through those offsets
    """
    line_cnt = 0
    offset_idx = 0
    line_nos = []
    with open(filename, "r", encoding="utf-8") as f:
        while offset_idx < len(start_offsets):
            f.readline()
            if start_offsets[offset_idx] <= f.tell():
                line_nos.append(line_cnt)
                offset_idx += 1
            line_cnt += 1
    return line_nos


class ChunkLineIterator:
    """
    Iterator to properly iterate over lines of a file chunck.
    """

    def __init__(self, fd, start_offset: int, end_offset: tp.Optional[int] = None):
        self._fd = fd
        self._start_offset = start_offset
        self._end_offset = end_offset

    def __iter__(self) -> tp.Iterable[str]:
        self._fd.seek(self._start_offset)
        # next(f) breaks f.tell(), hence readline() must be used
        line = _safe_readline(self._fd)
        while line:
            pos = self._fd.tell()
            # f.tell() does not always give the byte position in the file
            # sometimes it skips to a very large number
            # it is unlikely that through a normal read we go from
            # end bytes to end + 2**32 bytes (4 GB) and this makes it unlikely
            # that the procedure breaks by the undeterministic behavior of
            # f.tell()
            if (
                self._end_offset is not None
                and self._end_offset > 0
                and pos > self._end_offset
                and pos < self._end_offset + 2**32
            ):
                break
            yield line
            line = self._fd.readline()


class Chunker:
    """
    contextmanager to read a chunck of a file line by line.
    """

    def __init__(
        self, path: str, start_offset: int, end_offset: tp.Optional[int] = None
    ):
        self.path = path
        self.start_offset = start_offset
        self.end_offset = end_offset

    def __enter__(self) -> ChunkLineIterator:
        self.fd = open(self.path, "r", encoding="utf-8")
        return ChunkLineIterator(self.fd, self.start_offset, self.end_offset)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.fd.close()
