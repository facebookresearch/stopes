# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import itertools
import typing as tp
import zipfile
from pathlib import Path

import torchaudio
from fastapi import APIRouter
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse, Response

import stopes.core.utils as stutils
import stopes.modules.speech.utils as stopes_speech
from stopes.modules.speech.utils import LineResult, auto_parse_line

from .query_types import AnnotationQuery, AudioQuery, DefaultQuery, LineQuery

torchaudio.set_audio_backend("sox_io")

router = APIRouter(tags=["fileviewer"])


@router.post("/file_lines/")
async def get_file_lines(query: LineQuery) -> JSONResponse:
    with stutils.open(
        query.gz_path.strip(),
        mode="rt",
        encoding="utf-8",
    ) as f:
        lines = 0
        while f.readline():
            lines += 1
    return JSONResponse({"lines": lines})


def _read_audio(path: str, frame_offset: int, num_frames: int) -> tp.Any:
    if ".zip@" in path:
        file, subfile = path.split(".zip@")
        with zipfile.ZipFile(file + ".zip") as z:
            with z.open(subfile) as f:
                return torchaudio.load(
                    f, frame_offset=frame_offset, num_frames=num_frames
                )

    return torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)


@router.post("/servefile/")
async def serve_file(query: AudioQuery) -> Response:
    """
    Opens an audio file and extracts the relevant segment.

    We use `torchaudio.load` to open several kind of audio formats,
    but always returns '.wav' encoded audio files.
    """
    if query.start > -1:
        window_size = round(query.context_size * 16000)
        if query.sampling == "ms":
            start, end = query.start * 16, query.end * 16
        else:
            start, end = query.start, query.end
        frame_offset = max(0, start - window_size)
        num_frames = end - start + 2 * window_size
    else:
        frame_offset, num_frames = 0, -1

    try:
        audio_segment, sample_rate = _read_audio(query.path, frame_offset, num_frames)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")

    # Convert the audio tensor back into bytes
    audio_bytes = io.BytesIO()
    torchaudio.save(audio_bytes, audio_segment, sample_rate, format="wav")
    return Response(audio_bytes.getvalue(), media_type="audio/wav")


def open_segment_tsv(file: str, start_idx: int, end_idx: int) -> tp.List[LineResult]:
    results = []
    with stutils.open(file) as f:
        for line in itertools.islice(f, start_idx, end_idx):
            results.append(auto_parse_line(line.strip()))
    return results


def open_zip_file(file: str, start_idx: int, end_idx: int) -> tp.List[LineResult]:
    audio_samples = []
    with zipfile.ZipFile(file) as z:
        subfiles = (info for info in z.infolist() if not info.is_dir())
        for info in itertools.islice(subfiles, start_idx, end_idx):
            audio_sample = LineResult(
                columns=[
                    stopes_speech.Audio("@".join((file, info.filename)), -1, 0),
                    stopes_speech.Text(info.filename),
                ]
            )
            audio_samples.append(audio_sample)
    return audio_samples


@router.post("/annotations/")
def get_annotations(
    query: AnnotationQuery,
) -> tp.List[LineResult]:
    path = query.gz_path.strip()
    print(path, query.start_idx, query.end_idx)
    if path.endswith(".tsv.gz") or path.endswith(".tsv"):
        try:
            return open_segment_tsv(path, query.start_idx, query.end_idx)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail="""
                File not found
            """,
            )
    elif path.endswith(".zip"):
        if not Path(path).exists():
            raise HTTPException(status_code=404, detail="File not found")
        return open_zip_file(path, query.start_idx, query.end_idx)
    else:
        raise ValueError("Unknown format")


@router.post("/general/")
async def general_query(query: DefaultQuery) -> Response:
    query_path = query.gz_path.strip()
    print(query_path.split(" "))
    if query_path.endswith(".tsv.gz") or query_path.endswith(".zip"):
        return get_annotations(
            AnnotationQuery(
                gz_path=query.gz_path, start_idx=query.start_idx, end_idx=query.end_idx
            )
        )
    elif len(query_path.split(" ")) == 3:
        path, start, end = query.gz_path.split(" ")
        result = await serve_file(
            AudioQuery(sampling="wav", path=path, start=int(start), end=int(end))
        )
        return result
    elif len(query_path.split("|")) == 3:
        path, start, end = query.gz_path.split("|")
        result = await serve_file(
            AudioQuery(sampling="ms", path=path, start=int(start), end=int(end))
        )
        return result

    else:
        raise HTTPException(
            status_code=500,
            detail="""
        Query cannot be correctly parsed
        Expecting path to a .gz file
        OR
        a line with audio file, start and end timestamps
        """,
        )
