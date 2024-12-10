# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import logging
import re
import tempfile
import typing as tp
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torchaudio
from torchaudio.backend.common import AudioMetaData

import stopes.modules.speech.postprocess as sprocess
import stopes.modules.speech.utils as sputils
from stopes.modules.speech.audio_load_utils import load_audio
from stopes.modules.speech.utils import Audio, AudioBytes, Text


@pytest.fixture(scope="module")
def sample_audio():
    with contextlib.ExitStack() as stack:
        tmpdir = stack.enter_context(tempfile.TemporaryDirectory())
        lll_data = torchaudio.datasets.LibriLightLimited(tmpdir, download="True")[0]

        audio_path = tmpdir + "/tmp.ogg"
        torchaudio.save(audio_path, lll_data[0], lll_data[1], format="ogg")
        num_frames = torchaudio.info(tmpdir + "/tmp.ogg").num_frames

        audio_zip_path = tmpdir + "/tmp.zip"
        zip_o = stack.enter_context(zipfile.ZipFile(audio_zip_path, "w"))
        with zip_o.open("tmp.ogg", mode="w") as sample_o:
            sample_start = int(sample_o._fileobj.tell())  # type: ignore[attr-defined]
            torchaudio.save(sample_o, lll_data[0], lll_data[1], format="ogg")
            sample_end = int(sample_o._fileobj.tell())  # type: ignore[attr-defined]

        yield (audio_path, num_frames), (audio_zip_path, sample_start, sample_end)


def test_speech_utils_parse_audio(caplog: pytest.LogCaptureFixture) -> None:
    """
    Expected audio formats (by priority):
        1. full audio file: <file>
        2. time slice: <file>|<ts_start_ms>|<ts_end_ms>|<sampling_factor,optional>
        3. frame slice: <file> <ts_start> <ts_end> <to_be_ignored,optional>
    """
    # Invalid samples
    with pytest.raises(ValueError):
        # Too many parts
        sputils.parse_audio("path 10 15 20 30 40")
    with pytest.raises(ValueError):
        # mixed space and pipe
        sputils.parse_audio("path 10|15 20")
    with pytest.raises(ValueError):
        # start_not_int
        sputils.parse_audio("path start 10 15")

    # Pipe separated (ms)
    with pytest.raises(RuntimeError):
        # no sample rate specified nor supplied
        sputils.parse_audio("path|10|15")
    with assert_warns(
        caplog, match="Sampling factor not present in file, using provided value"
    ):
        assert sputils.parse_audio("path|10|15", sampling_factor=48) == Audio(
            "path", 480, 720, 48, sep="|"
        )
    assert sputils.parse_audio("path|10|15|48") == Audio("path", 480, 720, 48, sep="|")

    # Space separated (frames)
    with assert_warns(
        caplog, match="Sampling factor is assumed to be 16 for space-split text"
    ):
        assert sputils.parse_audio("path 10 15") == Audio("path", 10, 15, 16, sep=" ")
    with assert_warns(
        caplog, match="Sampling factor is assumed to be 16 for space-split text"
    ):
        assert sputils.parse_audio("path 10 15", 48) == Audio(
            "path", 10, 15, 16, sep=" "
        )
    with assert_warns(
        caplog, match="Sampling factor is assumed to be 16 for space-split text"
    ):
        # with space separated, the 4-th column is a "batch id" not a sample rate
        assert sputils.parse_audio("path 10 15 32") == Audio(
            "path", 10, 15, 16, sep=" "
        )
    with assert_warns(
        caplog, match="Sampling factor is assumed to be 16 for space-split text"
    ):
        assert sputils.parse_audio("path 10 15 32", 48) == Audio(
            "path", 10, 15, 16, sep=" "
        )

    # The warning is not outputted if you try a second time.
    assert caplog.messages == []
    assert sputils.parse_audio("path 10 15") == Audio("path", 10, 15, 16, sep=" ")
    assert caplog.messages == []


@pytest.mark.parametrize(
    "sampling_factor,expected_output",
    [
        (None, Audio("path", 0, 43008, 16, sep="|")),
        (8, Audio("path", 0, 43008, 8, sep="|")),
    ],
)
def test_speech_utils_parse_audio_when_audio_path_exists(
    sampling_factor, expected_output
):
    with patch("pathlib.Path.exists", return_value=True):
        with patch(
            "torchaudio.info",
            return_value=AudioMetaData(
                sample_rate=16000,
                num_frames=43008,
                num_channels=1,
                bits_per_sample=32,
                encoding="PCM_F",
            ),
        ):
            assert (
                sputils.parse_audio("path", sampling_factor=sampling_factor)
                == expected_output
            )


@pytest.mark.xfail()
def test_audio_duration() -> None:
    audio_frames = Audio("path", 16_000, 32_000, 16, sep=" ")
    audio_ms = Audio("path", 1000, 2000, 16, sep="|")
    # TODO: duration isn't correctly computed for Audio using ms as unit
    assert audio_frames.duration == audio_ms.duration


def test_speech_utils_parse_audio_or_text(caplog) -> None:
    # format 1: <file> <ts_start> <ts_end> <to_be_ignored> -> wav sampling
    # format 2: <file>|<ts_start_ms>|<ts_end_ms> -> ms sampling
    start = 10
    end = 15

    too_many_spaces = Text("path 10 15 20 30 40")
    too_few_spaces = Text("path 10")
    pipe_space_mixed = Text("path 10|15 20")
    start_not_int = Text("path start 10 15")
    audio_wav = Audio(path="path", start=start, end=end, sampling_factor=16, sep=" ")
    audio_ms = Audio(path="path", start=start, end=end, sampling_factor=1, sep="|")

    wav_string = str(audio_wav)
    ms_string = str(audio_ms)
    texts = [too_many_spaces, too_few_spaces, pipe_space_mixed, start_not_int]
    for text in texts:
        assert sputils.parse_audio_or_text(text.content) == text

    with pytest.raises(RuntimeError):
        sputils.parse_audio_or_text("path|10|15")

    with assert_warns(
        caplog, match="Sampling factor not present in file, using provided value"
    ):
        sputils.parse_audio_or_text("path|10|15", sampling_factor=1)

    assert sputils.parse_audio_or_text(wav_string, sampling_factor=1) == audio_wav
    assert audio_wav.duration == audio_ms.duration / 16
    assert sputils.parse_audio_or_text(ms_string) == audio_ms


def test_speech_utils_convert_to_string() -> None:
    part1 = "path 5 10 16"
    part2 = "path|100|120|16"
    input_string = f"1.5\t{part1}\t{part2}"
    mined_result = sputils.split_mining_line(input_string)

    rev_part_1 = str(mined_result.src)
    rev_part_2 = str(mined_result.tgt)
    assert f"{mined_result.score}\t{rev_part_1}\t{rev_part_2}" == input_string


def test_compute_overlap() -> None:
    audio_1 = Audio("", 0, 10)
    audio_2 = Audio("", 15, 25)
    audio_3 = Audio("", 5, 17)
    assert sputils.compute_overlap(audio_1, audio_2) == 0
    assert sputils.compute_overlap(audio_1, audio_2, sputils.IntersectMethods.IOU) == 0
    assert round(sputils.compute_overlap(audio_1, audio_3), 3) == round(5.0 / 12.0, 3)
    assert round(
        sputils.compute_overlap(audio_1, audio_3, sputils.IntersectMethods.IOU), 3
    ) == round(5.0 / 17.0, 3)


def test_postprocess() -> None:
    processor = sprocess.PostProcessAudioModule(
        sprocess.PostProcessAudioConfig(
            output_dir=Path("/"),
            output_filename="test.tsv.gz",
            mining_result_path=Path("/"),
            min_audio_length=2,
            mining_threshold=1,
            max_overlap=0.1,
        )
    )
    # check filters on length and score
    audio_path = "/path/to/example"
    # valid:
    assert processor.line_passes_thresholds(
        sputils.MiningLineResult(
            score=1.1,
            src=Audio(audio_path, start=5, end=10, sampling_factor=1, sep=" "),
            tgt=Text("a"),
        )
    )

    # score too low:
    assert not processor.line_passes_thresholds(
        sputils.MiningLineResult(
            score=0.9,
            src=Audio(path=audio_path, start=-100, end=0, sampling_factor=1, sep=" "),
            tgt=Text("b"),
        )
    )
    # duration too small
    assert not processor.line_passes_thresholds(
        sputils.MiningLineResult(
            1.1,
            Audio(audio_path, -1, 0, sampling_factor=1, sep=" "),
            Text("c"),
        )
    )

    # check accurate removal of overlapping segments
    sources = {
        audio_path: [
            sputils.MiningLineResult(
                1.1,
                Audio(audio_path, 5, 10, sampling_factor=1, sep=" "),
                Text("a"),
            ),
            sputils.MiningLineResult(
                1.2,
                Audio(audio_path, 6, 10, sampling_factor=1, sep=" "),
                Text("b"),
            ),  # score too low
            sputils.MiningLineResult(
                1.1,
                Audio(audio_path, 6, 11, sampling_factor=1, sep=" "),
                Text("c"),
            ),  # too small fragment
        ]
    }
    filtered, _ = processor.postprocess(sources=sources)
    assert len(filtered) == 1
    assert filtered[0].tgt.content == "b"  # type: ignore


def test_parse_audio_deprecated():
    assert sputils.parse_audio_deprecated("file.mp3") == ("file.mp3", None, None, None)
    with pytest.raises(ValueError):
        sputils.parse_audio_deprecated("file.mp3 21")

    assert sputils.parse_audio_deprecated("file.mp3 21 421") == (
        "file.mp3",
        21,
        421,
        None,
    )
    assert sputils.parse_audio_deprecated("file.mp3 21 421 382") == (
        "file.mp3",
        21,
        421,
        382,
    )
    assert sputils.parse_audio_deprecated("file.mp3 21 421 382 dummy") == (
        "file.mp3",
        21,
        421,
        382,
    )

    with pytest.raises(ValueError):
        sputils.parse_audio_deprecated("file.mp3 21 text")
    with pytest.raises(ValueError):
        sputils.parse_audio_deprecated("file.mp3 21 20")
    assert sputils.parse_audio_deprecated("file.mp3 21 -1") == (
        "file.mp3",
        21,
        -1,
        None,
    )


def test_parse_audio_can_replace_deprecated(tmp_path: Path):
    assert sputils.parse_audio_deprecated("file.mp3") == ("file.mp3", None, None, None)

    with pytest.raises(ValueError):
        sputils.parse_audio_deprecated("file.mp3 21")

    assert sputils.parse_audio_deprecated("file.mp3 21 421") == (
        "file.mp3",
        21,
        421,
        None,
    )
    assert sputils.parse_audio_deprecated("file.mp3 21 421 382") == (
        "file.mp3",
        21,
        421,
        382,
    )
    assert sputils.parse_audio_deprecated("file.mp3 21 421 382 dummy") == (
        "file.mp3",
        21,
        421,
        382,
    )

    with pytest.raises(ValueError):
        sputils.parse_audio_deprecated("file.mp3 21 text")
    with pytest.raises(ValueError):
        sputils.parse_audio_deprecated("file.mp3 21 20")
    assert sputils.parse_audio_deprecated("file.mp3 21 -1") == (
        "file.mp3",
        21,
        -1,
        None,
    )


def test_parse_audio_with_resample(sample_audio, caplog):
    audio_path, num_frames = sample_audio[0]
    start_ms, end_ms = 0, int(num_frames / 32)
    audio_text = f"{audio_path}|{start_ms}|{end_ms}"
    auto_meta = sputils.parse_audio_or_text(audio_text, sampling_factor=32)
    assert isinstance(auto_meta, Audio)

    with assert_warns(caplog, match="Audio has sample rate 16000. Resample to 32000"):
        s = auto_meta.load()
        assert s.dim() == 2
        assert s.shape[1] == 351808


def test_parse_audio_bytes_with_resample(sample_audio, caplog):
    audio_zip_path, start, end = sample_audio[1]
    audio_text = f"{audio_zip_path}:{start}:{end-start}"
    auto_meta = sputils.parse_audio_or_text(audio_text, sampling_factor=32)
    assert isinstance(auto_meta, AudioBytes)

    with assert_warns(caplog, match="Audio has sample rate 16000. Resample to 32000"):
        s = auto_meta.load()
        assert s.dim() == 2
        assert s.shape[1] == 351840


@pytest.mark.parametrize("gpu", [True, False])
@pytest.mark.parametrize("fp16", [True, False])
def test_load_audio_in_devices(sample_audio, gpu, fp16, caplog):
    audio_path, num_frames = sample_audio[0]
    line = f"{audio_path}|0|{num_frames}|16\tcoluimn2"
    load_res = load_audio(0, gpu, fp16, line, sampling_factor=32)  # type: ignore[arg-type]
    assert load_res[0] == line
    line_no_sample = f"{audio_path}|0|{num_frames}\tcoluimn2"
    with assert_warns(
        caplog, match="Sampling factor not present in file, using provided value."
    ):
        assert load_audio(0, gpu, fp16, line_no_sample)[0] == line_no_sample


@contextlib.contextmanager
def assert_warns(caplog, *, match: str) -> tp.Iterator[None]:
    caplog.clear()
    sputils.warn_once.cache_clear()

    with caplog.at_level(logging.WARN):
        yield
        assert len(caplog.messages) == 1
        assert re.match(match, caplog.messages[0])
        caplog.clear()


def test_read_audio(tmp_path: Path):
    """Testing that read_audio returns a waveform of the correct shape"""
    wav_path = str(tmp_path / "wav_old.wav")
    torch.manual_seed(0)

    # Creating a 3-second bi-channel waveform with non-standard sr and saving as .wav
    dur_old = 3
    sr_old = 48000
    wav_old = torch.randn([2, dur_old * sr_old])
    torchaudio.save(wav_path, wav_old, sample_rate=sr_old)

    # loading the audio with utils
    sr_new = 16000
    wav_new = sputils.read_audio(wav_path, sampling_rate=sr_new)
    assert (
        len(wav_new.shape) == 1
    ), f"The loaded wave should be 1D tensor, but it has shape {wav_new.shape}"
    dur_new = wav_new.shape[-1] / sr_new
    assert (
        dur_new == dur_old
    ), f"The wav had duration of {dur_old} sec. but was loaded having {dur_new} sec."
