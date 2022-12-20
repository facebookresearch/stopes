# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path

import pytest

import stopes.modules.speech.postprocess as sprocess
import stopes.modules.speech.utils as sputils


def test_speech_utils_linesplit():
    # format 1: <file> <ts_start> <ts_end> <to_be_ignored> -> wav sampling
    # format 2: <file>|<ts_start_ms>|<ts_end_ms> -> ms sampling
    start = 10
    end = 15

    too_many_spaces = sputils.Text("path 10 15 20 30 40")
    too_few_spaces = sputils.Text("path 10 15")
    pipe_space_mixed = sputils.Text("path 10|15 20")
    start_not_int = sputils.Text("path start 10 15")
    audio_wav = sputils.Audio(
        path="path", start=start, end=end, sampling_factor=16, sep=" "
    )
    audio_ms = sputils.Audio(
        path="path", start=start, end=end, sampling_factor=1, sep="|"
    )
    wav_string = str(audio_wav)
    ms_string = str(audio_ms)
    texts = [too_many_spaces, too_few_spaces, pipe_space_mixed, start_not_int]
    for text in texts:
        assert sputils.extract_info(text.content) == text

    with pytest.raises(ValueError):
        sputils.extract_info("path|10|15")
    with pytest.warns(Warning):
        sputils.extract_info("path|10|15", sampling_factor=1, raise_warning=True)

    assert (
        sputils.extract_info(wav_string, sampling_factor=1, raise_warning=False)
        == audio_wav
    )
    assert audio_wav.get_duration() == audio_ms.get_duration() / 16
    assert sputils.extract_info(ms_string) == audio_ms


def test_speech_utils_convert_to_string():
    part1 = "path 5 10 16"
    part2 = "path|15|20|1"
    input_string = f"1.5\t{part1}\t{part2}"
    mined_result = sputils.split_line(input_string)

    rev_part_1 = str(mined_result.src)
    rev_part_2 = str(mined_result.tgt)
    assert f"{mined_result.score}\t{rev_part_1}\t{rev_part_2}" == input_string


def test_compute_overlap():
    audio_1 = sputils.Audio("", 0, 10)
    audio_2 = sputils.Audio("", 15, 25)
    audio_3 = sputils.Audio("", 5, 17)
    assert sputils.compute_overlap(audio_1, audio_2) == 0
    assert sputils.compute_overlap(audio_1, audio_2, sputils.IntersectMethods.iou) == 0
    assert round(sputils.compute_overlap(audio_1, audio_3), 3) == round(5.0 / 12.0, 3)
    assert round(
        sputils.compute_overlap(audio_1, audio_3, sputils.IntersectMethods.iou), 3
    ) == round(5.0 / 17.0, 3)


def test_postprocess(monkeypatch):
    # create fake process
    def fake_init(self, config: sprocess.PostProcessAudioConfig):
        self.config = config
        self.src_attr = "src"
        self.file_lines = 0
        self.logger = logging.getLogger("test_postprocess")

    # fake init
    monkeypatch.setattr(sprocess.PostProcessAudioModule, "__init__", fake_init)

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
            1.1,
            sputils.Audio(audio_path, start=5, end=10, sampling_factor=1, sep=" "),
            sputils.Text("a"),
        )
    )

    # score too low:
    assert not processor.line_passes_thresholds(
        sputils.MiningLineResult(
            score=0.9,
            src=sputils.Audio(
                path=audio_path, start=-100, end=0, sampling_factor=1, sep=" "
            ),
            tgt=sputils.Text("b"),
        )
    )
    # duration too small
    assert not processor.line_passes_thresholds(
        sputils.MiningLineResult(
            1.1,
            sputils.Audio(audio_path, -1, 0, sampling_factor=1, sep=" "),
            sputils.Text("c"),
        )
    )

    # check accurate removal of overlapping segments
    sources = {
        audio_path: [
            sputils.MiningLineResult(
                1.1,
                sputils.Audio(audio_path, 5, 10, sampling_factor=1, sep=" "),
                sputils.Text("a"),
            ),
            sputils.MiningLineResult(
                1.2,
                sputils.Audio(audio_path, 6, 10, sampling_factor=1, sep=" "),
                sputils.Text("b"),
            ),  # score too low
            sputils.MiningLineResult(
                1.1,
                sputils.Audio(audio_path, 6, 11, sampling_factor=1, sep=" "),
                sputils.Text("c"),
            ),  # too small fragment
        ]
    }
    filtered, _ = processor.postprocess(sources=sources)
    assert len(filtered) == 1
    assert filtered[0].tgt.content == "b"  # type: ignore
