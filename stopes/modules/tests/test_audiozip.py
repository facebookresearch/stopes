# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path

import pytest

from stopes.modules.speech.audio_zip import AudioZipWriter
from stopes.modules.speech.utils import AudioBytes


@pytest.mark.parametrize(
    "metadata_column_values,expected_output",
    [
        (["test"], "/tmp/audio.zip:10:20\ttest"),
        (None, "/tmp/audio.zip:10:20"),
    ],
)
def test_AudioZipWriter_get_manifest_line(metadata_column_values, expected_output):
    audio_zip_writer = AudioZipWriter(
        output_zip=Path("/tmp/audio.zip"),
        manifest_path=Path("manifest.tsv"),
        metadata_column_names=["text"],
    )
    print(
        audio_zip_writer._get_manifest_line(
            audio_bytes=AudioBytes(
                path="/tmp/audiofile", byte_offset=10, length=20, sample_rate=16_000
            ),
            metadata_column_values=metadata_column_values,
        )
    )
    assert (
        audio_zip_writer._get_manifest_line(
            audio_bytes=AudioBytes(
                path="/tmp/audiofile", byte_offset=10, length=20, sample_rate=16_000
            ),
            metadata_column_values=metadata_column_values,
        )
        == expected_output
    )
