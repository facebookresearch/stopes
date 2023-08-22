# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from pathlib import Path

import pytest

import stopes
from stopes.core import utils
from stopes.modules.preprocess.uromanize_cli_module import (
    run_uroman_cli_standalone,
    uromanize,
)
from stopes.pipelines.tests import test_configs


def _mock_input_file(tmp_path: Path) -> Path:
    mock_input_file = tmp_path / "uroman_input"
    with utils.open(mock_input_file, "w") as in_f:
        in_f.write("ちょっとまってください\n" "アメリカ")
    return mock_input_file


def test_run_uroman_cli_standalone(tmp_path: Path) -> None:
    input_file = _mock_input_file(tmp_path)
    output_file = tmp_path / "uroman_output"
    run_uroman_cli_standalone(input_file, output_file, lang="xxx")
    expected_output = "chottomattekudasai\namerika\n"
    with utils.open(output_file) as out_f:
        assert out_f.read() == expected_output


def test_uroman_cli_module(tmp_path: Path) -> None:
    input_file = _mock_input_file(tmp_path)
    output_dir = tmp_path / "output"
    conf_path = (
        test_configs.STOPES / "pipelines" / "speech" / "conf" / "uromanization.yaml"
    )
    cfg = test_configs.load_conf(
        conf_path,
        (
            f"lang=xxx",
            f"input_file={input_file}",
            f"output_dir={output_dir}",
        ),
    )
    module = stopes.core.StopesModule.build(cfg)
    module.requirements()
    output_file = module.run()
    expected_output = "chottomattekudasai\namerika\n"
    with utils.open(output_file) as out_f:
        assert out_f.read() == expected_output
        assert os.path.basename(out_f.name) == "uroman.uroman_input.xxx"


@pytest.mark.parametrize(
    "text,expected_output",
    [
        (["ちょっとまってください", "アメリカ"], ["chottomattekudasai", "amerika"]),
        ([], []),
        (None, []),
    ],
)
def test_uromanize(text, expected_output):
    assert uromanize(text) == expected_output
