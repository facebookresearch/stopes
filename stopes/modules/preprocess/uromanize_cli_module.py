# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import functools
import logging
import shlex
import subprocess
import tempfile
import typing as tp
from dataclasses import dataclass
from multiprocessing import Lock
from pathlib import Path

from omegaconf import MISSING

from stopes.core import stopes_module, utils
from stopes.core.utils import download_zip_and_extract_all_to_dir

logger = logging.getLogger(__name__)

UROMAN_GIT_ZIP_URL = "https://github.com/isi-nlp/uroman/archive/refs/heads/master.zip"
_UROMAN_DIR_LOCK = Lock()


@functools.lru_cache()
def get_uroman_dir() -> Path:
    tmp_uroman_dir = Path(tempfile.gettempdir()) / "uroman"
    tmp_uroman_dir.mkdir(exist_ok=True)
    return tmp_uroman_dir


@functools.lru_cache()
def get_uroman_executable_path() -> Path:
    tmp_uroman_dir = get_uroman_dir()
    tmp_uroman_executable_path = tmp_uroman_dir / "uroman-master" / "bin" / "uroman.pl"
    with _UROMAN_DIR_LOCK:
        if not tmp_uroman_executable_path.exists():
            download_zip_and_extract_all_to_dir(tmp_uroman_dir, UROMAN_GIT_ZIP_URL)
    return tmp_uroman_executable_path


def get_uroman_commands(lang: str) -> tp.List[str]:
    return [
        f"perl {get_uroman_executable_path()} -l {lang}",
        "awk '{$1=$1};1'",  # strip whitespaces
    ]


def run_uroman_cli_standalone(input_file: Path, output_file: Path, lang: str) -> None:
    cmds = [utils.open_file_cmd(input_file)]
    cmds.extend(get_uroman_commands(lang))

    command = utils.bash_pipefail(*cmds)
    logger.info(f"uroman command: ${command}")
    try:
        subprocess.run(
            f"{command} > {shlex.quote(str(output_file))}",
            shell=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        if output_file.is_file():
            output_file.unlink()
        logger.error(
            f"ERROR during encoding of {input_file}. Deleted corrupted output file.",
            exc_info=e,
        )
        raise e


def uromanize(text: tp.List[str]) -> tp.List[str]:
    if text is None or len(text) == 0:
        return []
    with tempfile.NamedTemporaryFile() as input_file, tempfile.NamedTemporaryFile() as output_file:
        with open(input_file.name, "w") as f:
            for sentence in text:
                f.write(f"{sentence}\n")
        run_uroman_cli_standalone(
            input_file=Path(input_file.name),
            output_file=Path(output_file.name),
            lang="xxx",
        )
        uromanized_text = []
        with open(output_file.name) as f:
            for line in f:
                uromanized_text.append(line.rstrip("\n"))
    return uromanized_text


@dataclass
class UromanPreprocessConfig:
    # Default to "xxx" because only the "xxx" language code was used in the MMS TTS model.
    lang: str = "xxx"
    input_file: str = MISSING
    output_dir: str = MISSING


class UromanPreprocessModule(stopes_module.StopesModule):
    def __init__(self, config: UromanPreprocessConfig):
        super().__init__(config, UromanPreprocessConfig)
        self.output_dir = Path(self.config.output_dir).resolve()
        self.output_dir.mkdir(exist_ok=True)

    def requirements(self) -> stopes_module.Requirements:
        return stopes_module.Requirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=4,
            timeout_min=60 * 24,
        )

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> Path:
        input_file = Path(self.config.input_file)  # type: ignore
        assert input_file.exists()
        output_file = self.output_dir / f"uroman.{input_file.stem}.{self.config.lang}"
        run_uroman_cli_standalone(
            input_file,
            output_file,
            self.config.lang,
        )
        return output_file

    def name(self):
        return f"uroman_cli.{self.config.lang}"

    def version(cls):
        return "1.2.8"
