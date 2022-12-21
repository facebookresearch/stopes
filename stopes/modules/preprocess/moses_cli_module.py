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
from pathlib import Path

import requests
from omegaconf import MISSING

from stopes.core import stopes_module, utils
from stopes.utils.mining_utils import tokenization_type

log = logging.getLogger("stopes.moses")


@dataclass
class MosesPreprocessConfig:
    # core conf
    lang: str = MISSING
    shards: tp.List[str] = MISSING
    output_dir: str = MISSING
    lowercase: bool = True
    # shard name, if it is different for lang name (used for naming outputs)
    lang_shard_name: tp.Optional[str] = None
    normalize_punctuation: bool = True
    remove_non_printing_chars: bool = False
    deescape_special_chars: bool = False


class MosesPreprocessModule(stopes_module.StopesModule):
    """
    Module to run the moses processing perl scripts on a set of text files
    """

    def __init__(self, config: MosesPreprocessConfig = MosesPreprocessConfig()):
        super().__init__(config, MosesPreprocessConfig)

        N = len(self.config.shards)
        self.punc_lang = tokenization_type(self.config.lang)
        log.info(f"Preprocess {self.config.lang} ({self.punc_lang}), {N} files")
        self.output_dir = Path(self.config.output_dir).resolve()
        self.output_dir.mkdir(exist_ok=True)

    def array(self):
        return self.config.shards

    def requirements(self):
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
        input_file = Path(iteration_value)  # type: ignore
        assert input_file.exists()
        # TODO: find a way to allow the caller to specify output
        output_file = (
            self.output_dir
            / f"moses.{iteration_index:0>3}.{self.config.lang_shard_name or self.config.lang}"
        )

        cmds = [utils.open_file_cmd(input_file)]
        cmds.extend(get_moses_commands(self.config, self.punc_lang))

        command = utils.bash_pipefail(*cmds)
        log.info(f"moses command: ${command}")
        try:
            subprocess.run(
                f"{command} > {shlex.quote(str(output_file))}",
                shell=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            if output_file.is_file():
                output_file.unlink()
            log.error(
                f"ERROR during encoding of {input_file}. Deleted corrupted output file.",
                exc_info=e,
            )
            raise e

        return output_file

    def name(self):
        return (
            f"moses_cli.{self.config.lang}.{len(self.config.shards)}.{self.sha_key()}"
        )

    def version(cls):
        return "0.2"


def get_moses_commands(config, lang):
    cmds = []
    if config.remove_non_printing_chars:
        rm_non_printing = get_moses_script("remove-non-printing-char.perl")
        cmds.append(f"perl {rm_non_printing}")
    if config.normalize_punctuation:
        norm_punct = get_moses_script("normalize-punctuation.perl")
        cmds.append(f"perl {norm_punct} -l {lang}")
    if config.deescape_special_chars:
        deescape = get_moses_script("deescape-special-chars.perl")
        cmds.append(f"perl {deescape}")
    if config.lowercase:
        lowercase = get_moses_script("lowercase.perl")
        cmds.append(f"perl {lowercase}")
    return cmds


@functools.lru_cache()
def resolve_moses_dir() -> Path:
    try:
        current_dir = Path(__file__).parent.resolve()
        moses_dir = current_dir / "moses"
        moses_dir.mkdir(exist_ok=True)
    except:
        # We try to download the moses script inside the repo,
        # but depending how/where stopes is installed we might not be able to,
        # so we use /tmp instead.
        moses_dir = Path(tempfile.gettempdir()) / "moses"
        moses_dir.mkdir(exist_ok=True)
    return moses_dir


@functools.lru_cache()
def get_moses_script(name) -> Path:
    moses_dir = resolve_moses_dir()
    if "/" not in name:
        name = "scripts/tokenizer/" + name
    moses_script = moses_dir / name
    if moses_script.exists():
        return moses_script

    url = "https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/"
    url += name
    try:
        res = requests.get(url)
    except Exception as e:
        log.error(f"Didn't find moses script {name} at {url} ({e})")
        raise
    moses_script.parent.mkdir(parents=True, exist_ok=True)
    moses_script.write_text(res.text)
    return moses_script


if __name__ == "__main__":
    get_moses_script("remove-non-printing-char.perl")
    get_moses_script("normalize-punctuation.perl")
    get_moses_script("deescape-special-chars.perl")
    get_moses_script("lowercase.perl")
