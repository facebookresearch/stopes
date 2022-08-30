# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import shlex
import typing as tp
from pathlib import Path

from stopes.core.utils import open_file_cmd


def _maybe_cat(file: Path) -> str:
    if file.suffix in (".xz", ".gz"):
        opencommand = open_file_cmd(file)
        # https://www.gnu.org/software/bash/manual/html_node/Process-Substitution.html
        return f"<({opencommand})"
    return shlex.quote(str(file))


def build_sort_command(
    files: tp.Iterable[Path],
    is_merge: bool,
    num_cpu: int,
    tmp_dir: Path,
    field_def: str = "6",  # this is an offset for the `sort` command, it starts at 1.
) -> str:

    base_cmd = [
        "sort",
        "-S 50%",  # use 50% of available memory
        f"--parallel={num_cpu}",  # sort in parallel over how many processes
        "--unique",  # only keep the first occurence - this is the dedup core
        "-t$'\\t'",  # tab separator
        "-k",
        str(field_def),  # field to start looking from (6 -> end)
        f"--temporary-directory={shlex.quote(str(tmp_dir.resolve()))}",  # where to put the temp files when processing
    ]

    catfiles = [_maybe_cat(f) for f in files]

    if is_merge:
        # if merging, do not sort, just merge already sorted files
        base_cmd.append("--merge")

    return " ".join(base_cmd + catfiles)
