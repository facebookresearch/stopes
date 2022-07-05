# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import builtins
import contextlib
import gzip
import hashlib
import lzma
import os
import shlex
import subprocess
import tempfile
import typing as tp
from pathlib import Path

import omegaconf

if tp.TYPE_CHECKING:
    from stopes.core import StopesModule


class InputOutput(tp.NamedTuple):
    input: Path
    output: Path


def sha_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def bash_pipefail(*pipe_parts: str) -> str:
    """Run a bash pipelines with "-o pipefail".
    This allows to catch zcat failures.
    Note that it will also generate error if you use "head" in your pipeline.
    The arguments are supposed to be valid bash commands.
    """
    pipe = " | "
    return shlex.join(["/bin/bash", "-o", "pipefail", "-c", pipe.join(pipe_parts)])


def test_bash_pipefail():
    assert (
        bash_pipefail("cat utils.py", "wc -l")
        == "/bin/bash -o pipefail -c 'cat utils.py | wc -l'"
    )

    assert (
        bash_pipefail('cat "not utils.py"', "wc -l")
        == """/bin/bash -o pipefail -c 'cat "not utils.py" | wc -l'"""
    )

    assert (
        bash_pipefail("cat 'utils.py'", "wc -l")
        == """/bin/bash -o pipefail -c 'cat '"'"'utils.py'"'"' | wc -l'"""
    )


def open_file_cmd(filename: str) -> str:
    if isinstance(filename, Path):
        filename = shlex.quote(str(filename))
    cat = "cat"
    if filename.endswith(".xz"):
        cat = "xzcat"
    if filename.endswith(".gz"):
        cat = "zcat"

    return shlex.join((cat, filename))


def open(
    filename: Path, mode: str = "rt", encoding: tp.Optional[str] = "utf-8"
) -> tp.IO:
    if len(mode) == 1:
        mode += "t"
    if "b" in mode:
        encoding = None
    filename = Path(filename)
    if filename.suffix == ".gz":
        return gzip.open(filename, encoding=encoding, mode=mode)  # type: ignore
    elif filename.suffix == ".xz":
        return lzma.open(filename, encoding=encoding, mode=mode)  # type: ignore
    else:
        return builtins.open(filename, encoding=encoding, mode=mode)  # type: ignore


def tmp_file(output: Path) -> Path:
    suffix = "".join(output.suffixes)
    prefix = output.name[: -len(suffix) + 1]
    suffix = ".tmp" + suffix
    _, tmp_path = tempfile.mkstemp(dir=output.parent, prefix=prefix, suffix=suffix)
    return Path(tmp_path)


@contextlib.contextmanager
def open_write(output: Path, mode: str = "wt", **kwargs) -> tp.Iterator[tp.IO]:
    """Open a temporary file for writing, and on success rename it to the target name."""
    assert "w" in mode, f"Can't use open_write with mode: {mode}"
    tmp = tmp_file(output)
    with open(tmp, mode=mode, **kwargs) as o:
        yield o
    # No try/catch we are only renaming in case of success
    tmp.rename(output)


def xz_size(file: Path) -> int:
    """
    finds the uncompressed size of an xz file, if it's not an
    xz file (extension), return the stat size of the file
    """

    f = file.resolve()

    if not f.suffix.endswith("xz"):
        return f.stat().st_size

    proc = subprocess.run(["xz", "--robot", "--list", str(f)], capture_output=True)
    total = proc.stdout.decode("utf8").splitlines()[-1]
    columns = total.split("\t")
    return int(columns[4])


def convert_size_unit(size: str) -> int:
    """
    size argument passed down to `split`
    from the man page:
        The SIZE argument is an integer and optional unit (example: 10K is 10*1024).
        Units are K,M,G,T,P,E,Z,Y (powers of 1024) or KB,MB,... (powers of 1000)
    """
    units = ["K", "M", "G", "T", "P", "E", "Z", "Y"]
    multiplier = 1024
    if size.endswith("B"):
        multiplier = 1000
        size = size[:-1]

    try:
        return int(size)
    except ValueError:
        # there is a unit in there
        pass
    unit = size[-1:]
    number = int(size[:-1])
    power = units.index(unit) + 1
    multiplier = pow(multiplier, power)
    return number * multiplier


def test_convert_size_unit():
    assert convert_size_unit("10002342300") == 10002342300
    assert convert_size_unit("10KB") == 10_000
    assert convert_size_unit("10K") == 10_240
    assert convert_size_unit("10PB") == 10 * pow(1000, 5)


def split_large_files(
    files: tp.List[Path],
    max_size: str = "500M",
    tmp_dir: Path = Path("/tmp"),
) -> tp.Generator[Path, None, None]:
    """
    yields a set of path to files smaller than max_size.

    If an input file is bigger than max_size, it is split in smaller shards put in tmp_dir
    with the command `split`.

    size argument passed down to `split`
    from the man page:
        The SIZE argument is an integer and optional unit (example: 10K is 10*1024).
        Units are K,M,G,T,P,E,Z,Y (powers of 1024) or KB,MB,... (powers of 1000)

    /tmp will not work for distributed jobs, be careful
    """
    for f in files:
        if xz_size(f) < convert_size_unit(max_size):
            yield f
        else:
            tmp_dir.mkdir(exist_ok=True, parents=True)
            # split file
            file_name = f"{f.stem}.split."
            file_prefix = str((tmp_dir / file_name).resolve())
            # remove possible old files from there
            for spl in tmp_dir.glob(f"{file_name}*"):
                spl.unlink()
            subprocess.run(
                bash_pipefail(
                    open_file_cmd(str(f)),
                    shlex.join(["split", "-C", max_size, "-", file_prefix]),
                ),
                shell=True,
                check=True,
            )
            for spl in tmp_dir.glob(f"{file_name}*"):
                yield spl


def read_files(filenames: tp.List[Path], encoding: str = "utf-8") -> tp.Iterable[str]:
    for file in filenames:
        with open(file, encoding, "rt") as o:
            for line in o:
                yield line


def symlink(target: Path, actual: Path) -> None:
    """Symlink, but allows overidding previous symlink"""
    assert actual.exists(), f"actual path: {actual} doesn't exist"
    if target.is_symlink():
        target.unlink()
    target.symlink_to(actual)


@contextlib.contextmanager
def clone_config(config: omegaconf.DictConfig):
    with omegaconf.open_dict(config.copy()) as cfg:
        omegaconf.OmegaConf.set_readonly(cfg, False)
        yield cfg


def path_append_suffix(path: Path, suffix: str) -> Path:
    """
    Path.with_suffix replaces the current suffix. python 3.11 has append_suffix
    but we are not there yet, so let's do it ourselves. (suffix should include the . if you
    want one).
    """
    return path.with_suffix("".join(path.suffixes) + suffix)
