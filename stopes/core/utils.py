# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import builtins
import contextlib
import gzip
import hashlib
import json
import lzma
import os
import shlex
import subprocess
import tempfile
import time
import typing as tp
from pathlib import Path

import omegaconf

if tp.TYPE_CHECKING:
    from stopes.core import StopesModule


class InputOutput(tp.NamedTuple):
    input: Path
    output: Path


def config_sha(_parent_=None) -> str:
    conf = str(_parent_)
    assert (
        ": '???'" not in conf
    ), "${config_sha:} can only be used for config where you specified all the fields"
    return sha_key(conf)


def sha_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def ensure_dir(path: tp.Union[str, Path]):
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


def open_file_cmd(filename: tp.Union[Path, str]) -> str:
    if isinstance(filename, Path):
        filename = str(filename)
    filename = shlex.quote(filename)
    cat = "cat"
    if filename.endswith(".xz"):
        cat = "xzcat"
    if filename.endswith(".gz"):
        cat = "zcat"

    return shlex.join((cat, filename))


def set_file_compression(filename: str) -> tp.List[str]:
    """Produce a list of commant to write a file (a partial inverse of `open_file_cmd`)"""
    commands = []
    if isinstance(filename, Path):
        filename = shlex.quote(str(filename))
    if filename.endswith(".xz"):
        commands.append("xz")
    if filename.endswith(".gz"):
        commands.append("gzip")
    return commands


def open(
    filename: tp.Union[Path, str],
    mode: str = "rt",
    encoding: tp.Optional[str] = "utf-8",
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
    slurm_jobid = os.environ.get("SLURM_JOB_ID", None)
    if slurm_jobid:
        tmpdir = Path("/scratch") / slurm_jobid
    else:
        tmpdir = output.parent
    _, tmp_path = tempfile.mkstemp(dir=tmpdir, prefix=prefix, suffix=suffix)
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


def audio_duration(file: Path) -> float:
    """audio file duration in seconds"""

    duration = None

    # use M2C2 metadata if available
    json_file = file.with_suffix(".json")
    if json_file.exists():
        json_info = json.load(open(json_file, "r"))
        duration = json_info.get("duration", None)

    if duration is None:
        import torchaudio

        torchaudio.set_audio_backend("sox_io")
        wav, sample_rate = torchaudio.load(file)
        duration = wav.shape[1] / sample_rate

    return duration


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
    tmp_dir.mkdir(exist_ok=True, parents=True)
    for f in files:
        if xz_size(f) < convert_size_unit(max_size):
            yield f
        else:
            # split file
            file_name = f"{f.stem}.split."
            file_prefix = str((tmp_dir / file_name).resolve())
            # remove possible old files from there
            for spl in tmp_dir.glob(f"{file_name}*"):
                spl.unlink()
            subprocess.run(
                bash_pipefail(
                    open_file_cmd(f),
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


def expand_if_compressed(input_file: Path, tmp_dir: Path) -> tp.Optional[Path]:
    if input_file.suffix in {".gz", ".xz"}:
        print(f"expanding {input_file.name}")
        decompressed_tmp = tmp_dir / f"{input_file.name}_expanded.txt"
        decompressed_tmp = decompressed_tmp.resolve()
        subprocess.run(
            " ".join(
                [
                    open_file_cmd(input_file),
                    ">",
                    shlex.quote(str(decompressed_tmp)),
                ]
            ),
            shell=True,
            check=True,
        )
        return decompressed_tmp
    else:
        return None


TConfig = tp.TypeVar("TConfig")


def promote_config(
    config: omegaconf.DictConfig, config_class: tp.Type[TConfig]
) -> TConfig:
    if hasattr(config, "_target_"):
        # Remove magic Hydra config fields.
        # At this point, Hydra already did its job.
        read_only = config._get_flag("readonly")
        omegaconf.OmegaConf.set_readonly(config, False)
        del config._target_
        omegaconf.OmegaConf.set_readonly(config, read_only)

    # Note: we don't use config._promote since it merges the other way around:
    # the proto into the config.
    proto = omegaconf.OmegaConf.structured(config_class)
    proto.merge_with(config)
    if hasattr(config, "_parent"):
        proto._set_parent(config._parent)
    return proto


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


def count_lines(filename: str) -> int:
    """
    Count the number of lines in a file.
    """
    result = subprocess.run(
        bash_pipefail(
            open_file_cmd(filename),
            shlex.join(["wc", "-l"]),
        ),
        capture_output=True,
        shell=True,
    )
    out = result.stdout.decode("utf-8")
    lines_numbers = [int(line) for line in out.split() if line]
    assert len(lines_numbers) == 1
    return lines_numbers[0]


@contextlib.contextmanager
def measure(start_msg, logger, end_msg="done in ", enable_log=True):
    if enable_log:
        logger.info(start_msg)
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start
    if enable_log:
        logger.info(f"{start_msg} {end_msg}: {time.perf_counter() - start:.3f} secs")


def batch(items: tp.List[tp.Any], batch_size: int):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


TAwaitReturn = tp.TypeVar("TAwaitReturn")


async def _wrap_opt(
    opt: tp.Optional[tp.Awaitable[TAwaitReturn]],
) -> tp.Optional[TAwaitReturn]:
    if opt is None:
        return None
    return await opt


async def gather_optionals(
    *awaitables: tp.Optional[tp.Awaitable[TAwaitReturn]],
) -> tp.List[tp.Optional[TAwaitReturn]]:
    """
    sometimes we don't know if we'll really have an awaitable for all our params,
    but asyncio.gather wants only awaitables, so we end up building awkwards lists with conditional
    appends. But it would be a lot nicer if we could just write:
    x, y, z = gather_optionals(
        do_x(),
        do_y(),
        do_z() if cond else None
    )
    """
    return await asyncio.gather(*[_wrap_opt(opt) for opt in awaitables])
