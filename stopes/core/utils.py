# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import builtins
import contextlib
import dataclasses
import fcntl
import gzip
import hashlib
import heapq
import json
import logging
import lzma
import os
import resource
import shlex
import shutil
import subprocess
import tempfile
import time
import typing as tp
import warnings
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import omegaconf
import posix_ipc
import requests  # type: ignore

logger = logging.getLogger(__name__)


class InputOutput(tp.NamedTuple):
    input: Path
    output: Path


def config_sha(*, _parent_: tp.Optional[omegaconf.DictConfig] = None) -> str:
    """When using ${config_sha:} in a Hydra .yaml file,
    it will be replaced by a sha of the current node config.
    This make it easier to create unique folders.
    """
    conf = str(_parent_)
    assert (
        ": '???'" not in conf
    ), "${config_sha:} can only be used for config where you specified all the fields"
    return sha_key(conf)


def sha_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def ensure_dir(path: tp.Union[str, Path]) -> None:
    os.makedirs(path, exist_ok=True)


def bash_pipefail(*pipe_parts: str) -> str:
    """Run a bash pipelines with "-o pipefail".
    This allows to catch zcat failures.
    Note that it will also generate error if you use "head" in your pipeline.
    The arguments are supposed to be valid bash commands.
    """
    pipe = " | "
    return shlex.join(["/bin/bash", "-o", "pipefail", "-c", pipe.join(pipe_parts)])


def test_bash_pipefail() -> None:
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
        tmpdir = Path("/scratch") / "slurm_tmpdir" / slurm_jobid
    else:
        tmpdir = output.parent
    _, tmp_path = tempfile.mkstemp(dir=tmpdir, prefix=prefix, suffix=suffix)
    return Path(tmp_path)


@contextlib.contextmanager
def open_write(output: Path, mode: str = "wt", **kwargs) -> tp.Iterator[tp.IO]:
    """
    Open a temporary file for writing, and on success rename it to the target name.
    The operation may fail on some Unix flavors if src and dst are on different filesystems,
    in particular, if the temp file is local to a slurm job, it might fail moving it to an nfs
    directory. (see https://docs.python.org/3/library/os.html#os.rename)
    """
    assert "w" in mode, f"Can't use open_write with mode: {mode}"
    tmp = tmp_file(output)
    with open(tmp, mode=mode, **kwargs) as o:
        yield o
    # No try/catch we are only renaming in case of success
    tmp.rename(output)


def audio_duration(file: Path) -> float:
    """audio file duration in seconds"""
    # use M2C2 metadata if available
    json_file = file.with_suffix(".json")
    if json_file.exists():
        json_info = json.loads(json_file.read_text())
        duration = json_info.get("duration", None)
        if duration is not None:
            return float(duration)

    import torchaudio

    torchaudio.set_audio_backend("sox_io")
    info = torchaudio.info(file)
    num_frames, sample_rate = info.num_frames, info.sample_rate

    # For mp3 we don't have an actual num_frames.
    # The file has to be decoded first
    # https://github.com/pytorch/audio/issues/2524
    if num_frames == 0:
        audio, sample_rate = torchaudio.load(file)
        num_frames = audio.shape[-1]

    return float(num_frames / sample_rate)


def make_duration_batches(
    files: tp.Iterable[Path], max_duration: tp.Optional[float]
) -> tp.List[tp.List[Path]]:
    if max_duration is None:
        # Sometime the data is already sharded (eg M2C2), we don't need to reshard them
        return [[f] for f in files]

    all_batches = []
    batch_duration = 0.0
    batch: tp.List[Path] = []
    for file in files:
        file = Path(file)
        duration = audio_duration(file)
        if batch_duration + duration > max_duration:
            if batch:
                all_batches.append(batch)
            batch = []
            batch_duration = 0.0
        batch.append(file)
        batch_duration += duration

    if batch:
        all_batches.append(batch)
    return all_batches


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


def test_convert_size_unit() -> None:
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


def sort_file(
    input_file: tp.Union[Path, str],
    output_file: tp.Union[Path, str],
    col: tp.Optional[tp.Union[int, str]] = None,
    sep: tp.Optional[str] = None,
    no_duplicate: bool = False,
):
    """
    Sort the lines in input_file by the values of column `col` (or by the entire line if
    `col` is None), and save results to `output_file`. Note that `input_file` must be
    uncompressed before. If `no_duplicate` is True, sorting will do the internal
    de-duplication
    """

    # An empirical chunk to be sorted in-memory has a size of 64MB
    # This value can be set at runtime via "STOPES_SHARD_CHUNK_SIZE"
    chunk_file_size = int(os.getenv("STOPES_SHARD_CHUNK_SIZE", 64 * 1024))
    file_size = os.stat(input_file).st_size
    no_of_chunks = max(int(file_size / chunk_file_size), 1)

    if no_of_chunks > 1:
        from stopes.utils.file_chunker_utils import Chunker, find_offsets

        offsets = find_offsets(str(input_file), no_of_chunks)
        chunk_offsets = zip(offsets, offsets[1:])
        inputs = [
            Chunker(str(Path(input_file).resolve()), start, end)
            for start, end in chunk_offsets
        ]
    else:
        inputs = [open(input_file)]  # type: ignore[list-item]

    # If the argument `col` is a string, the input_file has a header and this must be
    # hold separately
    has_header = isinstance(col, str) and not col.isdecimal()
    header = None

    def col_value(line):
        if col is None:
            return line
        else:
            assert isinstance(col, int)
            cols = line.rstrip().split(sep)
            assert col < len(
                cols
            ), f"Invalid value for `col` ({col}), file has only {len(cols)} columns)"
            return cols[col]

    def merge_sort(chunk_files: tp.List[Path]):
        # Sort the lines from n opening chunks
        with contextlib.ExitStack() as stack:
            chunks = [stack.enter_context(open(f)) for f in chunk_files]
            keyed_chunks = [
                ((col_value(line), line) for line in chunk) for chunk in chunks
            ]

            # de-duplicate the line based on the key
            prev_key = None
            for key, line in heapq.merge(*keyed_chunks):
                if prev_key != key or not no_duplicate:
                    yield line
                prev_key = key

    def merge_output(
        files: tp.List[Path], output_dir: Path, iter_no: int
    ) -> tp.List[Path]:
        # Merge the (maybe big) list of partial files into one sorted output
        # Make sure not too many files are opened than what is allowed by the OS

        num_cpus = os.cpu_count()
        if num_cpus is None:
            num_cpus = 1
        max_ofile_per_cpu = (
            resource.getrlimit(resource.RLIMIT_OFILE)[0] // num_cpus // 2
        )
        output_files = []

        for batch_no, files_batch in enumerate(
            batch(files, batch_size=max_ofile_per_cpu)
        ):
            output_file = output_dir / f".m{iter_no}.{batch_no}"
            with open(output_file, "w") as output:
                output.writelines(merge_sort(files_batch))
            output_files.append(output_file)

        return output_files

    with tempfile.TemporaryDirectory() as tmp_dir:
        outputs = []
        for i, _input in enumerate(inputs):
            lines = None
            with _input as lines_chunk:
                lines = list(iter(lines_chunk))  # type: ignore[type-var]

            # This happens when e.g. the line is too big to fit in one chunk
            if lines is None or len(lines) == 0:
                continue
            if has_header and i == 0:
                header, *lines = lines
                col = header.rstrip("\n").split(sep).index(col)  # type: ignore[arg-type]
            lines.sort(key=col_value)

            tmp_output_path = Path(tmp_dir) / str(i)
            with open(tmp_output_path, "a+") as tmp_output:
                tmp_output.writelines(lines)

            outputs.append(tmp_output_path)

        # Small optimizattion of IO: In case of small input file (sorting in memory), no header and duplicates,
        # The first intermediately sorted file is already the final result
        if len(outputs) == 1 and header is None and not no_duplicate:
            shutil.move(str(outputs[0]), output_file)
            return

        round_no = 1
        while len(outputs) > 1:
            outputs = merge_output(outputs, Path(tmp_dir), round_no)
            round_no += 1

        with open(output_file, "w") as output, open(outputs[0]) as input:
            if header:
                output.write(header)
            for line in input:
                output.write(line)


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
    # We are returning a DictConfig with the same members than config_class,
    # so for mypy this shouldn't make a difference.
    return proto  # type: ignore


@contextlib.contextmanager
def clone_config(config: TConfig) -> tp.Iterator[TConfig]:
    with omegaconf.open_dict(config.copy()) as cfg:  # type: ignore
        omegaconf.OmegaConf.set_readonly(cfg, False)
        yield cfg  # type: ignore


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


class FileLock:
    """
    This class locks a shared directory from concurrent read / write

    One use case: Download AWS files into /share/XYZ in an stopes
    array module with many concurrent runs. We want to make sure
    no other process reads the incomplete files before the download
    is finished.

    with FileLock(lock_file_path):
        do_something_with_file()
    """

    def __init__(self, key_path: Path):
        key_path.parent.resolve().mkdir(exist_ok=True, parents=True)
        self._key_path = key_path

    def __enter__(self):
        self.f = open(self._key_path, "w")
        fcntl.flock(self.f.fileno(), fcntl.LOCK_EX)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        fcntl.flock(self.f.fileno(), fcntl.LOCK_UN)
        self.f.close()


@contextlib.contextmanager
def measure(
    start_msg: str,
    logger: "logging.Logger",
    end_msg: str = "done in ",
    enable_log: bool = True,
) -> tp.Iterator[None]:
    if enable_log:
        logger.info(start_msg)
    start = time.perf_counter()
    yield
    if enable_log:
        logger.info(f"{start_msg} {end_msg}: {time.perf_counter() - start:.3f} secs")


T = tp.TypeVar("T")


def batch(items: tp.Iterable[T], batch_size: int) -> tp.Iterator[tp.List[T]]:
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


#####################################################
# AsyncIO utils
#####################################################

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


semaphore_logger = logging.getLogger("stopes.utils.semaphore")


@dataclasses.dataclass(frozen=True)
class AsyncIPCSemaphore:
    """
    An async contextmanager that uses posix_ipc to provide a system backed semaphore. This means that
    the semaphore can be shared between difference python processes.

    If you set `initial_value` to 0 or a negative number, the semaphore will not block, it will just go through immediately.

    # from the posix_ipc documentation:

    If `name` is None, this will create a new semaphore with a random name.
    Otherwise, the name should begin with a `/` and be a valid path component.

    The flags specify whether you want to create a new semaphore or open an existing one.
        - With flags set to the default of 0, the module attempts to open an existing semaphore
          and raises an error if that semaphore doesn't exist.
        - With flags set to posix_ipc.O_CREAT, the module opens the semaphore if it exists
          (in which case mode and initial value are ignored) or creates it if it doesn't.
        - With flags set to posix_ipc.O_CREAT | posix_ipc.O_EXCL (or posix_ipc.O_CREX), the module creates a new semaphore identified by name.
          If a semaphore with that name already exists, the call raises an ExistentialError.

    A timeout of None (the default) implies no time limit. The call will not return until its wait condition is satisfied.
    When timeout is 0, the call immediately raises a BusyError if asked to wait. Since it will return immediately if not asked to wait,
      this can be thought of as "non-blocking" mode.
    When the timeout is > 0, the call will wait no longer than timeout seconds before either returning
      (having acquired the semaphore) or raising a BusyError.

    On platforms that don't support the sem_timedwait() API, a timeout > 0 is treated as infinite.
      The call will not return until its wait condition is satisfied.

    see https://github.com/osvenskan/posix_ipc/blob/develop/USAGE.md#the-semaphore-class for details
    want to see what's going on, on linux check `ls -l /dev/shm`
    """

    name: tp.Optional[str]
    flags: int = 0
    mode: int = 0o0600
    initial_value: int = 0
    timeout: tp.Optional[float] = None

    _sem: tp.Optional[posix_ipc.Semaphore] = dataclasses.field(init=False, default=None)

    def __post_init__(
        self,
    ) -> None:
        assert (self.name is None) or self.name[
            0
        ] == "/", "semaphore name should have a leading "
        if not posix_ipc.SEMAPHORE_TIMEOUT_SUPPORTED and self.timeout:
            warnings.warn(
                "Semaphore timeouts are not supported on your system, some code might block indefinitely.",
            )

        if self.initial_value > 0:
            object.__setattr__(
                self,
                "_sem",
                posix_ipc.Semaphore(
                    self.name,
                    self.flags,
                    self.mode,
                    self.initial_value,
                ),
            )
            if self._sem and self._sem.value != self.initial_value:
                warnings.warn(
                    "The semaphore was initialized with a different initial_value, you are reusing an existing semaphore."
                )

    def _blocking_aquire(self):
        if not self._sem:
            # don't do anything, just go through
            return
        semaphore_logger.debug("aquiring lock, current count: %d", self.value)
        # this will block, not async
        self._sem.acquire(self.timeout)

    async def acquire(self):
        if not self._sem:
            # don't do anything, just go through
            return self
        # we get the asyncio loop
        loop = asyncio.get_running_loop()
        # so we can run the blocking aquire in a separate thread/process (depends on the default executor here)
        await loop.run_in_executor(None, self._blocking_aquire)

    def release(self):
        if self._sem:
            semaphore_logger.debug("releasing lock, current count: %d", self.value)
            self._sem.release()

    async def __aenter__(self) -> "AsyncIPCSemaphore":
        await self.acquire()
        # note that we will not return (enter the context) until the await above
        # doesn't release, that is, until the semaphore.aquire doesn't pass.
        return self

    async def __aexit__(self, exc_t, exc_v, exc_tb) -> bool:
        self.release()
        return False  # let the exception be raised pass the context

    @property
    def value(self) -> tp.Optional[int]:
        """
        return the number of slots left in the semaphore
        """
        if posix_ipc.SEMAPHORE_VALUE_SUPPORTED and self._sem:
            return self._sem.value
        return None

    def __del__(self):
        # we need to cleanup the semaphore otherwise the lock file stays around
        # and creates problems if you want to reuse it
        if self._sem:
            self._sem.unlink()
            self._sem.close()


def download_zip_and_extract_all_to_dir(dest_dir: Path, zip_url: str) -> None:
    try:
        url = requests.get(zip_url)
    except Exception as e:
        logger.error(f"Fail to download zip from {zip_url}")
        raise e
    zipfile = ZipFile(BytesIO(url.content))
    zipfile.extractall(dest_dir)
