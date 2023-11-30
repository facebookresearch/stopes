# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools
import gzip
import io
import logging
import shutil
import tempfile
import time
import typing as tp
import zipfile
from pathlib import Path

import requests
from tqdm.auto import tqdm

from stopes.core import utils

log = logging.getLogger(__name__)

# not sure it helps since connections are reseted anyway.
_session = functools.lru_cache()(requests.Session)

STOPES_CACHE_DIR = Path("~/.cache/stopes").expanduser()


def request_get_content(url: str, n_retry: int = 6, **kwargs) -> bytes:
    """Retrieve the binary content at url.

    Retry on connection errors.
    """
    t0 = time.time()
    for i in range(1, n_retry + 1):
        try:
            r = _session().get(url, **kwargs)
            r.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            # Sleep and try again on error, unless it's a 404.
            message = e.args[0] if isinstance(e.args[0], str) else ""
            if i == n_retry or "Client Error" in message:
                raise e
            log.warn(
                f"Swallowed error {e} while downloading {url} ({i} out of {n_retry})"
            )
            time.sleep(10 * 2**i / 2)
    dl_time = time.time() - t0
    dl_speed = len(r.content) / dl_time / 1024
    if dl_time > 60:
        log.info(
            f"Downloaded {url} [{r.status_code}] took {dl_time:.0f}s ({dl_speed:.1f}kB/s)"
        )
    return r.content


def open_remote_file(
    url: str, cache: Path = None, headers: dict = None
) -> tp.Iterable[str]:
    """Download the files at the given url to memory and opens it as a file.
    Assumes that the file is small, and fetch it when this function is called.
    """
    if cache and cache.exists():
        return utils.open(cache)

    # TODO: open the remote file in streaming mode.
    # The hard part is that we need to write the content on disk at the same time,
    # to implement disk caching.
    raw_bytes = request_get_content(url, headers=headers)
    if cache and not cache.exists():
        # The file might have been created while downloading/writing.
        tmp_cache = utils.tmp_file(cache)
        tmp_cache.write_bytes(raw_bytes)
        if not cache.exists():
            tmp_cache.replace(cache)
        else:
            tmp_cache.unlink()

    content = io.BytesIO(raw_bytes)
    if url.endswith(".gz"):
        f: TextIO = gzip.open(content, mode="rt")  # type: ignore
    else:
        f = io.TextIOWrapper(content)

    with f:
        yield from f


def cached_file_download(
    url: str,
    filename: str,
    force_redownload: bool = False,
    base_dir: Path = STOPES_CACHE_DIR,
    unzip: bool = False,
    warning_size: float = 1024**3,
) -> Path:
    """
    If the `filename` file or directory exists in the `base_dir`, and `force_redownload` is false, just return its full path.
    Otherwise, download the file from `url`, save it to this path (optionally, extract from a zip archive), and return it.
    """
    local_file_path = base_dir / filename
    if local_file_path.exists():
        if force_redownload:
            log.info(f"{filename} already downloaded; removing it for redownload")
            if local_file_path.is_dir():
                shutil.rmtree(local_file_path)
            else:
                local_file_path.unlink()
        else:
            log.info(f"{filename} already downloaded")
            return local_file_path

    log.info(f"Downloading {filename}...")
    f = tempfile.NamedTemporaryFile(delete=False)
    temp_file_path = f.name

    with f:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("Content-Length", 0))
        if total_size >= warning_size:
            log.warning(f"Downloading a large file ({total_size:,d} bytes) from {url}")
        progress_bar = tqdm(total=total_size, unit_scale=True, unit="B")

        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
            progress_bar.update(len(chunk))
        progress_bar.close()

    local_file_path.parent.mkdir(parents=True, exist_ok=True)
    if unzip:
        with zipfile.ZipFile(temp_file_path, "r") as zip_ref:
            zip_ref.extractall(local_file_path)
        Path(temp_file_path).unlink()
    else:
        shutil.move(temp_file_path, local_file_path)
    return local_file_path
