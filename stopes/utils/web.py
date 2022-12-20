# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools
import gzip
import io
import logging
import time
import typing as tp
from pathlib import Path

import requests

from stopes.core import utils

log = logging.getLogger(__name__)

# not sure it helps since connections are reseted anyway.
_session = functools.lru_cache()(requests.Session)


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
