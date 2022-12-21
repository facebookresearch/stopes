#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import pickle
import unicodedata
from typing import Any, Dict

logger = logging.getLogger(__name__)


def ngrams(string: str, order: int = 6):
    string = string.strip().replace(" ", "")
    return [string[i : i + order] for i in range(len(string) - order + 1)]


def normalize_unicode(string: str):
    normalized = unicodedata.normalize("NFKC", string)
    # normalize whitespace
    return " ".join(normalized.split())


def cache_step_sync(default_step: str):
    def decorator(function: Any):
        def wrapper(*args, **kwargs):
            output_dir = kwargs["output_dir"]
            step = kwargs.get("custom_step_name", default_step)
            cache = check_cache(step, output_dir, kwargs)
            if cache is not None:
                logger.info(f"Re-using cache for {step}")
                return cache
            new = function(*args, **kwargs)
            cache_results(step, output_dir, new, kwargs)
            return new

        return wrapper

    return decorator


def check_cache(step: str, output_dir: str, kwargs: Dict[Any, Any]) -> None:
    output_path = os.path.join(output_dir, "progress", f"{step}.output")
    input_path = os.path.join(output_dir, "progress", f"{step}.input")
    if os.path.exists(output_path) and os.path.exists(input_path):
        try:
            with open(input_path, "rb") as file_in:
                cached_kwargs = pickle.load(file_in)
            if cached_kwargs == kwargs:
                with open(output_path, "rb") as file_out:
                    return pickle.load(file_out)
        except Exception:
            return None
    return None


def cache_results(
    step: str, output_dir: str, results: Any, kwargs: Dict[Any, Any]
) -> None:
    progress_dir = os.path.join(output_dir, "progress")
    os.makedirs(progress_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "progress", f"{step}.output")
    input_path = os.path.join(output_dir, "progress", f"{step}.input")
    with open(output_path, "wb") as file_out:
        pickle.dump(results, file_out)
    with open(input_path, "wb") as file_in:
        pickle.dump(kwargs, file_in)
