# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pydantic import BaseModel


class LineQuery(BaseModel):
    gz_path: str


class AudioQuery(BaseModel):
    path: str
    start: int = -1
    end: int = -1
    sampling: str = "wav"
    context_size: float = 0  # window in seconds


class AnnotationQuery(BaseModel):
    gz_path: str
    start_idx: int = 0
    end_idx: int = 10


class DefaultQuery(BaseModel):
    gz_path: str
    start_idx: int = 0
    end_idx: int = 10
