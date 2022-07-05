# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import typing as tp

import numpy as np


class Embedding:
    def __init__(self, file_path, embedding_dimensions, dtype=np.float32):
        self.file_path = file_path
        self.embedding_dimensions = embedding_dimensions
        self.dtype = dtype

    def __len__(self) -> int:
        return int(
            os.path.getsize(self.file_path)
            / self.embedding_dimensions
            / np.dtype(self.dtype).itemsize
        )

    @contextlib.contextmanager
    def open_for_read(self, mode="mmap") -> tp.Iterator[np.ndarray]:
        emb = None
        try:
            if mode == "mmap":
                emb = np.memmap(
                    self.file_path,
                    dtype=self.dtype,
                    mode="r",
                    shape=(len(self), self.embedding_dimensions),
                )

            elif mode == "memory":
                emb = np.fromfile(self.file_path, dtype=self.dtype, count=-1)
                emb.resize(len(self), self.embedding_dimensions)

            else:
                raise NotImplementedError(
                    f"open_for_read was called with not implemented option: {mode}. Currently accepted modes: mmap, memory, or an empty mode."
                )

            yield emb
        finally:
            # looks like there is no clean way to close an mmap in numpy
            # and that the best way is to remove the ref and hope for it
            # to be GC
            if emb is not None:
                del emb
