# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import pickle
import sys
import typing as tp
from pathlib import Path

import torch


class FakeTensor(tp.NamedTuple):
    dtype: torch.dtype
    key: int
    device: str
    numel: int


class SkipTensorUnpickler(pickle._Unpickler):
    def __init__(self, f: tp.BinaryIO, **kwargs):
        super().__init__(f, **kwargs)
        self._fix_dispatch()

    def persistent_load(self, saved_id):
        _, storage, key, device, numel = saved_id
        return FakeTensor(storage.dtype, key, device, numel)

    def load_reduce(self):
        stack = self.stack  # type: ignore
        args = stack.pop()
        func = stack[-1]
        if func is torch._utils._rebuild_tensor_v2:
            # Return the fake tensor directly
            stack[-1] = args[0]
            return
        stack[-1] = func(*args)

    def _fix_dispatch(self):
        self.dispatch[ord("R")] = SkipTensorUnpickler.load_reduce  # type: ignore


def load_model_conf(model_path: Path):
    """Load a fairseq checkpoint, but don't read the tensors.

    This makes it very fast to read a model config.
    """
    # TODO: commit this to fairseq, it's not useful here anymore,
    # but the code is too cool to remove
    with model_path.open("rb") as zip_file:
        with torch.serialization._open_zipfile_reader(zip_file) as zf:
            data_file = io.BytesIO(zf.get_record("data.pkl"))
            unpkl = SkipTensorUnpickler(data_file)
            checkpoint = unpkl.load()
            print(checkpoint["cfg"])
            print(checkpoint["model"]["decoder.layers.0.final_layer_norm.bias"])


if __name__ == "__main__":
    load_model_conf(Path(sys.argv[1]))
