# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from abc import abstractmethod
from pathlib import Path

import numpy as np

from stopes.modules.preprocess.line_processor import LineProcessorCallback
from stopes.utils.embedding_utils import EmbeddingConcatenator


class EncodeToNPY(LineProcessorCallback):
    """
    a text encoder is responsible for encoding sentences and writing them to an output file.
    It's used as a context manager so that it can deal with opening/closing the output file resource properly.
    """

    def __init__(
        self,
        outfile_prefix: str,
        input_file: str,
        input_file_idx: int,
        output_dir: str,
        outfile_postfix: str = "",
        normalize: bool = False,
        fp16: bool = False,
    ) -> None:
        super().__init__(
            outfile_prefix=outfile_prefix,
            outfile_postfix=outfile_postfix,
            input_file=input_file,
            input_file_idx=input_file_idx,
            output_dir=output_dir,
        )
        self.normalize = normalize
        self.fp16 = fp16
        self.output_file = self.name_output_file()
        self.concat = EmbeddingConcatenator(
            output_file=self.output_file, fp16=self.fp16
        )

    @abstractmethod
    def name_output_file(self) -> str:
        """
        return a valid name for the output file
        """
        pass

    def __enter__(self):
        self.concat.__enter__()
        self.output_file_pointer = self.concat.output_file_pointer
        return self

    def __exit__(self, *_exc) -> None:
        self.concat.__exit__(*_exc)

    def process_lines(self, lines_with_number: tp.Iterator[tp.Tuple[int, str]]) -> None:
        array = self.encode_to_np(lines_with_number)
        self.concat.append_embedding_from_array(self._normalize(array))

    def final_result(self) -> Path:
        return Path(self.output_file)

    @abstractmethod
    def encode_to_np(
        self, lines_with_number: tp.Iterator[tp.Tuple[int, str]]
    ) -> np.ndarray:
        """
        encode a batch of sentences and return them as an numpy array
        """
        pass

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        if isinstance(embeddings, np.ndarray) and self.normalize:
            assert (
                embeddings.dtype == np.float32
            ), f"cannot normalize {embeddings.dtype}"
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings
