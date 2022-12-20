# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import typing as tp

import numpy as np
from sentence_transformers import SentenceTransformer

from stopes.modules.preprocess.encode_to_npy import EncodeToNPY
from stopes.utils.mining_utils import extract_shard_id


class HFTextEncoder(EncodeToNPY):
    """
    1. load a pre-trained model located in the HuggingFace sentence transformers Hub
    2. tokenize and encode input
    3. send embeddings to specified output file
    """

    def __init__(
        self,
        encoder_model: str,
        input_file: str,
        _name: str = "hf",
        output_dir: str = ".",
        input_file_idx: int = 0,
        outfile_prefix: str = "encf",
        outfile_postfix: str = "",
        normalize: bool = False,
        fp16: bool = False,
        # ignored
        spm_vocab: str = "",
        spm_model: str = "",
    ) -> None:
        super().__init__(
            outfile_prefix=outfile_prefix,
            outfile_postfix=outfile_postfix,
            input_file=input_file,
            input_file_idx=input_file_idx,
            output_dir=output_dir,
            normalize=normalize,
            fp16=fp16,
        )
        self.model = SentenceTransformer(encoder_model)

    def name_output_file(self) -> str:
        shard_idx = extract_shard_id(self.input_file, default=self.input_file_idx)

        return os.path.abspath(
            os.path.join(
                self.output_dir,
                f"{self.outfile_prefix}.{shard_idx:03d}.{self.outfile_postfix}",
            )
        )

    def encode_to_np(
        self, lines_with_number: tp.Iterator[tp.Tuple[int, str]]
    ) -> np.ndarray:
        return self.model.encode([s for (_, s) in lines_with_number])

    def __exit__(self, _exc_type, _exc_value, _traceback):
        return None
