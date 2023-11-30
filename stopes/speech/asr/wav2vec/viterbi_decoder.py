# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, List

import torch

from .base_decoder import BaseDecoder


class ViterbiDecoder(BaseDecoder):
    def decode(
        self,
        emissions: torch.FloatTensor,
    ) -> List[List[Dict[str, torch.LongTensor]]]:
        def get_pred(e):
            toks = e.argmax(dim=-1).unique_consecutive()
            return toks[toks != self.blank]

        return [
            [{"tokens": get_pred(x), "score": torch.LongTensor([0])}] for x in emissions
        ]
