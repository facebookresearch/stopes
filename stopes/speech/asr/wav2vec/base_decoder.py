# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import itertools as it
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Union

import torch
from fairseq.data.dictionary import Dictionary
from fairseq.models.fairseq_model import FairseqModel


class BaseDecoder:
    def __init__(self, tgt_dict: Dictionary) -> None:
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)

        self.blank = (
            tgt_dict.index("<ctc_blank>")
            if "<ctc_blank>" in tgt_dict.indices
            else tgt_dict.bos()
        )
        if "<sep>" in tgt_dict.indices:
            self.silence = tgt_dict.index("<sep>")
        elif "|" in tgt_dict.indices:
            self.silence = tgt_dict.index("|")
        else:
            self.silence = tgt_dict.eos()

    def generate(
        self, models: List[FairseqModel], sample: Dict[str, Any], **unused
    ) -> List[List[Dict[str, torch.LongTensor]]]:
        encoder_input = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }
        emissions = self.get_emissions(models, encoder_input)
        return self.decode(emissions)

    def get_emissions(
        self,
        models: List[FairseqModel],
        encoder_input: Dict[str, Any],
    ) -> torch.FloatTensor:
        model = models[0]
        encoder_out = model(**encoder_input)
        if hasattr(model, "get_logits"):
            emissions = model.get_logits(encoder_out)
        else:
            emissions = model.get_normalized_probs(encoder_out, log_probs=True)
        return emissions.transpose(0, 1).float().cpu().contiguous()

    def get_tokens(self, idxs: Union[Iterator, torch.IntTensor]) -> torch.LongTensor:
        if isinstance(idxs, torch.IntTensor):
            idxs = idxs.tolist()  # type: ignore[assignment]
        idxs = (g[0] for g in it.groupby(idxs))  # type: ignore[assignment]
        idxs = filter(lambda x: x != self.blank, idxs)  # type: ignore[assignment]
        return torch.LongTensor(list(idxs))

    def decode(
        self,
        emissions: torch.FloatTensor,
    ) -> List[List[Dict[str, torch.LongTensor]]]:
        raise NotImplementedError
