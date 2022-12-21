# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BLASER(nn.Module):
    def __init__(
        self,
        idim: int,
        odim: int,
        nhid: tp.List,
        dropout: float,
        use_gpu: bool,
        activation: str,
        input_form: str,
        norm_emb: bool,
        output_act: bool,
    ):
        super(BLASER, self).__init__()
        self.use_gpu = use_gpu
        self.input_form = input_form
        self.dropout = dropout
        self.idim = idim
        self.odim = odim
        self.norm_emb = norm_emb
        self.nhid = nhid
        self.activation = activation
        self.output_act = output_act

        if input_form == "comet":
            idim = 6 * idim
        else:
            raise Exception(f"Unrecognized input format: {input_form}")

        modules = []
        if len(self.nhid) > 0:
            if dropout > 0:
                modules.append(nn.Dropout(p=dropout))
            nprev = idim
            for nh in self.nhid:
                if nh > 0:
                    modules.append(nn.Linear(nprev, nh))
                    nprev = nh
                    if activation == "TANH":
                        modules.append(nn.Tanh())
                    elif activation == "RELU":
                        modules.append(nn.ReLU())
                    else:
                        raise Exception(f"Unrecognized activation: {activation}")
                    if dropout > 0:
                        modules.append(nn.Dropout(p=dropout))
            modules.append(nn.Linear(nprev, odim))
            if output_act:
                modules.append(nn.Tanh())
        else:
            modules.append(nn.Linear(idim, odim))
        self.mlp = nn.Sequential(*modules)
        logger.info(self.mlp)

        if self.use_gpu:
            self.mlp = self.mlp.cuda()

    @property
    def filename(self):
        return f"i{self.idim}.o{self.odim}.dp{self.dropout}.nhid{','.join([str(x) for x in self.nhid])}.ip{self.input_form}"

    def save(self, output_dir: Path):
        save_path = output_dir / f"{self.filename}.pt"
        torch.save(self.state_dict(), save_path)
        logger.info(f"model saved to {save_path.resolve()}")
        save_path = output_dir / f"{self.filename}.config"
        torch.save(
            {
                "input_form": self.input_form,
                "dropout": self.dropout,
                "idim": self.idim,
                "odim": self.odim,
                "nhid": self.nhid,
                "activation": self.activation,
                "norm_emb": self.norm_emb,
                "output_act": self.output_act,
            },
            save_path,
        )
        logger.info(f"config saved to {save_path.resolve()}")

    def load_from_ckpt_file(self, ckpt_path: Path):
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(ckpt, strict=True)
        logger.info(f"model loaded from {ckpt_path.resolve()}")
        if self.use_gpu:
            self.cuda()

    def forward(self, src: torch.tensor, ref: torch.tensor, mt: torch.tensor):
        proc = self._process_input(
            self._norm_vec(src), self._norm_vec(ref), self._norm_vec(mt)
        )
        return self.mlp(proc)

    def _norm_vec(self, emb: torch.tensor):
        if self.norm_emb:
            return F.normalize(emb)
        else:
            return emb

    def _process_input(self, src: torch.tensor, ref: torch.tensor, mt: torch.tensor):
        if self.input_form == "comet":
            processed = torch.cat(
                [
                    ref,
                    mt,
                    src * mt,
                    ref * mt,
                    torch.absolute(mt - src),
                    torch.absolute(mt - ref),
                ],
                dim=-1,
            )
        return processed.cuda() if self.use_gpu else processed


def unsupervised_blaser(
    src: torch.tensor, ref: tp.Optional[torch.tensor], mt: torch.tensor, use_gpu: bool
):
    if use_gpu:
        mt = mt.cuda()
        ref = ref.cuda() if ref is not None else None
        src = src.cuda()
    if ref is not None:
        return (F.cosine_similarity(mt, ref) + F.cosine_similarity(mt, src)).unsqueeze(
            1
        )
    else:
        return (F.cosine_similarity(mt, src)).unsqueeze(1)
