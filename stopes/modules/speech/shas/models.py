# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This source code is adapted with minimal changes
# from https://github.com/mt-upc/SHAS

import torch
from transformers import Wav2Vec2Model


class SegmentationFrameClassifer(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        n_transformer_layers: int = 0,
        init_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if n_transformer_layers:
            self.transformer = torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(
                    d_model,
                    nhead=nhead,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=n_transformer_layers,
            )

        self.dropout = torch.nn.Dropout(p=init_dropout)
        self.layer_norm = torch.nn.LayerNorm(d_model)

        self.classification_layer = torch.nn.Linear(d_model, 1)

    def forward(
        self, x: torch.FloatTensor, attention_mask: torch.Tensor
    ) -> torch.FloatTensor:

        attention_mask = ~attention_mask.bool()

        x = self.dropout(x)

        if hasattr(self, "transformer"):
            x = self.transformer(x, src_key_padding_mask=attention_mask)

        logits = self.classification_layer(self.layer_norm(x))

        return logits.squeeze(-1)


def prepare_wav2vec(model_name: str, layer_id: int, main_device: torch.device = None):
    """
    Loads a wav2vec 2.0 model from transformers library
    and keeps only certain layers by replacing the rest with identities

    Args:
        model_name (str): wav2vec 2.0 model name in transformers
        layer_id (int): layers (including and) above this one are replaced with identities
        main_device (torch.DeviceObjType, optional): Defaults to None.

    Returns:
        wav2vec 2.0 model, reduced and in eval mode
    """

    # init pre-trained wav2vec
    wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)

    if main_device is not None:
        wav2vec_model = wav2vec_model.to(main_device)

    # keep only some layers of wav2vec
    wav2vec_model.encoder.layers = torch.nn.modules.container.ModuleList(
        [layer for i, layer in enumerate(wav2vec_model.encoder.layers) if i < layer_id]
    )
    # also remove final layer norm since it corresponds to the 24th-layer
    # the input of the classifier will be normalized again
    wav2vec_model.encoder.layer_norm = torch.nn.Identity()

    wav2vec_model.eval()

    return wav2vec_model
