# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import functools
import json
import logging
import os
import typing as tp
from abc import ABC
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.parameter import Parameter

from stopes.speech.tokenizers import SpeechTokenizer, SpeechTokenizerConfig

logger = logging.getLogger(__name__)


class Kmeans(torch.nn.Module):
    @staticmethod
    def load_model(km_path: Path) -> "Kmeans":
        file_extension = os.path.splitext(km_path)[-1]
        if file_extension == ".npy":
            km_model = np.load(km_path)
            centroids_numpy = km_model.transpose()
        else:
            # The model have been created with joblib special pickler
            # and also need to be loaded with the same pickler.
            # I think it would be nice to use pytorch here,
            # instead of introducing a new kind of python serialization format.
            km_model = joblib.numpy_pickle.load(km_path)
            centroids_numpy = km_model.cluster_centers_.transpose()
        return Kmeans(torch.from_numpy(centroids_numpy))

    def __init__(self, centroids: torch.Tensor):
        super().__init__()

        self.centroids = Parameter(centroids)
        self.centroid_norm = Parameter((centroids**2).sum(0, keepdims=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dist: torch.Tensor = (
            x.pow(2).sum(1, keepdim=True)
            - 2 * torch.matmul(x, self.centroids)
            + self.centroid_norm
        )
        return dist.argmin(dim=-1)


class KMeansHifiGANSpeechTokenizer(SpeechTokenizer, ABC):
    """The speech tokenizer that uses KMeans to make discrete units and GANVocoder to convert
    them back to wave form"""

    import fairseq
    from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder

    config: SpeechTokenizerConfig

    def __init__(self, config: tp.Dict[str, tp.Any]):
        # Map the user-friendly encoder args to the attribute `speech_encoder``
        # in SpeechTokenizerConfig
        if hasattr(config, "encoder"):
            _config: tp.Any = OmegaConf.to_container(config, resolve=True)
            _config["speech_encoder"] = _config.pop("encoder")
            config = OmegaConf.create(_config)

        super().__init__(config, SpeechTokenizerConfig)
        self.layer = int(self.config.speech_encoder.feature_layer)
        self.should_normalize = False

    @functools.cached_property
    def encoder(self):
        if len(str(self.config.speech_encoder.checkpoint)) > 0:
            (
                [model],
                _,
                task,
            ) = self.fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [self.config.speech_encoder.checkpoint]
            )
            self.should_normalize = task.cfg.normalize
            model.eval()
            return self._sanitize(model)

    def __repr__(self) -> str:
        if hasattr(self.config, "name") and len(self.config.name) > 0:
            return self.config.name
        _name = self.__class__.__name__
        if hasattr(self.config, "lang"):
            _name = f"{_name}_{self.config.lang}"
        if hasattr(self.config, "feature_layer"):
            _name = f"{_name}_{self.config.feature_layer}"
        if hasattr(self.config, "km_size"):
            _name = f"{_name}_{self.config.km_size}"
        return _name

    @functools.cached_property
    def kmeans(self):
        assert self.config.units.checkpoint, "Cannot load checkpoint for kmeans"
        kmeans = Kmeans.load_model(Path(self.config.units.checkpoint))
        kmeans.eval()
        return self._sanitize(kmeans)

    @functools.cached_property
    def vocoder(self):
        with open(self.config.vocoder.config_path) as f:
            vocoder_cfg = json.load(f)

        # Many nn.Modules such as slow_conv2d_cpu are not implemented for HalfTensor yet
        fp16 = self.config.gpu and self.config.fp16

        _vocoder = self.CodeHiFiGANVocoder(
            checkpoint_path=self.config.vocoder.checkpoint,
            model_cfg=vocoder_cfg,
            fp16=fp16,
        )

        _vocoder = _vocoder.to("cuda") if self.config.gpu else _vocoder.to("cpu")
        return _vocoder

    @torch.no_grad()
    def encode(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        """Convert one audio sample into a stream of discrete units.

        Internally, this function calls extract_features() and to_unit()
        in sequence.
        Args:
           input: tensor of audio wave form (batch x channels x frame_len or channels x frame_len)
        """
        assert len(input.shape) >= 2

        x = self._sanitize(input)  # type: torch.Tensor
        frame_len = x.shape[-1]
        if len(x.shape) > 2:
            num_channels, channel_dim = x.shape[1], 1
        else:
            num_channels, channel_dim = x.shape[0], 0

        if num_channels > 1:
            x = x.mean(dim=channel_dim, keepdim=True)

        # Force to call encoder to update `should_normalize` attr
        assert isinstance(self.encoder, torch.nn.Module)

        if self.should_normalize:
            # normalize over frames
            x = F.layer_norm(x, [frame_len])

        # batch x channels x frames -> batch x frames
        if len(x.shape) > 2:
            x = x.squeeze(channel_dim)
        feats = []
        for start in range(0, frame_len, self.config.max_frames_chunk):
            x_chunk = x[:, start : start + self.config.max_frames_chunk]
            feat_chunk = self.extract_features(x_chunk)
            feats.append(feat_chunk)

        # x can be features x encoding_dim or batch x features x encoding_dim
        x = torch.cat(feats, 1)

        # batch x features x encoding_dim -> 1 x flatten_features x encoding_dim (to call kmeans)
        if len(x.shape) > 2:
            *bach_n_features, encoding_dim = x.shape
            x = x.view(1, -1, encoding_dim)
        else:
            # No need to preserved batch shape
            bach_n_features = None

        # At this point first dim should always be 1: If it is a channel, it
        # was averaged out in the beginning. If it is a batch, it is flatten
        x = x.squeeze(0)

        units = self.to_unit(x)
        if bach_n_features:
            units = units.reshape(*bach_n_features)

        return units

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._sanitize(x)

    def to_unit(self, x: torch.Tensor) -> torch.Tensor:
        return self.kmeans.forward(self._sanitize(x))

    @torch.no_grad()
    def decode(self, units: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self._sanitize(units)

        # A dummy index for multi-speaker vocoders
        spkr = torch.tensor([[0]])
        if hasattr(self.config, "gpu") and self.config.gpu:
            spkr = spkr.cuda()

        if len(x.shape) == 1:
            return self.vocoder({"code": x})
        elif len(x.shape) > 2:
            raise NotImplementedError(
                "Not support decoding units of dimension larger than 2"
            )
        elif x.size(0) == 1:
            # TODO: How to handle multi-speaker scenarios ?
            if self.vocoder.model.multispkr:
                logger.warning(
                    "Tokenizer does not support multispeaker vocoding. Use with care"
                )
            return self.vocoder({"code": x.long(), "spkr": spkr}).unsqueeze(0)

        else:
            codes = [
                self.vocoder({"code": x[i].long(), "spkr": spkr})
                for i in range(x.size(0))
            ]
            return torch.stack(codes, 0).unsqueeze(1)


class XlsrSpeechTokenizer(KMeansHifiGANSpeechTokenizer):
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._sanitize(x)

        if self.layer > 1:
            feats = self.encoder.extract_features(
                x, padding_mask=None, mask=False, layer=self.layer - 1
            )
        else:
            feats = self.encoder.extract_features(x, padding_mask=None, mask=False)
        return feats["x"]  # type: ignore[no-any-return]


class HuBertSpeechTokenizer(KMeansHifiGANSpeechTokenizer):
    def validate_model_config(self, *inputs, **kwargs):
        model_cls = type(self.encoder).__name__
        assert "Hubert" in model_cls

    def __post_init__(self, *inputs, **kwargs):
        super().__post_init__(*inputs, **kwargs)
        self.layer = self.layer if self.layer is not None else None

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.extract_features(x, output_layer=self.layer)[0]
