# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Classes and methods related to Encodec (speech tokenizers, util functions, etc.)

import dataclasses as dc
import typing as tp
from pathlib import Path

import torch

try:
    from encodec import modules as encodec_m
    from encodec import quantization as encodec_q
    from encodec.model import EncodecModel
    from omegaconf import OmegaConf
except ModuleNotFoundError as err:
    print(
        "Encoder Speech Tokenizer requires Encodec to be installed"
        "(https://github.com/facebookresearch/encodec)"
    )
    raise

from stopes.speech.tokenizers import SpeechTokenizer


@dc.dataclass
class EncodecConfig:
    checkpoint: tp.Union[str, Path]
    target_sample_rate: int
    quantizer_bins: int
    name: str = ""
    target_bandwidths: tp.List[float] = dc.field(default_factory=lambda: [])
    channels: int = -1
    gpu: bool = True
    fp16: bool = False
    max_frames_chunk: int = 16_000_00


class EncodecSpeechTokenizer(SpeechTokenizer):
    """The wrapper of EncodecModel in SpeechTokenizer interface. See
    <https://github.com/facebookresearch/encodec/blob/main/encodec/model.py>"""

    config: EncodecConfig

    def __init__(self, config: tp.Any, name: str = None):
        super().__init__(config, EncodecConfig)

        # Set model to None. Later this can be overriden with an existing
        # model, e.g. EncodecModel.encodec_model_24khz(True)
        self._model: tp.Optional[EncodecModel] = None

    def _load_local_model(self) -> EncodecModel:

        if self.config.gpu:
            state = torch.load(self.config.checkpoint)
        else:
            state = torch.load(self.config.checkpoint, map_location=torch.device("cpu"))
        assert (
            "xp.cfg" in state
        ), f"Could not load compression model from ckpt: {self.config.checkpoint}"
        cfg = state["xp.cfg"]
        cfg = OmegaConf.to_container(cfg, resolve=True)
        assert (
            self.config.channels == -1
            or self.config.channels == cfg["encodec"]["channels"]
        ), f"Invalid channel in checkpoint: expect {self.config.channels}, get {cfg['encodec']['channels']}"

        assert (
            "seanet" in cfg
        ), f"Invalid checkpoint {self.config.checkpoint}: Only SEANet-based Encodec is supported"
        cfg["seanet"].pop("disable_norm_outer_blocks")
        encoder_cfg = cfg["seanet"].pop("encoder")
        decoder_cfg = cfg["seanet"].pop("decoder")
        encoder = encodec_m.SEANetEncoder(**cfg["seanet"], **encoder_cfg)
        decoder = encodec_m.SEANetDecoder(**cfg["seanet"], **decoder_cfg)

        assert (
            "rvq" in cfg
        ), f"Quantization in checkpoint {self.config.checkpoint} is expected but not found"
        cfg["rvq"].pop("q_dropout")
        cfg["rvq"].pop("orthogonal_reg_weight")
        cfg["rvq"].pop("orthogonal_reg_active_codes_only")
        units = encodec_q.ResidualVectorQuantizer(
            **cfg["rvq"],
            dimension=cfg["seanet"]["dimension"],
        )
        model = EncodecModel(
            encoder=encoder,
            decoder=decoder,
            quantizer=units,
            target_bandwidths=self.config.target_bandwidths,
            sample_rate=cfg["encodec"]["sample_rate"],
            channels=cfg["encodec"]["channels"],
            normalize=cfg["encodec"]["renormalize"],
        )
        model.eval()
        model.load_state_dict(state["ema"]["state"]["model"])
        return model

    def _load_public_model(self) -> EncodecModel:
        state = torch.hub.load_state_dict_from_url(
            str(self.config.checkpoint), map_location="cpu"
        )
        model = EncodecModel._get_model(
            target_bandwidths=self.config.target_bandwidths,
            sample_rate=self.config.target_sample_rate,
            channels=self.config.channels,
            causal=False,
            model_norm="time_group_norm",
            audio_normalize=True,
            segment=1.0,
            name=self.config.name,
        )
        model.load_state_dict(state)
        return model

    @property
    def model(self) -> EncodecModel:
        """Load model and config from the checkpoint from stopes hub (public or FAIR)"""
        if not self._model:
            if str(self.config.checkpoint).startswith("http"):
                model = self._load_public_model()
            else:
                model = self._load_local_model()
            assert len(str(self.config.checkpoint)) > 0

            self._model = tp.cast(EncodecModel, self._sanitize(model))

        return self._model  # type: ignore[return]

    def __repr__(self):
        if len(self.config.name) > 0 and self.config.name != "encodec":
            return self.config.name
        _name = (
            f"encodec_{self.config.target_sample_rate}hz_{self.config.quantizer_bins}"
        )
        if self.config.target_bandwidths:
            _name = _name + "_bw" + "-".join(map(str, self.config.target_bandwidths))
        return _name

    def encode(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        """encode the waveform to discrete units"""
        # channels x frame_len (single audio), or
        # Batch x channels x frame_len (batched audio)
        if len(input.shape) < 3:  # add dummy batch dim for encodec
            x = input.unsqueeze(0)
        else:
            x = input

        x = self._sanitize(x)
        num_channels, frame_len = x.shape[1], x.shape[-1]
        if num_channels > 1:
            x = x.mean(dim=1, keepdim=True)

        feats = []
        with torch.no_grad():
            for start in range(0, frame_len, self.config.max_frames_chunk):
                x_chunk = x[:, :, start : start + self.config.max_frames_chunk]
                encoded_frames = self.model.encode(x_chunk)
                feats += [encoded[0] for encoded in encoded_frames]
        return torch.cat(feats, dim=-1)

    @torch.no_grad()
    def decode(self, units: torch.Tensor, **kwargs) -> torch.Tensor:
        # TODO: Test with very long audio data
        frames: tp.List[tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]] = [(units, None)]  # type: ignore
        return self.model.decode(frames).squeeze(0)
