# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
The core code for applying BLASER-like models for comparing two audios.

A recommended entry point is the `compare_audio_pairs` function.

If you want to run it in scale, please use stopes.pipelines.speech.compare_audio.
"""
import logging
import typing as tp
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from stopes.core import utils
from stopes.modules.speech.audio_load_utils import parallel_audio_read

logger = logging.getLogger(__name__)

INFTY = 1e6


class Comparator(nn.Module):
    """This is a module for trainable comparison of vectors.
    Possible input formats are:
        - "comet": source, translation, reference
        - "qe": source, translation
    """

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
        trainable_pooler: bool = False,
    ):
        super(Comparator, self).__init__()
        self.use_gpu = use_gpu
        self.input_form = input_form
        self.dropout = dropout
        self.idim = idim
        self.odim = odim
        self.norm_emb = norm_emb
        self.nhid = nhid
        self.activation = activation
        self.output_act = output_act
        self.trainable_pooler = trainable_pooler

        self.pooler: tp.Optional[nn.Module]

        if self.trainable_pooler:
            self.pooler = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(idim, 1))
        else:
            self.pooler = None

        if input_form == "comet":
            idim = 6 * idim
        elif input_form == "qe":
            idim = 4 * idim
        else:
            raise Exception(f"Unrecognized input format: {input_form}")

        modules: tp.List[nn.Module] = []
        if len(self.nhid) > 0:
            if dropout > 0:
                modules.append(nn.Dropout(p=dropout))
            nprev = idim
            for hidden_size in self.nhid:
                if hidden_size > 0:
                    modules.append(nn.Linear(nprev, hidden_size))
                    nprev = hidden_size
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
            if self.pooler is not None:
                self.pooler = self.pooler.cuda()

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
                "trainable_pooler": self.trainable_pooler,
            },
            save_path,
        )
        logger.info(f"config saved to {save_path.resolve()}")

    @classmethod
    def load(cls, config_path: tp.Union[str, Path], use_gpu=False):
        """Load both model and config"""
        config_path = Path(config_path).resolve()
        # If the file points to the directory, try to find the config in it
        if config_path.is_dir():
            for filename in config_path.iterdir():
                if filename.is_file() and filename.suffix == ".config":
                    config_path = filename
                    break
        model_config = torch.load(config_path)
        model = cls(**model_config, use_gpu=use_gpu)
        model.load_from_ckpt_file(config_path.with_suffix(".pt"))
        return model

    def load_from_ckpt_file(self, ckpt_path: Path):
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(ckpt, strict=True)
        logger.info(f"model loaded from {ckpt_path.resolve()}")
        if self.use_gpu:
            self.cuda()

    def forward(
        self,
        src: torch.Tensor,
        ref: tp.Optional[torch.Tensor] = None,
        mt: tp.Optional[torch.Tensor] = None,
        src_mask: tp.Optional[torch.Tensor] = None,
        ref_mask: tp.Optional[torch.Tensor] = None,
        mt_mask: tp.Optional[torch.Tensor] = None,
    ):
        src, ref, mt = (
            self._pool(src, src_mask),
            self._pool(ref, ref_mask),
            self._pool(mt, mt_mask),
        )
        proc = self._process_input(
            self._norm_vec(src), self._norm_vec(ref), self._norm_vec(mt)
        )
        return self.mlp(proc)

    def _norm_vec(self, emb: tp.Optional[torch.Tensor]):
        if self.norm_emb and emb is not None:
            return F.normalize(emb)
        return emb

    def _pool(
        self, emb: tp.Optional[torch.Tensor], mask: tp.Optional[torch.Tensor] = None
    ):
        if self.pooler is None or emb is None:
            return emb
        if self.use_gpu:
            emb = emb.cuda()

        preds = self.pooler(emb).squeeze(-1)  # (batch_size, seq_len)
        if mask is None:  # infer the mask automatically from entries with all zeros
            mask = 1 - ((emb == 0).sum(-1) / emb.shape[-1]).to(torch.int)
        preds = preds - (1 - mask.to(preds.device)) * INFTY
        weights = torch.softmax(preds, -1)
        return torch.einsum("bsd, bs -> bd", emb, weights)

    def _process_input(
        self,
        src: torch.Tensor,
        ref: tp.Optional[torch.Tensor] = None,
        mt: tp.Optional[torch.Tensor] = None,
    ):
        if self.input_form == "comet":
            if ref is None or mt is None:
                raise ValueError(
                    "Comparator with `comet` input form requires non-empty `ref` and `mt` inputs."
                )
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
        if self.input_form == "qe":
            if mt is None:
                mt = ref
            if mt is None:
                raise ValueError(
                    "Comparator with `qe` input form requires non-empty `ref` or `mt` inputs."
                )
            processed = torch.cat(
                [
                    src,
                    mt,
                    src * mt,
                    torch.absolute(mt - src),
                ],
                dim=-1,
            )
        return processed.cuda() if self.use_gpu else processed


def encode_audios(
    audio_paths,
    model_name_or_path="facebook/wav2vec2-large-xlsr-53",
    use_cuda=True,
    pick_layer=7,
    avg_pool=True,
    batch_size=16,
    num_process=1,
    fex=None,
    model=None,
    sampling_factor=16,
    progress=True,
):
    """Compute wav2vec embeddings of audios. Optionally, pick a specific layer and/or avgpool along tokens.
    The output is a single tensor (in case of pooling) or a list of tensors.
    """
    if fex is None:
        fex = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
    if model is None:
        model = Wav2Vec2Model.from_pretrained(model_name_or_path)
        if use_cuda:
            model.cuda()

    itr = parallel_audio_read(
        lines=audio_paths,
        column_offset=0,
        sampling_factor=sampling_factor,
        num_process=num_process,
        collapse_channels=True,
    )
    if progress:
        itr = tqdm(itr, total=len(audio_paths))

    results: tp.List[torch.Tensor] = []
    for batch in utils.batch(itr, batch_size=batch_size):
        audios = [wav for line_no, wav in batch]
        y = fex(
            audios,
            sampling_rate=sampling_factor * 1000,
            padding=True,
            return_tensors="pt",
        )
        y = {k: v.to(model.device) for k, v in y.items()}

        with torch.inference_mode():
            if pick_layer is not None and isinstance(model, Wav2Vec2Model):
                # do not infer the upper layers that we won't need
                all_layers = [x for x in model.encoder.layers]
                model.encoder.layers = torch.nn.ModuleList(all_layers[: pick_layer + 1])
                model_out = model(**y, output_hidden_states=True)
                model.encoder.layers = torch.nn.ModuleList(all_layers)
            else:
                model_out = model(**y, output_hidden_states=True)
            if pick_layer is not None:
                # (batch_size, 1, length, dim)
                states = model_out.hidden_states[pick_layer].unsqueeze(1)
            else:
                # (batch_size, depth, length, dim)
                states = torch.cat(model_out.hidden_states).transpose(0, 1)

            attention_mask = model._get_feature_vector_attention_mask(
                model_out.extract_features.shape[1],
                y["attention_mask"],
                add_adapter=False,
            )
            states = torch.einsum("bdlh,bl->bdlh", states, attention_mask)
            if avg_pool:
                states = states.mean(-2)  # (batch_size, depth, dim)
        results.append(states.cpu())
    if avg_pool:
        return torch.cat(results)  # (n, depth, dim)
    # without avg_pool, length is different, so we cannot concatenate without padding
    return results


def get_batches(
    src: torch.Tensor,
    ref: tp.Optional[torch.Tensor] = None,
    mt: tp.Optional[torch.Tensor] = None,
    batch_size: tp.Optional[int] = None,
) -> tp.List[tp.List[tp.Optional[torch.Tensor]]]:
    """
    Split src, ref and mt tensors into batches.
    `ref` and/or `mt` can be None; in this case, the batches will contain None for them.
    """
    n = src.shape[0]
    batches = []
    if batch_size and batch_size < n:
        for idx in range(0, n, batch_size):
            batches.append(
                [
                    None if emb is None else emb[idx : idx + batch_size]
                    for emb in [src, ref, mt]
                ]
            )
    else:
        batches = [[src, ref, mt]]
    return batches


def get_model_pred(
    model,
    src: torch.Tensor,
    ref: tp.Optional[torch.Tensor] = None,
    mt: tp.Optional[torch.Tensor] = None,
    use_gpu: bool = True,
    batch_size: tp.Optional[int] = None,
):
    results = []
    with torch.no_grad():
        for src, ref, mt in get_batches(src, ref, mt, batch_size=batch_size):  # type: ignore
            if isinstance(model, torch.nn.Module):
                model.train(mode=False)
                pred = model(src, ref, mt)
                model.train(mode=True)
            else:
                pred = model(src, ref, mt, use_gpu)
            results.append(pred)
    return torch.cat(results)


def compare_audio_pairs(
    src_paths: tp.List[str],
    tgt_paths: tp.List[str],
    comparator_path: tp.Union[str, Path],
    encoder_path: str = "facebook/wav2vec2-large-xlsr-53",
    use_cuda: bool = True,
    batch_size: int = 16,
    pick_layer: int = 7,
    symmetrize: bool = False,
    num_process: tp.Optional[int] = 1,
    progress: bool = True,
) -> tp.List[float]:
    """Vectorize the audios, compare the representations, return the list of predicted scores"""
    logger.info("Loading vectorizer")
    fex = Wav2Vec2FeatureExtractor.from_pretrained(encoder_path)
    encoder = Wav2Vec2Model.from_pretrained(encoder_path)
    model = Comparator.load(comparator_path, use_gpu=use_cuda)
    if use_cuda:
        encoder.cuda()

    if batch_size is None:
        batch_size = len(src_paths)

    logger.info("Embedding source")
    src_emb: torch.Tensor = encode_audios(
        src_paths,
        fex=fex,
        model=encoder,
        batch_size=batch_size,
        pick_layer=pick_layer,
        progress=progress,
        num_process=num_process,
    )[:, 0]
    logger.info("Embedding target")
    tgt_emb: torch.Tensor = encode_audios(
        tgt_paths,
        fex=fex,
        model=encoder,
        batch_size=batch_size,
        pick_layer=pick_layer,
        progress=progress,
        num_process=num_process,
    )[:, 0]
    logger.info("Comparing source and target embedding")
    preds = (
        get_model_pred(
            model,
            src=src_emb,
            mt=tgt_emb,
            use_gpu=model.use_gpu,
            batch_size=batch_size,
        )[:, 0]
        .cpu()
        .numpy()
    )
    if symmetrize:
        preds2 = (
            get_model_pred(
                model,
                src=tgt_emb,
                mt=src_emb,
                use_gpu=model.use_gpu,
                batch_size=batch_size,
            )[:, 0]
            .cpu()
            .numpy()
        )
        preds = (preds2 + preds) / 2
    return list(preds)
