# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# These utils likely require the internal `seamless_main` branch of fairseq

from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path

import sentencepiece as spm
import torch
from fairseq.data import Dictionary
from fairseq.models.speech_to_speech.xm_transformer_unity2 import (
    set_default_alignment_encoder_args,
)
from fairseq.models.speech_to_text.xm_transformer import build_embedding
from fairseq.models.text_to_speech.modules.aligner import AlignmentEncoder

from stopes.modules.speech.utils import parse_audio


def strip_aligner_prefix(state_key, find_key: str):
    # a.b.c.find_key.d.e.f -> d.e.f
    assert find_key in state_key, f"{find_key} is not found in the key: {state_key}"
    split_keys = state_key.split(".")
    aligner_position = split_keys.index(f"{find_key}")
    new_key = split_keys[(aligner_position + 1) :]
    return ".".join(new_key)


def load_aligner_args(ckpt_obj):
    model_cfg = vars(ckpt_obj["cfg"]["model"])
    parser = ArgumentParser()
    aligner_args = parser.parse_args(args=[])
    set_default_alignment_encoder_args(aligner_args)
    for k, v in vars(aligner_args).items():
        if k in model_cfg:
            model_cfg_val = model_cfg[k]
            setattr(aligner_args, k, model_cfg_val)
    return aligner_args


class Aligner(object):
    def __init__(
        self,
        unity2_ckpt_filepath,
        char_dict_path,
        unit_dict_path,
        char_spm_path,
        lang_token_dict_path=None,
        default_token="▁",
    ):
        state_dicts = Aligner.make_aligner_ckpt(unity2_ckpt_filepath)
        self.build_aligner_model(
            state_dicts,
            char_dict_path,
            unit_dict_path,
            char_spm_path,
            lang_token_dict_path,
        )
        self.default_token = default_token

    @classmethod
    def make_aligner_ckpt(cls, unity2_ckpt_filepath):
        assert Path(unity2_ckpt_filepath).exists()
        unity2_ckpt = torch.load(unity2_ckpt_filepath)
        aligner_args = load_aligner_args(unity2_ckpt)
        aligner_state = OrderedDict()
        text_emb_state = OrderedDict()
        unit_emb_state = OrderedDict()

        for k, v in unity2_ckpt["model"].items():
            if "alignment_encoder" in k:
                aligner_state[f"{strip_aligner_prefix(k, 'alignment_encoder')}"] = v
            if "embed_tokens_text" in k:
                text_emb_state[f"{strip_aligner_prefix(k, 'embed_tokens_text')}"] = v
            if "embed_tokens_unit" in k:
                unit_emb_state[f"{strip_aligner_prefix(k, 'embed_tokens_unit')}"] = v

        return {
            "aligner_state": aligner_state,
            "aligner_args": aligner_args,
            "text_emb_state": text_emb_state,
            "unit_emb_state": unit_emb_state,
        }

    def build_aligner_model(
        self,
        loaded_obj_dict,
        char_dict_path,
        unit_dict_path,
        char_spm_path,
        lang_token_dict_path=None,
    ):
        self.args = loaded_obj_dict["aligner_args"]
        self.aligner = AlignmentEncoder(
            self.args.alignment_encoder_embed_dim,
            self.args.alignment_encoder_embed_dim,
            text_layers=self.args.alignment_encoder_text_layers,
            feat_layers=self.args.alignment_encoder_feat_layers,
            dropout=self.args.alignment_encoder_dropout,
            temperature=self.args.alignment_encoder_temperature,
            reduction_factor=self.args.alignment_encoder_reduction_factor,
            prior_end_steps=self.args.alignment_encoder_prior_end_steps,
        )

        self.aligner.training = False
        self.aligner.eval()
        self.aligner.num_updates = 0

        self.aligner.load_state_dict(loaded_obj_dict["aligner_state"])
        assert Path(char_spm_path).exists()
        self.char_spm = spm.SentencePieceProcessor(model_file=char_spm_path)
        self.char_dict = Dictionary.load(char_dict_path)
        self.unit_dict = Dictionary.load(unit_dict_path)

        # this is needed for multilingual aligner
        if lang_token_dict_path is not None:
            with open(lang_token_dict_path) as f:
                langs = f.readlines()
                for lang in langs:
                    lang = lang.strip()

                    self.char_dict.add_symbol(lang)

        self.text_emb = build_embedding(
            self.char_dict, self.args.alignment_encoder_embed_dim
        )
        msg = self.text_emb.load_state_dict(loaded_obj_dict["text_emb_state"])
        self.text_emb.eval()
        self.unit_emb = build_embedding(
            self.unit_dict, self.args.alignment_encoder_embed_dim
        )
        msg = self.unit_emb.load_state_dict(loaded_obj_dict["unit_emb_state"])
        self.unit_emb.eval()
        self.text_spm = spm.SentencePieceProcessor(model_file=char_spm_path)

    def compute_alignment(self, text_seq: str, unit_seq: str, append_sow: bool = False):
        # prepare text
        spm_seq = self.char_spm.encode(text_seq, out_type=str)
        if append_sow:
            # repeat the first silence token in the end
            spm_seq.append(spm_seq[0])
        # if the input text is empty, we replace it with a token that denotes silence;
        # otherwise, the computations below would crash.
        if len(spm_seq) == 0 and self.default_token is not None:
            spm_seq = [self.default_token]
        char_ids = torch.tensor([self.char_dict.index(c) for c in spm_seq])
        text_emb = self.text_emb(char_ids)
        # prepare audio
        unit_tok = unit_seq.split()
        unit_ids = torch.tensor([self.unit_dict.index(u) for u in unit_tok])
        unit_emb = self.unit_emb(unit_ids)

        (attn_lprob, attn_hard_dur, _,) = self.aligner.forward(
            text_emb[None, :, :],
            unit_emb[None, :, :],
            torch.tensor([text_emb.size(0)]),
            torch.tensor([unit_emb.size(0)]),
        )

        return attn_hard_dur, attn_lprob, spm_seq, char_ids, unit_ids
