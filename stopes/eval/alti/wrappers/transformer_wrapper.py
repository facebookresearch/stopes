# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code was adapted from the repository https://github.com/mt-upc/transformer-contributions-nmt by Javier Ferrando.


import inspect
import typing as tp
from collections import defaultdict
from functools import partial

import einops
import torch
import torch.nn.functional as F
from fairseq.hub_utils import GeneratorHubInterface
from fairseq.models.transformer import TransformerModel
from fairseq.modules import MultiheadAttention

INFTY = 1e9  # TODO: decrease this number if the model is in half precision


class FairseqTransformerHub(GeneratorHubInterface):
    """A wrapper around the translation model that computes its token attributions."""

    ATTN_MODULES = ["encoder.self_attn", "decoder.self_attn", "decoder.encoder_attn"]

    def __init__(self, cfg, task, models, change_tokens=False):
        if "change_tokens" in inspect.signature(super().__init__).parameters:
            # in the NLLB branch of Fairseq, the vocabulary is modified by hub_utils, but we do not want this
            super().__init__(cfg, task, models, change_tokens=change_tokens)
        else:
            if (
                cfg.task
                and hasattr(cfg.task, "langs")
                and isinstance(cfg.task.langs, list)
            ):
                cfg.task.langs = ",".join(cfg.task.langs)
        super().__init__(cfg, task, models)
        self.eval()
        # self.to("cuda" if torch.cuda.is_available() else "cpu")
        self.fix_multihead_attention()

    @classmethod
    def from_pretrained(
        cls, checkpoint_dir, checkpoint_file, data_name_or_path, **kwargs
    ):
        hub_interface = TransformerModel.from_pretrained(
            checkpoint_dir, checkpoint_file, data_name_or_path, **kwargs
        )
        return cls(hub_interface.cfg, hub_interface.task, hub_interface.models)

    def fix_multihead_attention(self):
        """this is a trick to avoid computing F.multi_head_attention_forward, which does not save qkv outputs"""
        for k, v in self.named_modules():
            if isinstance(v, MultiheadAttention):
                v.skip_embed_dim_check = True

    def decode2(self, tensor, dictionary, as_string=False):
        tok = []
        for token in torch.squeeze(tensor):
            tok.append(dictionary[token])
        if as_string:
            return "".join(tok).replace("▁", " ")
        else:
            return tok

    def get_sample(self, split, index):

        if split not in self.task.datasets.keys():
            self.task.load_dataset(split)

        src_tensor = self.task.dataset(split)[index]["source"]
        src_tok = self.decode2(src_tensor, self.task.src_dict)
        src_sent = self.decode2(src_tensor, self.task.src_dict, as_string=True)

        tgt_tensor = self.task.dataset(split)[index]["target"]
        # get_sample returns tensor [..., </s>]
        # we need [</s>, ...] to feed into the decoder
        tgt_tensor = torch.cat([torch.tensor([tgt_tensor[-1]]), tgt_tensor[:-1]])
        tgt_tok = self.decode2(tgt_tensor, self.task.tgt_dict)
        tgt_sent = self.decode2(tgt_tensor, self.task.tgt_dict, as_string=True)

        return {
            "src_tok": src_tok,
            "src_tensor": src_tensor,
            "tgt_tok": tgt_tok,
            "tgt_tensor": tgt_tensor,
            "src_sent": src_sent,
            "tgt_sent": tgt_sent,
        }

    def get_interactive_sample(
        self, i, test_set_dir, src, tgt, tokenizer, hallucination=None
    ):
        """Get interactive sample from tokenized and original word files."""

        test_src_bpe = f"{test_set_dir}/test.{tokenizer}.{src}"
        test_tgt_bpe = f"{test_set_dir}/test.{tokenizer}.{tgt}"
        test_src_word = f"{test_set_dir}/test.{src}"
        test_tgt_word = f"{test_set_dir}/test.{tgt}"

        with open(test_src_bpe, encoding="utf-8") as fbpe:
            # BPE source sentences
            src_bpe_sents = fbpe.readlines()
        with open(test_tgt_bpe, encoding="utf-8") as fbpe:
            # BPE target sentences
            tgt_bpe_sents = fbpe.readlines()
        with open(test_src_word, encoding="utf-8") as fword:
            # Original source sentences
            src_word_sents = fword.readlines()
        with open(test_tgt_word, encoding="utf-8") as fword:
            # Original target sentences
            tgt_word_sents = fword.readlines()

        src_eos_id = self.task.src_dict.eos_index
        tgt_eos_id = self.task.tgt_dict.eos_index

        src_tok_str = src_bpe_sents[
            i
        ].strip()  # removes leading and trailing whitespaces
        src_tok = src_tok_str.split() + [self.task.src_dict[src_eos_id]]

        # removes leading and trailing whitespaces and add EOS
        tgt_tok_str = tgt_bpe_sents[i].strip()
        tgt_tok = [self.task.tgt_dict[tgt_eos_id]] + tgt_tok_str.split()

        # Add token to beginning of source sentence
        if hallucination is not None:
            src_tok = [hallucination] + ["▁" + src_tok[0]] + src_tok[1:]
            # tgt_tok = ['<pad>'] + ['▁'+ tgt_tok[0]] + tgt_tok[1:]
        # src_tensor = torch.tensor([self.src_dict.index(t) for t in src_tok] + [eos_id])
        # tgt_tensor = torch.tensor([eos_id] + [self.tgt_dict.index(t) for t in tgt_tok])

        src_tensor = torch.tensor([self.src_dict.index(t) for t in src_tok])
        tgt_tensor = torch.tensor([self.tgt_dict.index(t) for t in tgt_tok])

        if test_src_word and test_tgt_word:
            src_word_sent = src_word_sents[i]
            tgt_word_sent = tgt_word_sents[i]

            return {
                "src_word_sent": src_word_sent,
                "src_tok": src_tok,
                "src_tok_str": src_tok_str,
                "src_tensor": src_tensor,
                "tgt_word_sent": tgt_word_sent,
                "tgt_tok": tgt_tok,
                "tgt_tok_str": tgt_tok_str,
                "tgt_tensor": tgt_tensor,
            }

        return {
            "src_word_sent": None,
            "src_tok": src_tok,
            "src_tok_str": src_tok_str,
            "src_tensor": src_tensor,
            "tgt_word_sent": None,
            "tgt_tok": tgt_tok,
            "tgt_tok_str": tgt_tok_str,
            "tgt_tensor": tgt_tensor,
        }

    def parse_module_name(self, module_name):
        """Returns (enc_dec, layer, module)"""
        parsed_module_name = module_name.split(".")
        if not isinstance(parsed_module_name, list):
            parsed_module_name = [parsed_module_name]

        if len(parsed_module_name) < 1 or len(parsed_module_name) > 3:
            raise AttributeError(f"'{module_name}' unknown")

        if len(parsed_module_name) > 1:
            try:
                parsed_module_name[1] = int(parsed_module_name[1])
            except ValueError:
                parsed_module_name.insert(1, None)
            if len(parsed_module_name) < 3:
                parsed_module_name.append(None)
        else:
            parsed_module_name.extend([None, None])

        return parsed_module_name

    def get_module(self, module_name):
        e_d, l, m = self.parse_module_name(module_name)
        module = getattr(self.models[0], e_d)
        if l is not None:
            module = module.layers[l]
            if m is not None:
                module = getattr(module, m)
        else:
            if m is not None:
                raise AttributeError(f"Cannot get'{module_name}'")

        return module

    def trace_forward(self, src_tensor, tgt_tensor):
        r"""Forward-pass through the model.
        Args:
            src_tensor (`tensor`):
                Source sentence tensor.
            tgt_tensor (`tensor`):
                Target sentence tensor (teacher forcing).
        Returns:
            model_output ('tuple'):
                output of the model.
            log_probs:
                log probabilities output by the model.
            encoder_output ('dict'):
                dictionary with 'encoder_out', 'encoder_padding_mask', 'encoder_embedding',
                                'encoder_states', 'src_tokens', 'src_lengths', 'attn_weights'.
            layer_inputs:
                dictionary with the input of the modules of the model.
            layer_outputs:
                dictionary with the input of the modules of the model.
        """
        with torch.no_grad():

            layer_inputs = defaultdict(list)
            layer_outputs = defaultdict(list)

            def save_activation(name, mod, inp, out):
                layer_inputs[name].append(inp)
                layer_outputs[name].append(out)

            handles = {}

            for name, layer in self.named_modules():
                handles[name] = layer.register_forward_hook(
                    partial(save_activation, name)
                )

            src_tensor = src_tensor.unsqueeze(0).to(self.device)
            tgt_tensor = tgt_tensor.unsqueeze(0).to(self.device)

            model_output, encoder_out = self.models[0](
                src_tensor,
                src_tensor.size(-1),
                tgt_tensor,
            )

            log_probs = self.models[0].get_normalized_probs(
                model_output, log_probs=True, sample=None
            )

            for k, v in handles.items():
                handles[k].remove()

            return model_output, log_probs, encoder_out, layer_inputs, layer_outputs

    def normalize_contrib(self, x, mode=None, temperature=0.5, resultant_norm=None):
        """Normalization applied to each row of the layer-wise contributions."""
        if mode == "min_max":
            # Min-max normalization
            x_min = x.min(-1, keepdim=True)[0]
            x_max = x.max(-1, keepdim=True)[0]
            x_norm = (x - x_min) / (x_max - x_min)
            x_norm = x_norm / x_norm.sum(dim=-1, keepdim=True)
        elif mode == "max_min":
            x = -x
            # Min-max normalization
            x_min = x.min(-1, keepdim=True)[0]
            x_max = x.max(-1, keepdim=True)[0]
            x_norm = (x_max - x) / (x_max - x_min)
            # x_norm = x_norm / x_norm.sum(dim=-1, keepdim=True)
        elif mode == "softmax":
            # Softmax
            x_norm = F.softmax(x / temperature, dim=-1)
        elif mode == "sum_one":
            # Sum one
            x_norm = x / x.sum(dim=-1, keepdim=True)
        elif mode == "min_sum":
            # Minimum value selection
            if resultant_norm == None:
                x_min = x.min(-1, keepdim=True)[0]
                x_norm = x + torch.abs(x_min)
                x_norm = x_norm / x_norm.sum(dim=-1, keepdim=True)
            else:
                x_norm = x + torch.abs(resultant_norm.unsqueeze(1))
                x_norm = torch.clip(x_norm, min=0)
                x_norm = x_norm / x_norm.sum(dim=-1, keepdim=True)
        elif mode is None:
            x_norm = x
        else:
            raise AttributeError(f"Unknown normalization mode '{mode}'")
        return x_norm

    def _get_attn_weights_module(self, layer_outputs, module_name):
        enc_dec_, l, attn_module_ = self.parse_module_name(module_name)

        attn_module = self.get_module(module_name)
        num_heads = attn_module.num_heads
        head_dim = attn_module.head_dim

        k = layer_outputs[f"models.0.{enc_dec_}.layers.{l}.{attn_module_}.k_proj"][0]
        q = layer_outputs[f"models.0.{enc_dec_}.layers.{l}.{attn_module_}.q_proj"][0]

        q, k = map(
            lambda x: einops.rearrange(
                x, "t b (n_h h_d) -> (b n_h) t h_d", n_h=num_heads, h_d=head_dim
            ),
            (q, k),
        )

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if enc_dec_ == "decoder" and attn_module_ == "self_attn":
            tri_mask = torch.triu(torch.ones_like(attn_weights), 1).bool()
            attn_weights[tri_mask] = -INFTY

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = einops.rearrange(
            attn_weights, "(b n_h) t_q t_k -> b n_h t_q t_k", n_h=num_heads
        )
        return attn_weights

    def _get_contributions_module(
        self, layer_inputs, layer_outputs, contrib_type, module_name
    ):
        # Get info about module: encoder, decoder, self_attn, cross-attn
        enc_dec_, l, attn_module_ = self.parse_module_name(module_name)
        # Get info about LN (Pre-LN or Post-LN)
        if enc_dec_ == "encoder":
            pre_layer_norm = self.cfg["model"].encoder_normalize_before
        else:
            pre_layer_norm = self.cfg["model"].decoder_normalize_before

        attn_w = self._get_attn_weights_module(
            layer_outputs, module_name
        )  # (batch_size, num_heads, src:len, src_len)

        def l_transform(x, w_ln):
            """Computes mean and performs hadamard product with ln weight (w_ln) as a linear transformation."""
            ln_param_transf = torch.diag(w_ln)
            ln_mean_transf = torch.eye(w_ln.size(0)).to(w_ln.device) - 1 / w_ln.size(
                0
            ) * torch.ones_like(ln_param_transf).to(w_ln.device)

            out = torch.einsum(
                "... e , e f , f g -> ... g", x, ln_mean_transf, ln_param_transf
            )
            return out

        attn_module = self.get_module(module_name)
        w_o = attn_module.out_proj.weight
        b_o = attn_module.out_proj.bias

        ln = self.get_module(f"{module_name}_layer_norm")
        w_ln = ln.weight.data
        b_ln = ln.bias
        eps_ln = ln.eps

        ## LN2
        ln2 = self.get_module(f"{enc_dec_}.{l}.final_layer_norm")
        w_ln2 = ln.weight.data
        b_ln2 = ln.bias
        eps_ln2 = ln.eps

        in_q = layer_inputs[f"models.0.{enc_dec_}.layers.{l}.{attn_module_}.q_proj"][0][
            0
        ].transpose(0, 1)
        in_v = layer_inputs[f"models.0.{enc_dec_}.layers.{l}.{attn_module_}.v_proj"][0][
            0
        ].transpose(0, 1)
        in_res = layer_inputs[
            f"models.0.{enc_dec_}.layers.{l}.{attn_module_}_layer_norm"
        ][0][0].transpose(0, 1)
        ##
        in_res2 = layer_inputs[f"models.0.{enc_dec_}.layers.{l}.final_layer_norm"][0][
            0
        ].transpose(0, 1)

        if "self_attn" in attn_module_:
            # TODO: check the speed here
            if pre_layer_norm:
                residual_ = torch.einsum(
                    "sk,bsd->bskd", torch.eye(in_res.size(1)).to(in_res.device), in_res
                )
            else:
                residual_ = torch.einsum(
                    "sk,bsd->bskd", torch.eye(in_q.size(1)).to(in_res.device), in_q
                )
        else:
            if pre_layer_norm:
                residual_ = in_res
            else:
                residual_ = in_q

        v = attn_module.v_proj(in_v)
        v = einops.rearrange(
            v,
            "b t_v (n_h h_d) -> b n_h t_v h_d",
            n_h=attn_module.num_heads,
            h_d=attn_module.head_dim,
        )

        w_o = einops.rearrange(
            w_o,
            "out_d (n_h h_d) -> n_h h_d out_d",
            n_h=attn_module.num_heads,
        )

        # this one consumes unnecessary memory, replace it with an equivalent matmul
        # attn_v_wo = torch.einsum("b h q k , b h k e , h e f -> b q k f", attn_w, v, w_o)
        attn_v_wo = torch.matmul(
            attn_w.permute((0, 3, 2, 1)),  # bhqk-> bkqh
            torch.matmul(v, w_o).permute(0, 2, 1, 3),  # bhkf -> bkhf
        ).permute([0, 2, 1, 3])

        # Add residual
        if "self_attn" in attn_module_:
            out_qv_pre_ln = attn_v_wo + residual_
        # Concatenate residual in cross-attention (as another value vector)
        else:
            out_qv_pre_ln = torch.cat((attn_v_wo, residual_.unsqueeze(-2)), dim=2)

        # Assert MHA output + residual is equal to 1st layer normalization input
        out_q_pre_ln = out_qv_pre_ln.sum(-2) + b_o

        #### NEW
        if pre_layer_norm == False:
            # In post-ln we compare with the input of the first layernorm
            out_q_pre_ln_th = layer_inputs[
                f"models.0.{enc_dec_}.layers.{l}.{attn_module_}_layer_norm"
            ][0][0].transpose(0, 1)
        else:
            if "encoder" in enc_dec_:
                # Encoder (self-attention) -> final_layer_norm
                out_q_pre_ln_th = layer_inputs[
                    f"models.0.{enc_dec_}.layers.{l}.final_layer_norm"
                ][0][0].transpose(0, 1)
            else:
                if "self_attn" in attn_module_:
                    # Self-attention decoder -> encoder_attn_layer_norm
                    out_q_pre_ln_th = layer_inputs[
                        f"models.0.{enc_dec_}.layers.{l}.encoder_attn_layer_norm"
                    ][0][0].transpose(0, 1)
                else:
                    # Cross-attention decoder -> final_layer_norm
                    out_q_pre_ln_th = layer_inputs[
                        f"models.0.{enc_dec_}.layers.{l}.final_layer_norm"
                    ][0][0].transpose(0, 1)
        #### NEW

        # if pre_layer_norm:
        #     if 'encoder' in enc_dec_:
        #          # Encoder (self-attention) -> final_layer_norm
        #         out_q_pre_ln_th = layer_inputs[f"models.0.{enc_dec_}.layers.{l}.final_layer_norm"][0][0].transpose(0, 1)

        #     else:
        #         if "self_attn" in attn_module_:
        #             # Self-attention decoder -> encoder_attn_layer_norm
        #             out_q_pre_ln_th = layer_inputs[f"models.0.{enc_dec_}.layers.{l}.encoder_attn_layer_norm"][0][0].transpose(0, 1)
        #         else:
        #             # Cross-attention decoder -> final_layer_norm
        #             out_q_pre_ln_th = layer_inputs[f"models.0.{enc_dec_}.layers.{l}.final_layer_norm"][0][0].transpose(0, 1)

        # else:
        #     # In post-ln we compare with the input of the first layernorm
        #     out_q_pre_ln_th = layer_inputs[f"models.0.{enc_dec_}.layers.{l}.{attn_module_}_layer_norm"][0][0].transpose(0, 1)

        assert (
            torch.dist(out_q_pre_ln_th, out_q_pre_ln).item()
            < 1e-3 * out_q_pre_ln.numel()
        )

        if pre_layer_norm:
            transformed_vectors = out_qv_pre_ln
            resultant = out_q_pre_ln
        else:
            ln_std_coef = 1 / (out_q_pre_ln_th + eps_ln).std(-1).view(
                1, -1, 1
            ).unsqueeze(
                -1
            )  # (batch,src_len,1,1)
            transformed_vectors = (
                l_transform(out_qv_pre_ln, w_ln) * ln_std_coef
            )  # (batch,src_len,tgt_len,embed_dim)
            dense_bias_term = (
                l_transform(b_o, w_ln) * ln_std_coef
            )  # (batch,src_len,1,embed_dim)
            attn_output = transformed_vectors.sum(dim=2)  # (batch,seq_len,embed_dim)
            resultant = (
                attn_output + dense_bias_term.squeeze(2) + b_ln
            )  # (batch,seq_len,embed_dim)

            # Assert resultant (decomposed attention block output) is equal to the real attention block output
            out_q_th_2 = layer_outputs[
                f"models.0.{enc_dec_}.layers.{l}.{attn_module_}_layer_norm"
            ][0].transpose(0, 1)
            assert torch.dist(out_q_th_2, resultant).item() < 1e-3 * resultant.numel()

        if contrib_type == "l1":
            contributions = -F.pairwise_distance(
                transformed_vectors, resultant.unsqueeze(2), p=1
            )  # TODO: try to decrease memory consumption here, if possible
            resultants_norm = torch.norm(torch.squeeze(resultant), p=1, dim=-1)

        elif contrib_type == "l2":
            contributions = -F.pairwise_distance(
                transformed_vectors, resultant.unsqueeze(2), p=2
            )
            resultants_norm = torch.norm(torch.squeeze(resultant), p=2, dim=-1)
            # resultants_norm=None
        elif contrib_type == "koba":
            contributions = torch.norm(transformed_vectors, p=2, dim=-1)
            return contributions, None
        else:
            raise ArgumentError(f"contribution_type '{contrib_type}' unknown")

        return contributions, resultants_norm

    def get_contributions(
        self,
        src_tensor: torch.Tensor,
        tgt_tensor: torch.Tensor,
        contrib_type: str = "l1",
        norm_mode: str = "min_sum",
    ) -> tp.Dict[str, torch.Tensor]:
        r"""
        Get contributions for each ATTN_MODULE: 'encoder.self_attn', 'decoder.self_attn', 'decoder.encoder_attn.
        Args:
            src_tensor ('tensor' ()):
                Source sentence tensor.
            tgt_tensor ('tensor' ()):
                Target sentence tensor (teacher forcing).
            contrib_type ('str', defaults to 'l1' (Ferrando et al ., 2022)):
                Type of layer-wise contribution measure: 'l1', 'l2', 'koba' (Kobayashi et al ., 2021) or 'attn_w'.
            norm_mode ('str', defaults to 'min_sum' (Ferrando et al ., 2022)):
                Type of normalization applied to layer-wise contributions: 'min_sum', 'min_max', 'sum_one', 'softmax'.
        Returns:
            Dictionary with elements in ATTN_MODULE as keys, and tensor with contributions (batch_size, num_layers, src_len, tgt_len) as values.
        """
        contributions_all = defaultdict(list)
        _, _, _, layer_inputs, layer_outputs = self.trace_forward(
            src_tensor, tgt_tensor
        )

        if contrib_type == "attn_w":
            f = partial(self._get_attn_weights_module, layer_outputs)
        else:
            f = partial(
                self._get_contributions_module,
                layer_inputs,
                layer_outputs,
                contrib_type,
            )

        for attn in self.ATTN_MODULES:
            enc_dec_, _, attn_module_ = self.parse_module_name(attn)
            enc_dec = self.get_module(enc_dec_)

            for l in range(len(enc_dec.layers)):
                if contrib_type == "attn_w":
                    contributions = f(attn.replace(".", f".{l}."))
                    resultant_norms = None
                    contributions = contributions.sum(1)
                    if norm_mode != "sum_one":
                        print("Please change the normalization mode to sum one")
                else:
                    contributions, resultant_norms = f(attn.replace(".", f".{l}."))
                contributions = self.normalize_contrib(
                    contributions, norm_mode, resultant_norm=resultant_norms
                ).unsqueeze(1)
                # Mask upper triangle of decoder self-attention matrix (and normalize)
                # if attn == 'decoder.self_attn':
                #     contributions = torch.tril(torch.squeeze(contributions,dim=1))
                #     contributions = contributions / contributions.sum(dim=-1, keepdim=True)
                #     contributions = contributions.unsqueeze(1)
                contributions_all[attn].append(contributions)
        contributions_all = {
            k: torch.cat(v, dim=1) for k, v in contributions_all.items()
        }
        return contributions_all

    def get_contribution_rollout(
        self,
        src_tensor: torch.Tensor,
        tgt_tensor: torch.Tensor,
        contrib_type: str = "l1",
        norm_mode: str = "min_sum",
        **contrib_kwargs,
    ) -> tp.Dict[str, tp.Any]:  # TODO: define the output type correctly
        """Given the source and target tensors, compute token attributions:
        for encoder, decoder, across them, and in total."""

        # Rollout encoder
        c_roll = defaultdict(list)
        enc_sa = "encoder.self_attn"

        contribs = self.get_contributions(
            src_tensor, tgt_tensor, contrib_type, norm_mode=norm_mode, **contrib_kwargs
        )

        # Compute contributions rollout encoder self-attn
        enc_self_attn_contributions = torch.squeeze(contribs[enc_sa])
        layers, _, _ = enc_self_attn_contributions.size()
        enc_self_attn_contributions_mix = compute_joint_attention(
            enc_self_attn_contributions
        )
        c_roll[enc_sa] = enc_self_attn_contributions_mix.detach().clone()
        # repeat num_layers times

        # Get last layer relevances w.r.t input
        relevances_enc_self_attn = enc_self_attn_contributions_mix[-1]
        relevances_enc_self_attn = relevances_enc_self_attn.unsqueeze(0).repeat(
            layers, 1, 1
        )

        dec_sa = "decoder.self_attn"
        dec_ed = "decoder.encoder_attn"

        # Compute joint cross + self attention
        self_dec_contributions = torch.squeeze(contribs[dec_sa])
        cross_contributions = torch.squeeze(contribs[dec_ed])
        self_dec_contributions = (
            self_dec_contributions.transpose(1, 2)
            * cross_contributions[:, :, -1].unsqueeze(1)
        ).transpose(1, 2)
        joint_self_cross_contributions = torch.cat(
            (cross_contributions[:, :, :-1], self_dec_contributions), dim=-1
        )

        contributions_full_rollout = rollout(
            joint_self_cross_contributions, relevances_enc_self_attn[-1]
        )

        c_roll["total"] = contributions_full_rollout
        c_roll[dec_ed] = cross_contributions

        return c_roll


def compute_joint_attention(att_mat: torch.Tensor) -> torch.Tensor:
    """Compute attention rollout given contributions or attn weights + residual."""

    joint_attentions = torch.zeros(att_mat.size()).to(att_mat.device)
    layers = joint_attentions.shape[0]
    joint_attentions = att_mat[0].unsqueeze(0)
    for i in range(1, layers):
        C_roll_new = torch.matmul(att_mat[i], joint_attentions[i - 1])
        joint_attentions = torch.cat([joint_attentions, C_roll_new.unsqueeze(0)], dim=0)
    return joint_attentions


def rollout(C: torch.Tensor, C_enc_out: torch.Tensor) -> torch.Tensor:
    """Contributions rollout whole Transformer-NMT model.
    Args:
        C: [num_layers, cross_attn;self_dec_attn] contributions decoder layers
        C_enc_out: encoder rollout last layer
    """
    src_len = C.size(2) - C.size(1)
    tgt_len = C.size(1)

    C_sa_roll = C[:, :, -tgt_len:]  # Self-att decoder, only has 1 layer
    C_ed_roll = torch.einsum(  # encoder rollout*cross-attn
        "lie , ef -> lif",
        C[:, :, :src_len],  # Cross-att
        C_enc_out,  # Encoder rollout
    )

    C_roll = torch.cat(
        [C_ed_roll, C_sa_roll], dim=-1
    )  # [(cross_attn*encoder rollout);self_dec_attn]
    C_roll_new_accum = C_roll[0].unsqueeze(0)

    for i in range(1, len(C)):
        C_sa_roll_new = torch.einsum(
            "ij , jk -> ik",
            C_roll[i, :, -tgt_len:],  # Self-att dec
            C_roll_new_accum[i - 1, :, -tgt_len:],  # Self-att (prev. roll)
        )
        C_ed_roll_new = (
            torch.einsum(
                "ij , jk -> ik",
                C_roll[i, :, -tgt_len:],  # Self-att dec
                C_roll_new_accum[i - 1, :, :src_len],  # Cross-att (prev. roll)
            )
            + C_roll[i, :, :src_len]
        )  # Cross-att

        C_roll_new = torch.cat([C_ed_roll_new, C_sa_roll_new], dim=-1)
        C_roll_new = C_roll_new / C_roll_new.sum(dim=-1, keepdim=True)

        C_roll_new_accum = torch.cat([C_roll_new_accum, C_roll_new.unsqueeze(0)], dim=0)

    return C_roll_new_accum
