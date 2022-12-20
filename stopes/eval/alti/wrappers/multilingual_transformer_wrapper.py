# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code was adapted from the repository https://github.com/mt-upc/transformer-contributions-nmt by Javier Ferrando.

import copy
import logging
from collections import defaultdict
from functools import partial
from typing import Dict, List

import torch
from fairseq import utils
from fairseq.data.multilingual.multilingual_utils import LangTokStyle, get_lang_tok
from fairseq.models.transformer import TransformerModel
from omegaconf import open_dict

from .transformer_wrapper import FairseqTransformerHub

logger = logging.getLogger(__name__)


class FairseqMultilingualTransformerHub(FairseqTransformerHub):
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir,
        checkpoint_file,
        data_name_or_path,
        source_lang,
        target_lang,
        lang_pairs,
        fixed_dictionary,
    ):
        hub_interface = TransformerModel.from_pretrained(
            checkpoint_dir,
            checkpoint_file,
            data_name_or_path,
            source_lang=source_lang,
            target_lang=target_lang,
            lang_pairs=lang_pairs,
            fixed_dictionary=fixed_dictionary,
        )
        return cls(hub_interface.cfg, hub_interface.task, hub_interface.models)

    def change_langs(self, src_lang, tgt_lang):
        """Change the source and target languages (may be useful for interactive translation)"""
        prev_lang = self.task.args.source_lang
        self.task.args.target_lang = tgt_lang
        self.task.data_manager.dicts[tgt_lang] = self.task.data_manager.dicts[prev_lang]
        self.task.args.source_lang = src_lang
        self.task.data_manager.dicts[src_lang] = self.task.data_manager.dicts[prev_lang]

    def decode2(self, tensor, dictionary, as_string=False):
        tok = []
        for token in torch.squeeze(tensor):
            tok.append(dictionary[token])
        # tok = dictionary.string(tensor,'sentencepiece').split()
        if as_string:
            return "".join(tok).replace("▁", " ")
        else:
            return tok

    def get_sample(self, split, index):

        if split not in self.task.datasets.keys():
            self.task.load_dataset(split)

        src_tensor = self.task.dataset(split)[index]["source"]
        src_tok = self.decode2(src_tensor, self.task.source_dictionary)
        src_sent = self.decode2(src_tensor, self.task.source_dictionary, as_string=True)

        tgt_tensor = self.task.dataset(split)[index]["target"]
        # get_sample returns tensor [lang_tok, ..., </s>]
        # we need [</s>, lang_tok, ...] to feed into the decoder
        tgt_tensor = torch.cat([torch.tensor([tgt_tensor[-1]]), tgt_tensor[:-1]])
        tgt_tok = self.decode2(tgt_tensor, self.task.target_dictionary)
        tgt_sent = self.decode2(tgt_tensor, self.task.target_dictionary, as_string=True)

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

        src_word_sent = src_word_sents[i]
        tgt_word_sent = tgt_word_sents[i]

        # removes leading and trailing whitespaces
        src_tok_str = src_bpe_sents[i].strip()
        tgt_tok_str = tgt_bpe_sents[i].strip()

        def prepare_input_encoder(hub, tok_sentence):
            max_positions = utils.resolve_max_positions(
                hub.task.max_positions(),
                *[model.max_positions() for model in hub.models],
            )

            def encode_fn(x):
                return x

            batch = make_batches(
                tok_sentence, hub.cfg, hub.task, max_positions, encode_fn
            )
            src_tensor = next(batch).src_tokens
            src_tok = [hub.task.target_dictionary[t] for t in src_tensor[0]]
            return src_tok, src_tensor[0]  # first element in batch

        def prepare_input_decoder(hub, tok_sentence):
            tok_sentence = tok_sentence.split()
            if self.cfg["model"].decoder_langtok:
                lang = hub.task.args.langtoks["main"][1]
                if lang == "tgt":
                    lang_tok = hub.task.args.target_lang
                else:
                    lang_tok = hub.task.args.source_lang
                lang_tok = get_lang_tok(
                    lang=lang_tok, lang_tok_style=LangTokStyle.multilingual.value
                )
                tgt_tok = (
                    [hub.task.target_dictionary[hub.task.target_dictionary.eos_index]]
                    + [lang_tok]
                    + tok_sentence
                )
            else:
                print("No decoder langtok used")
                tgt_tok = [
                    hub.task.target_dictionary[hub.task.target_dictionary.eos_index]
                ] + tok_sentence
            tgt_tensor = torch.tensor(
                [hub.task.target_dictionary.index(t) for t in tgt_tok]
            )
            return tgt_tok, tgt_tensor

        src_tok, src_tensor = prepare_input_encoder(self, [src_tok_str])
        tgt_tok, tgt_tensor = prepare_input_decoder(self, tgt_tok_str)

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
                dictionary with the input of the modeules of the model.
            layer_outputs:
                dictionary with the input of the modeules of the model.
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

            # tgt_tensor = torch.cat([
            #     torch.tensor([self.task.target_dictionary.eos_index]),
            #     tgt_tensor[:-1]
            # ]).unsqueeze(0).to(self.device)
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

    def generate(
        self,
        tokenized_sentences: List[torch.LongTensor],
        beam: int = 5,
        verbose: bool = False,
        skip_invalid_size_inputs=False,
        inference_step_args=None,
        prefix_allowed_tokens_fn=None,
        **kwargs,
    ) -> List[List[Dict[str, torch.Tensor]]]:
        if torch.is_tensor(tokenized_sentences) and tokenized_sentences.dim() == 1:
            return self.generate(
                tokenized_sentences.unsqueeze(0), beam=beam, verbose=verbose, **kwargs
            )[0]

        # build generator using current args as well as any kwargs
        gen_args = copy.deepcopy(self.cfg.generation)
        with open_dict(gen_args):
            gen_args.beam = beam
            for k, v in kwargs.items():
                setattr(gen_args, k, v)
        generator = self.task.build_generator(
            self.models,
            gen_args,
            # prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )
        inference_step_args = inference_step_args or {}
        results = []
        for batch in self._build_batches(tokenized_sentences, skip_invalid_size_inputs):
            batch = utils.apply_to_sample(lambda t: t.to(self.device), batch)
            translations = self.task.inference_step(
                generator, self.models, batch, **inference_step_args
            )
            for id, hypos in zip(batch["id"].tolist(), translations):
                results.append((id, hypos))

        # sort output to match input order
        outputs = [hypos for _, hypos in sorted(results, key=lambda x: x[0])]

        if verbose:

            def getarg(name, default):
                return getattr(gen_args, name, getattr(self.cfg, name, default))

            for source_tokens, target_hypotheses in zip(tokenized_sentences, outputs):
                src_str_with_unk = self.string(source_tokens)
                logger.info("S\t{}".format(src_str_with_unk))
                for hypo in target_hypotheses:
                    hypo_str = self.decode2(hypo["tokens"], self.task.target_dictionary)
                    logger.info("H\t{}\t{}".format(hypo["score"], hypo_str))
                    logger.info(
                        "P\t{}".format(
                            " ".join(
                                map(
                                    lambda x: "{:.4f}".format(x),
                                    hypo["positional_scores"].tolist(),
                                )
                            )
                        )
                    )
                    if hypo["alignment"] is not None and getarg(
                        "print_alignment", False
                    ):
                        logger.info(
                            "A\t{}".format(
                                " ".join(
                                    [
                                        "{}-{}".format(src_idx, tgt_idx)
                                        for src_idx, tgt_idx in hypo["alignment"]
                                    ]
                                )
                            )
                        )
        return outputs
