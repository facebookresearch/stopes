# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import itertools
import logging
import typing as tp
from collections import defaultdict

import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class AwesomeAlignWrapper:
    def __init__(
        self,
        base_model="aneuraz/awesome-align-with-co",
        align_layer=8,
        threshold=1e-3,
        temperature=1,
        device="cuda",
    ):
        self.base_model = base_model
        self.align_layer = align_layer
        self.threshold = threshold
        self.temperature = temperature
        if "cuda" in device and not torch.cuda.is_available():
            logger.warning(
                f"Device `{device}` was not found when initializaing AwesomeAlignWrapper; using cpu instead."
            )
            device = "cpu"
        self.device = device
        self.model = AutoModel.from_pretrained(self.base_model).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)

    def embed_words(self, words: tp.List[str]):
        tokens = [self.tokenizer.tokenize(word) for word in words]
        wids = [self.tokenizer.convert_tokens_to_ids(x) for x in tokens]
        ids = self.tokenizer.prepare_for_model(
            list(itertools.chain(*wids)),
            return_tensors="pt",
            model_max_length=self.tokenizer.model_max_length,
            truncation=True,
        )["input_ids"].to(self.device)
        with torch.inference_mode():
            out = self.model(ids.unsqueeze(0), output_hidden_states=True)[2][
                self.align_layer
            ][0, 1:-1]
        sub2word_map = [i for i, word_list in enumerate(tokens) for x in word_list]
        max_len = self.tokenizer.max_len_single_sentence
        if max_len:
            if len(sub2word_map) > max_len:
                logger.warning(
                    f"Truncating from {len(sub2word_map)} to {max_len} tokens for {words}"
                )
                sub2word_map = sub2word_map[:max_len]
        return out, sub2word_map

    def align_single_pair(
        self, src: tp.List[str], tgt: tp.List[str]
    ) -> tp.Set[tp.Tuple[int, int]]:
        """Find the alignments that score above the threshold in both directions"""
        if len(src) == 0 or len(tgt) == 0:
            return set()
        with torch.inference_mode():
            out_src, sub2word_map_src = self.embed_words(src)
            out_tgt, sub2word_map_tgt = self.embed_words(tgt)

            dot_prod = (
                torch.matmul(out_src, out_tgt.transpose(-1, -2)) * self.temperature
            )

            softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
            softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

            softmax_inter = (softmax_srctgt > self.threshold) * (
                softmax_tgtsrc > self.threshold
            )
        align_subwords = torch.nonzero(softmax_inter, as_tuple=False).tolist()
        align_words = {
            (sub2word_map_src[i], sub2word_map_tgt[j]) for i, j in align_subwords
        }
        return align_words

    def force_align_single_pair(
        self, src: tp.List[str], tgt: tp.List[str], warmup: float = 1e-3
    ) -> tp.Tuple[tp.Set[tp.Tuple[int, int]], tp.Set[tp.Tuple[int, int]]]:
        """
        Find the alignments that score above the threshold in both directions ("sure alignments")
        and an extra set of alignments that cover all unaligned words ("possible alignments").
        """
        if len(src) == 0 or len(tgt) == 0:
            return set(), set()
        with torch.inference_mode():
            out_src, sub2word_map_src = self.embed_words(src)
            out_tgt, sub2word_map_tgt = self.embed_words(tgt)
            dot_prod = (
                torch.matmul(out_src, out_tgt.transpose(-1, -2)) * self.temperature
            )
            softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
            softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

            # first stage: sure alignment
            softmax_inter = (softmax_srctgt > self.threshold) * (
                softmax_tgtsrc > self.threshold
            )
            align_subwords = torch.nonzero(softmax_inter, as_tuple=False).tolist()
            align_words = {
                (sub2word_map_src[i], sub2word_map_tgt[j]) for i, j in align_subwords
            }

            # second stage: forced alignment of the rest
            unaligned_src = set(range(len(src))).difference({i for i, j in align_words})
            unaligned_tgt = set(range(len(tgt))).difference({j for i, j in align_words})

            extra_alignments = set()
            if unaligned_src or unaligned_tgt:
                w2s_src = reverse_mapping(sub2word_map_src)
                w2s_tgt = reverse_mapping(sub2word_map_tgt)
                softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod * warmup)
                softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod * warmup)

                # zero out already aligned words
                for i, j in align_words:
                    softmax_srctgt[w2s_src[i], :] = -1
                    softmax_tgtsrc[:, w2s_tgt[j]] = -1

                # iteratively align the subwords pair with the strongest linkage among the unaligned ones
                for _ in range(len(unaligned_src) + len(unaligned_tgt)):
                    max_src = softmax_srctgt.max()
                    max_tgt = softmax_tgtsrc.max()
                    if max_src <= 0 and max_tgt <= 0:
                        break
                    if max_src > max_tgt:
                        si, sj = torch.nonzero(softmax_srctgt == max_src)[0].tolist()
                    else:
                        si, sj = torch.nonzero(softmax_tgtsrc == max_tgt)[0].tolist()

                    # map the subwords to words and mark them as aligned
                    i, j = sub2word_map_src[si], sub2word_map_tgt[sj]
                    extra_alignments.add((i, j))
                    softmax_srctgt[w2s_src[i], :] = -1
                    softmax_tgtsrc[:, w2s_tgt[j]] = -1

        return align_words, extra_alignments


def reverse_mapping(s2w: tp.List[int]) -> tp.Dict[int, tp.List[int]]:
    result = defaultdict(list)
    for sw, w in enumerate(s2w):
        result[w].append(sw)
    return dict(result)
