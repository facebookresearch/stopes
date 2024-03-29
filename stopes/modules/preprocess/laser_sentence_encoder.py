# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import re
import typing as tp
from collections import namedtuple
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
from fairseq.data.dictionary import Dictionary
from fairseq.models.transformer import Embedding, TransformerEncoder
from fairseq.modules import LayerNorm

from stopes.modules.preprocess.encode_to_npy import EncodeToNPY
from stopes.utils.mining_utils import extract_shard_id

SPACE_NORMALIZER = re.compile(r"\s+")
Batch = namedtuple("Batch", "srcs tokens lengths")

logger = logging.getLogger("laser_encoder")

SPM_MODEL_SUFFIX = ".spm"
SPM_VOCAB_SUFFIX = ".cvocab"


def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


class SentenceEncoder:
    def __init__(
        self,
        model_path,
        max_sentences=None,
        max_tokens=None,
        spm_vocab=None,
        cpu=False,
        fp16=False,
        verbose=False,
        sort_kind="quicksort",
    ):
        self.use_cuda = torch.cuda.is_available() and not cpu
        self.max_sentences = max_sentences
        self.max_tokens = max_tokens
        if self.max_tokens is None and self.max_sentences is None:
            self.max_sentences = 1

        state_dict = torch.load(model_path)
        if "params" in state_dict:
            self.encoder = LaserLstmEncoder(**state_dict["params"])
            self.encoder.load_state_dict(state_dict["model"])
            self.dictionary = state_dict["dictionary"]
            self.prepend_bos = False
            self.left_padding = False
        else:
            self.encoder = LaserTransformerEncoder(state_dict, spm_vocab)
            self.dictionary = self.encoder.dictionary.indices
            self.prepend_bos = state_dict["cfg"]["model"].prepend_bos
            self.left_padding = state_dict["cfg"]["model"].left_pad_source
        del state_dict
        self.bos_index = self.dictionary["<s>"] = 0
        self.pad_index = self.dictionary["<pad>"] = 1
        self.eos_index = self.dictionary["</s>"] = 2
        self.unk_index = self.dictionary["<unk>"] = 3
        if fp16:
            self.encoder.half()
        if self.use_cuda:
            if verbose:
                print(" - transfer encoder to GPU")
            self.encoder.cuda()
        self.encoder.eval()
        self.sort_kind = sort_kind

    def _process_batch(self, batch):
        tokens = batch.tokens
        lengths = batch.lengths
        if self.use_cuda:
            tokens = tokens.cuda()
            lengths = lengths.cuda()

        with torch.no_grad():
            sentemb = self.encoder(tokens, lengths)["sentemb"]
        embeddings = sentemb.detach().cpu().numpy()
        return embeddings

    def _tokenize(self, line):
        tokens = SPACE_NORMALIZER.sub(" ", line).strip().split()
        ntokens = len(tokens)
        if self.prepend_bos:
            ids = torch.LongTensor(ntokens + 2)
            ids[0] = self.bos_index
            for i, token in enumerate(tokens):
                ids[i + 1] = self.dictionary.get(token, self.unk_index)
            ids[ntokens + 1] = self.eos_index
        else:
            ids = torch.LongTensor(ntokens + 1)
            for i, token in enumerate(tokens):
                ids[i] = self.dictionary.get(token, self.unk_index)
            ids[ntokens] = self.eos_index
        return ids

    def _make_batches(self, lines):
        tokens = [self._tokenize(line) for line in lines]
        lengths = np.array([t.numel() for t in tokens])
        indices = np.argsort(-lengths, kind=self.sort_kind)

        def batch(tokens, lengths, indices):
            toks = tokens[0].new_full((len(tokens), tokens[0].shape[0]), self.pad_index)
            if not self.left_padding:
                for i in range(len(tokens)):
                    toks[i, : tokens[i].shape[0]] = tokens[i]
            else:
                for i in range(len(tokens)):
                    toks[i, -tokens[i].shape[0] :] = tokens[i]
            return (
                Batch(srcs=None, tokens=toks, lengths=torch.LongTensor(lengths)),
                indices,
            )

        batch_tokens, batch_lengths, batch_indices = [], [], []
        ntokens = nsentences = 0
        for i in indices:
            if nsentences > 0 and (
                (self.max_tokens is not None and ntokens + lengths[i] > self.max_tokens)
                or (self.max_sentences is not None and nsentences == self.max_sentences)
            ):
                yield batch(batch_tokens, batch_lengths, batch_indices)
                ntokens = nsentences = 0
                batch_tokens, batch_lengths, batch_indices = [], [], []
            batch_tokens.append(tokens[i])
            batch_lengths.append(lengths[i])
            batch_indices.append(i)
            ntokens += tokens[i].shape[0]
            nsentences += 1
        if nsentences > 0:
            yield batch(batch_tokens, batch_lengths, batch_indices)

    def encode_sentences(self, sentences):
        #
        indices = []
        results = []
        for batch, batch_indices in self._make_batches(sentences):
            indices.extend(batch_indices)
            results.append(self._process_batch(batch))
        return np.vstack(results)[np.argsort(indices, kind=self.sort_kind)]


class LaserTransformerEncoder(TransformerEncoder):
    def __init__(self, state_dict, vocab_path):
        self.dictionary = Dictionary.load(vocab_path)
        if any(
            k in state_dict["model"]
            for k in ["encoder.layer_norm.weight", "layer_norm.weight"]
        ):
            self.dictionary.add_symbol("<mask>")
        cfg = state_dict["cfg"]["model"]
        self.sentemb_criterion = cfg.sentemb_criterion
        self.pad_idx = self.dictionary.pad_index
        self.bos_idx = self.dictionary.bos_index
        embed_tokens = Embedding(
            len(self.dictionary),
            cfg.encoder_embed_dim,
            self.pad_idx,
        )
        super().__init__(cfg, self.dictionary, embed_tokens)
        if "decoder.version" in state_dict["model"]:
            self._remove_decoder_layers(state_dict)
        if "layer_norm.weight" in state_dict["model"]:
            self.layer_norm = LayerNorm(cfg.encoder_embed_dim)
        self.load_state_dict(state_dict["model"])

    def _remove_decoder_layers(self, state_dict):
        for key in list(state_dict["model"].keys()):
            if not key.startswith(
                (
                    "encoder.layer_norm",
                    "encoder.layers",
                    "encoder.embed",
                    "encoder.version",
                )
            ):
                del state_dict["model"][key]
            else:
                renamed_key = key.replace("encoder.", "")
                state_dict["model"][renamed_key] = state_dict["model"].pop(key)

    def forward(self, src_tokens, src_lengths):
        encoder_out = super().forward(src_tokens, src_lengths)
        if isinstance(encoder_out, dict):
            x = encoder_out["encoder_out"][0]  # T x B x C
        else:
            x = encoder_out[0]
        if self.sentemb_criterion == "cls":
            cls_indices = src_tokens.eq(self.bos_idx).t()
            sentemb = x[cls_indices, :]
        else:
            padding_mask = src_tokens.eq(self.pad_idx).t().unsqueeze(-1)
            if padding_mask.any():
                x = x.float().masked_fill_(padding_mask, float("-inf")).type_as(x)
            sentemb = x.max(dim=0)[0]
        return {"sentemb": sentemb}


class LaserLstmEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings,
        padding_idx,
        embed_dim=320,
        hidden_size=512,
        num_layers=1,
        bidirectional=False,
        left_pad=True,
        padding_value=0.0,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.padding_idx = padding_idx
        self.embed_tokens = nn.Embedding(
            num_embeddings, embed_dim, padding_idx=self.padding_idx
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad
        self.padding_value = padding_value

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def forward(self, src_tokens, src_lengths):
        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.tolist())

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.data.new(*state_size).zero_()
        c0 = x.data.new(*state_size).zero_()
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outs, padding_value=self.padding_value
        )
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:

            def combine_bidir(outs):
                return torch.cat(
                    [
                        torch.cat([outs[2 * i], outs[2 * i + 1]], dim=0).view(
                            1, bsz, self.output_units
                        )
                        for i in range(self.num_layers)
                    ],
                    dim=0,
                )

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        # Set padded outputs to -inf so they are not selected by max-pooling
        padding_mask = src_tokens.eq(self.padding_idx).t().unsqueeze(-1)
        if padding_mask.any():
            x = x.float().masked_fill_(padding_mask, float("-inf")).type_as(x)

        # Build the sentence embedding by max-pooling over the encoder outputs
        sentemb = x.max(dim=0)[0]

        return {
            "sentemb": sentemb,
            "encoder_out": (x, final_hiddens, final_cells),
            "encoder_padding_mask": encoder_padding_mask
            if encoder_padding_mask.any()
            else None,
        }


class LaserSentenceEncoder(EncodeToNPY):
    """
    spm or bpe preprocessing is first performed before encoding and writing to an output file.
    """

    def __init__(
        self,
        _name: str,
        encoder_model: str,
        spm_model: str,
        outfile_prefix: str,
        input_file: str,
        input_file_idx: int,
        output_dir: str,
        outfile_postfix: str = "",
        spm_vocab: str = None,
        max_sentences: tp.Optional[int] = None,
        max_tokens: int = 12_000,
        stable_sort: bool = False,
        normalize: bool = False,
        fp16: bool = False,
        cpu: bool = False,
        fp16_model: bool = False,
    ) -> None:
        super().__init__(
            outfile_prefix=outfile_prefix,
            outfile_postfix=outfile_postfix,
            input_file=input_file,
            input_file_idx=input_file_idx,
            output_dir=output_dir,
            normalize=normalize,
            fp16=fp16,
        )
        spm_model, spm_vocab = self.gather_spm(
            encoder_model=encoder_model,
            spm_vocab=spm_vocab,
            spm_model=spm_model,
        )
        self.spm_tokenizer = spm.SentencePieceProcessor()
        self.spm_tokenizer.Load(spm_model)
        self.encoder = SentenceEncoder(
            encoder_model,
            max_sentences=max_sentences,
            max_tokens=max_tokens,
            spm_vocab=spm_vocab,
            sort_kind="mergesort" if stable_sort else "quicksort",
            cpu=cpu,
        )

    @staticmethod
    def gather_spm(
        encoder_model: str,
        spm_vocab: tp.Optional[str],
        spm_model: tp.Optional[str],
    ) -> tp.Tuple[str, str]:
        if not spm_model:
            encoder_path = Path(encoder_model)
            spm_model = str(encoder_path.with_suffix(SPM_MODEL_SUFFIX))
            logger.warning(f"SPM model not specified. Defaulting to: {spm_model}")
        if not spm_vocab:
            spm_vocab = str(Path(spm_model).with_suffix(SPM_VOCAB_SUFFIX))
            logger.warning(f"SPM vocab not specified. Defaulting to: {spm_vocab}")
        assert os.path.isfile(spm_model) and os.path.isfile(
            spm_vocab
        ), f"Couldn't find spm model: {spm_model} and vocab: {spm_vocab}"
        return spm_model, spm_vocab

    def name_output_file(self) -> str:
        shard_idx = extract_shard_id(self.input_file, default=self.input_file_idx)

        return os.path.abspath(
            os.path.join(
                self.output_dir,
                f"{self.outfile_prefix}.{shard_idx:03d}.{self.outfile_postfix}",
            )
        )

    def encode_to_np(
        self, lines_with_number: tp.Iterator[tp.Tuple[int, str]]
    ) -> np.ndarray:
        # TODO: it's not very efficient to tokenize, then join, then latter split again
        tokenized_sents = [
            " ".join(self.spm_tokenizer.EncodeAsPieces(line))
            for (_, line) in lines_with_number
        ]
        return self.encoder.encode_sentences(tokenized_sents)
