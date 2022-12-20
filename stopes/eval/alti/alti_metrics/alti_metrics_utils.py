# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import numpy as np
import torch
from tqdm.auto import trange

from stopes.eval.alti.alignment import align
from stopes.eval.alti.wrappers.multilingual_transformer_wrapper import (
    FairseqMultilingualTransformerHub,
)
from stopes.eval.alti.wrappers.transformer_wrapper import FairseqTransformerHub


def binarize_pair(
    hub: FairseqTransformerHub,
    input_text: str,
    output_text: str,
    src_lang: tp.Optional[str] = None,
    tgt_lang: tp.Optional[str] = None,
    max_length: tp.Optional[int] = None,
) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a pair of texts into source, target and predicted tensors with all special tokens."""

    st = hub.binarize(hub.apply_bpe(hub.tokenize(input_text)))
    pt = hub.binarize(hub.apply_bpe(hub.tokenize(output_text)))

    # adding language tokens
    if isinstance(hub, FairseqMultilingualTransformerHub):
        t = hub.task
        dic = t.data_manager.get_source_dictionary(t.args.source_lang)
        src_lang = f"__{src_lang or t.args.source_lang}__"
        tgt_lang = f"__{tgt_lang or t.args.target_lang}__"
        src_lang_tok, tgt_lang_tok = dic.index(src_lang), dic.index(tgt_lang)
        # src_lang_tok = t.data_manager.get_decoder_langtok(t.args.source_lang, t.args.langtoks['main'][0])
        # tgt_lang_tok = t.data_manager.get_decoder_langtok(t.args.target_lang, t.args.langtoks['main'][1])
        st = torch.cat([torch.tensor([src_lang_tok]), st])
        pt = torch.cat([torch.tensor([tgt_lang_tok]), pt])

    tt = torch.cat(
        [torch.tensor([hub.task.target_dictionary.eos()]), pt[:-1]]
    )  # yes, it is eos \_00_/

    if max_length is not None:
        st, pt, tt = st[:max_length], pt[:max_length], tt[:max_length]

    return st, pt, tt


def get_loss(
    hub: FairseqTransformerHub,
    input_text: str,
    output_text: str,
    src_lang: tp.Optional[str] = None,
    tgt_lang: tp.Optional[str] = None,
) -> tp.Dict[str, float]:
    """Using an ALTI hub, use its model to compute loss for a given text pair."""

    st, pt, tt = binarize_pair(
        hub, input_text, output_text, src_lang=src_lang, tgt_lang=tgt_lang
    )

    with torch.inference_mode():
        logits, out = hub.models[0].forward(
            src_tokens=st.unsqueeze(0).to(hub.device),
            prev_output_tokens=tt.unsqueeze(0).to(hub.device),
            src_lengths=torch.tensor(st.shape).to(hub.device),
        )
        loss_fct = torch.nn.CrossEntropyLoss()
        log_loss = loss_fct(logits.view(-1, logits.size(-1)), pt.to(hub.device)).item()
    return {"loss_avg": log_loss, "loss_sum": log_loss * len(pt)}


def compute_alti_nllb(
    hub: FairseqTransformerHub,
    input_text: str,
    output_text: str,
    src_lang: tp.Optional[str] = None,
    tgt_lang: tp.Optional[str] = None,
    max_length: tp.Optional[int] = None,
    contrib_type="l1",
    norm_mode="min_sum",
) -> tp.Tuple[np.ndarray, tp.List[str], tp.List[str], tp.List[str]]:
    """Compute ALTI+ matrix and all tokenized sentences using an NLLB-like seq2seq model."""
    src_tensor, pred_tensor, tgt_tensor = binarize_pair(
        hub,
        input_text,
        output_text,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        max_length=max_length,
    )

    source_sentence = hub.decode2(src_tensor, hub.task.source_dictionary)
    target_sentence = hub.decode2(tgt_tensor, hub.task.target_dictionary)
    predicted_sentence = hub.decode2(pred_tensor, hub.task.target_dictionary)

    with torch.inference_mode():
        all_alti = hub.get_contribution_rollout(
            src_tensor, tgt_tensor, contrib_type=contrib_type, norm_mode=norm_mode
        )
        token_level_alti = all_alti["total"][-1].detach().cpu().numpy()

    return token_level_alti, source_sentence, target_sentence, predicted_sentence


def alti_to_word(
    token_level_alti: np.ndarray,
    source_sentence: tp.List[str],
    target_sentence: tp.List[str],
    predicted_sentence: tp.List[str],
    eos: str = "</s>",
) -> tp.Tuple[np.ndarray, tp.List[str], tp.List[str], tp.List[str]]:
    """Aggregate token contributions and tokens themselves to the word level."""
    word_level_alti, words_in, words_out = align.contrib_tok2words(
        token_level_alti,
        tokens_in=source_sentence + target_sentence,
        tokens_out=predicted_sentence,
    )
    source_words = words_in[: words_in.index(eos) + 1]
    target_words = words_in[words_in.index(eos) + 1 :]
    return word_level_alti, source_words, target_words, words_out


def entropy(proba: np.ndarray) -> float:
    """Literally the formula of entropy"""
    return np.sum(proba * -np.log(proba))


def compute_alti_metrics(
    alti: np.ndarray,
    source_sentence: tp.List[str],
    target_sentence: tp.List[str],
    predicted_sentence: tp.List[str],
    skip_first: bool = False,
) -> tp.Dict[str, float]:
    """Compute sentence-level metrics of the alignment quality based on the contributions matrix."""
    # for each of the metrics, the higher it is, the better is the alignment
    sc = alti[:, : len(source_sentence)]
    if skip_first:  # skip first token if it is special (e.g. language token after BOS)
        sc = sc[1:, :]
    src_ax, tgt_ax = 0, 1
    total_sc_by_source_token = sc.sum(src_ax)
    total_sc_by_target_token = sc.sum(tgt_ax)
    max_sc_by_source_token = sc.max(src_ax)
    max_sc_by_target_token = sc.max(tgt_ax)
    return dict(
        # detecting hallucinations by averaging over the predicted tokens
        avg_sc=total_sc_by_target_token.mean(),
        min_sc=total_sc_by_target_token.min(),
        top_sc_mean=max_sc_by_target_token.mean(),
        top_sc_min=max_sc_by_target_token.min(),
        sc_above_50=(total_sc_by_target_token >= 0.5).mean(),
        sc_above_40=(total_sc_by_target_token >= 0.4).mean(),
        sc_above_30=(total_sc_by_target_token >= 0.3).mean(),
        sc_above_20=(total_sc_by_target_token >= 0.2).mean(),
        sc_above_10=(total_sc_by_target_token >= 0.1).mean(),
        sc_share_wo_eos=1 - sc[:, -1].sum() / sc.sum(),
        avg_sc_wo_eos=sc[:, :-1].sum(tgt_ax).mean(),
        avg_sc_wo_lang=sc[1:, :].sum(tgt_ax).mean(),
        sc_entropy=entropy(total_sc_by_target_token / total_sc_by_target_token.sum()),
        # detecting undertranslations by averaging over the source tokens
        src_max_contr_below_001=(max_sc_by_source_token < 0.01).mean(),
        src_max_contr_min=max_sc_by_source_token.min(),
        src_sum_contr_below_01=(total_sc_by_source_token < 0.10).mean(),
        src_sum_contr_mean=total_sc_by_source_token.mean(),
        src_sum_contr_min=total_sc_by_source_token.min(),
    )


def compute_alti_scores_for_batch(
    alti_hub: FairseqTransformerHub,
    src_texts: tp.List[str],
    tgt_texts: tp.List[str],
    src_langs: tp.Union[None, str, tp.List[str]] = None,
    tgt_langs: tp.Union[None, str, tp.List[str]] = None,
    alignment_threshold: float = 0.08,
) -> tp.Tuple[tp.List[tp.Dict[str, float]], tp.List[tp.Dict]]:
    """Compute ALTI, sentence-level metrics and alignments for a list of sentence pairs."""
    results = []
    alignments = []
    for i in trange(len(src_texts)):
        src_lang = src_langs[i] if isinstance(src_langs, list) else src_langs
        tgt_lang = tgt_langs[i] if isinstance(tgt_langs, list) else tgt_langs
        token_level_alti, src_toks, tgt_toks, pred_toks = compute_alti_nllb(
            alti_hub, src_texts[i], tgt_texts[i], src_lang=src_lang, tgt_lang=tgt_lang
        )
        metrics = compute_alti_metrics(token_level_alti, src_toks, tgt_toks, pred_toks)
        results.append(metrics)
        # TODO: find a better alignment algorithm
        alignment = [
            (int(x), int(y))
            for x, y in zip(*np.where(token_level_alti > alignment_threshold))
        ]
        alignments.append(
            {
                "contributions": token_level_alti.tolist(),
                "alignment": alignment,
                "src_toks": src_toks,
                "tgt_toks": tgt_toks,
                "pred_toks": pred_toks,
            }
        )
    return results, alignments
