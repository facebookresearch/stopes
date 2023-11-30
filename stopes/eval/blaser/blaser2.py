# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.models.blaser.loader import load_blaser_model

from stopes.utils.math_utils import rowwise_cosine_similarity


def apply_supervised_blaser2(
    model_path: Path,
    src_embs: torch.Tensor,
    tgt_embs: torch.Tensor,
    ref_embs: tp.Optional[torch.Tensor] = None,
):
    blaser_model = load_blaser_model(model_path).eval()
    blaser_is_referenceless = blaser_model.input_form == "QE"
    with torch.inference_mode():
        if ref_embs is None or blaser_is_referenceless:
            scores = blaser_model(src=src_embs, mt=tgt_embs)
        else:
            scores = blaser_model(src=src_embs, mt=tgt_embs, ref=ref_embs)
    return scores.cpu().numpy()[:, 0]


def apply_unsupervised_blaser(
    src_embs: torch.Tensor,
    tgt_embs: torch.Tensor,
    ref_embs: tp.Optional[torch.Tensor] = None,
) -> np.ndarray:
    src = src_embs.cpu().numpy()
    tgt = tgt_embs.cpu().numpy()
    if ref_embs is None:
        return rowwise_cosine_similarity(src, tgt)
    ref = ref_embs.cpu().numpy()
    return 0.5 * (
        rowwise_cosine_similarity(src, tgt) + rowwise_cosine_similarity(src, ref)
    )


def apply_encoder(
    is_speech: bool,
    texts_or_paths: tp.List[str],
    lang: str,
    model_path: tp.Optional[Path] = None,
) -> torch.Tensor:
    """
    Load a sentence encoder and encode the list of inputs (texts sentences or paths to spoken utterances).
    The language code format is 3-letter for speech (like "eng") and 8-letter for text (like "eng_Latn").
    """
    if is_speech:
        if model_path is None:
            if lang is None:
                raise ValueError(
                    "For speech, either model_path or lang should be provided"
                )
            model_name = "sonar_speech_encoder_" + lang
        else:
            model_name = str(model_path)
        s2vec_model = SpeechToEmbeddingModelPipeline(encoder=model_name)
        vectors = s2vec_model.predict(texts_or_paths)
        return vectors
    else:
        if model_path is not None:
            model_name = str(model_path)
        else:
            model_name = "text_sonar_basic_encoder"
        t2vec_model = TextToEmbeddingModelPipeline(
            encoder=model_name, tokenizer=model_name
        )
        vectors = t2vec_model.predict(texts_or_paths, source_lang=lang)
        return vectors


def compute_blaser2(
    data_path: tp.Optional[Path] = None,
    src_column: tp.Optional[str] = None,
    tgt_column: tp.Optional[str] = None,
    ref_column: tp.Optional[str] = None,
    blaser_path: tp.Optional[Path] = None,
    src_encoder_path: tp.Optional[Path] = None,
    tgt_encoder_path: tp.Optional[Path] = None,
    src_lang: tp.Optional[str] = None,
    tgt_lang: tp.Optional[str] = None,
    src_is_speech: bool = True,
    tgt_is_speech: bool = True,
    src: tp.Optional[tp.List[str]] = None,
    tgt: tp.Optional[tp.List[str]] = None,
    ref: tp.Optional[tp.List[str]] = None,
    result_path: tp.Optional[Path] = None,
) -> tp.Tuple[
    tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor], torch.Tensor], pd.DataFrame
]:
    """
    Read the inputs (source, translation, and optionally reference texts or speech), encode them with SONAR models,
    and compute the BLASER-2.0 scores based on these embeddings.
    Optionally, save the scores to disk as a Pandas tsv dataframe.
    The encoders and BLASER models are taken from https://github.com/facebookresearch/SONAR.
    """
    if not src_lang:
        raise ValueError(
            "For encoding source text or speech, source language must be provided"
        )
    if not tgt_lang:
        raise ValueError(
            "For encoding target text or speech, target language must be provided"
        )
    if src is None or tgt is None:
        input_df = pd.read_csv(data_path, sep="\t")
        src = src if src is not None else input_df[src_column].tolist()
        tgt = tgt if tgt is not None else input_df[tgt_column].tolist()
        if ref_column is not None:
            ref = ref if ref is not None else input_df[ref_column].tolist()
    if ref is None and blaser_path and str(blaser_path).endswith("ref"):
        raise ValueError(
            "Reference inputs are not provided, but they are required for a reference-based BLASER model."
        )
    src_embs = apply_encoder(
        src_is_speech, model_path=src_encoder_path, texts_or_paths=src, lang=src_lang
    )
    tgt_embs = apply_encoder(
        tgt_is_speech, model_path=tgt_encoder_path, texts_or_paths=tgt, lang=tgt_lang
    )
    if ref is not None:
        ref_embs = apply_encoder(
            tgt_is_speech,
            model_path=tgt_encoder_path,
            texts_or_paths=ref,
            lang=tgt_lang,
        )
    else:
        ref_embs = None
    # TODO: optimize the calls so that we don't re-initialize the encoder too many times
    unsupervised_scores = apply_unsupervised_blaser(
        src_embs=src_embs, ref_embs=ref_embs, tgt_embs=tgt_embs
    )
    if blaser_path:
        supervised_scores = apply_supervised_blaser2(
            blaser_path, src_embs=src_embs, ref_embs=ref_embs, tgt_embs=tgt_embs
        )
    else:
        supervised_scores = None
    print("unsupervised_scores", unsupervised_scores.shape)
    print("supervised_scores", supervised_scores.shape)
    result_df = pd.DataFrame(
        dict(
            src=src,
            ref=ref,
            tgt=tgt,
            unsupervised_scores=unsupervised_scores,
            supervised_scores=supervised_scores,
        )
    )
    if result_path:
        result_df.to_csv(result_path, sep="\t", index=None)
    return (src_embs, ref_embs, tgt_embs), result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--result-path", type=Path, default=None)
    parser.add_argument("--src-column", type=str)
    parser.add_argument("--tgt-column", type=str)
    parser.add_argument("--ref-column", type=str)
    parser.add_argument("--blaser-path", type=Path)
    parser.add_argument("--src-encoder-path", type=Path, default=None)
    parser.add_argument("--tgt-encoder-path", type=Path, default=None)
    parser.add_argument("--src-lang", type=str, default=None)
    parser.add_argument("--tgt-lang", type=str, default=None)
    parser.add_argument("--src-is-speech", action="store_true")
    parser.add_argument("--tgt-is-speech", action="store_true")

    args = parser.parse_args()
    embs, result = compute_blaser2(**vars(args))
    print("Similarity scores were computed and saved to", args.result_path)
    print("Mean results:")
    print(result.mean(numeric_only=True))
