# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import pickle
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from attention_optimal_transport import att_maps_compute, optimal_transport_scoring
from comet import download_model, load_from_checkpoint
from laser_encoders import LaserEncoderPipeline
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm, trange
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from stopes.eval.alti.alti_metrics.alti_metrics_utils import (
    compute_alti_metrics,
    compute_alti_nllb,
    get_loss,
)
from stopes.eval.alti.alti_metrics.nllb_alti_detector import load_nllb_model


def add_nllb_scores(
    dataset: pd.DataFrame,
    nllb_checkpoint: str,
    nllb_data_dir: str,
    nllb_spm_path: str,
    attn_references_path: tp.Optional[str] = None,
):
    print("Computing the internal detection methods...")

    alti_hub = load_nllb_model(
        checkpoint=Path(nllb_checkpoint),
        data_dir=Path(nllb_data_dir),
        spm=Path(nllb_spm_path),
        src_lang="eng_Latn",
        tgt_lang="eng_Latn",
    )
    alti_metrics = [
        compute_alti_metrics(
            *compute_alti_nllb(
                alti_hub, row.src_text, row.mt_text, row.src_lang, row.tgt_lang
            )
        )
        for row in tqdm(dataset.itertuples())
    ]
    dataset["score2_alti_mean"] = [-m["avg_sc"] for m in alti_metrics]
    dataset["score2_alti_t_mean"] = [-m["src_sum_contr_mean"] for m in alti_metrics]
    losses = [
        get_loss(alti_hub, row.src_text, row.mt_text, row.src_lang, row.tgt_lang)[
            "loss_avg"
        ]
        for row in tqdm(dataset.itertuples())
    ]
    dataset["score2_log_loss"] = losses

    if attn_references_path:
        add_optimal_transport_scores(dataset, alti_hub, attn_references_path)


def add_optimal_transport_scores(dataset: pd.DataFrame, alti_hub, references_path):
    print("Computing the Attn-OT detection methods...")
    subset = dataset[["src_text", "mt_text", "direction"]].copy()
    subset.columns = ["src", "mt", "direction"]
    subset.direction = subset.direction.apply(lambda x: f"{x[:8]}-{x[-8:]}")

    attmaps_test = att_maps_compute.get_attention_maps(subset, alti_hub)
    attmaps_test64 = optimal_transport_scoring.norm64(attmaps_test)
    wass_to_uniform = [
        optimal_transport_scoring.wass2unif(am) for am in tqdm(attmaps_test)
    ]

    with open(references_path, "rb") as f:
        filtered_attmaps_by_dir_and_len = pickle.load(f)
    filtered_attmaps_by_dir_and_len = {
        direction: {
            length: optimal_transport_scoring.norm64(maps)
            for length, maps in len2maps.items()
        }
        for direction, len2maps in filtered_attmaps_by_dir_and_len.items()
    }
    wass_to_data = [
        optimal_transport_scoring.get_best_dist(
            att_map, filtered_attmaps_by_dir_and_len[direction], random_seed=1
        )
        for att_map, direction in tqdm(
            zip(attmaps_test64, subset.direction), total=len(attmaps_test)
        )
    ]
    wass_combo = optimal_transport_scoring.get_combo_score(
        wass_to_uniform, wass_to_data
    )
    dataset["score2_attn_ot"] = wass_combo


def add_comet_scores(dataset: pd.DataFrame, gpus=1, batch_size=8):
    print("Computing COMET-QE scores...")
    model_path = download_model("wmt20-comet-qe-da-v2")
    model = load_from_checkpoint(model_path)
    comet_sample = (
        dataset[["src_text", "mt_text"]]
        .rename({"src_text": "src", "mt_text": "mt"}, axis=1)
        .to_dict("records")
    )
    comet_qe = model.predict(comet_sample, batch_size=batch_size, gpus=gpus)
    dataset["score2_comet_qe"] = 1 - np.array(comet_qe.scores)


def add_labse_scores(dataset: pd.DataFrame):
    print("Computing LaBSE scores...")
    labse = SentenceTransformer("sentence-transformers/LaBSE")
    emb_src = labse.encode(dataset.src_text.tolist(), show_progress_bar=True)
    emb_mt = labse.encode(dataset.mt_text.tolist(), show_progress_bar=True)
    labse_sims_src = (emb_src * emb_mt).sum(1)
    dataset["score2_labse"] = -labse_sims_src


def laser_embed_without_preprocessing(text, model):
    # return model.encode_sentences([text])
    # LASER normally preprocesses texts before tokenization.
    # I didn't knew this at the moment of experiments,
    # so for reproducibility, I don't normalize here either
    encoded_text = " ".join(model.tokenizer.spm_encoder.encode(text, out_type=str))
    return model.encoder.encode_sentences([encoded_text])


def l2norm(x):
    return x / (x**2).sum(-1, keepdims=True) ** 0.5


def add_laser_scores(
    dataset: pd.DataFrame, laser3_langs=("yor_Latn", "kas_Deva", "mni_Beng")
):
    print("Computing LASER scores...")
    laser2 = LaserEncoderPipeline(lang="eng", laser="laser2")
    lasers3 = {
        lang: LaserEncoderPipeline(lang=lang, laser="laser3") for lang in laser3_langs
    }
    src_embs = l2norm(
        np.concatenate(
            [
                laser_embed_without_preprocessing(
                    row.src_text, lasers3.get(row.src_lang, laser2)
                )
                for row in tqdm(dataset.itertuples())
            ]
        )
    )
    tgt_embs = l2norm(
        np.concatenate(
            [
                laser_embed_without_preprocessing(
                    row.mt_text, lasers3.get(row.tgt_lang, laser2)
                )
                for row in tqdm(dataset.itertuples())
            ]
        )
    )
    dataset["score2_laser"] = -(src_embs * tgt_embs).sum(-1)


def compute_entailment_score(
    dataset, model, tokenizer, batch_size=16, column1="src_text", column2="mt_text"
):
    scores = []
    for i in trange(0, dataset.shape[0], batch_size):
        batch = dataset.iloc[i : i + batch_size]
        with torch.inference_mode():
            inputs = tokenizer(
                batch[column1].tolist(),
                batch[column2].tolist(),
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            proba = (
                torch.softmax(model(**inputs).logits, -1)[
                    :, model.config.label2id["entailment"]
                ]
                .cpu()
                .numpy()
            )
        scores.append(proba)
    return np.concatenate(scores)


def add_xnli_scores(dataset: pd.DataFrame):
    print("Computing XNLI scores...")
    model_name = "joeddav/xlm-roberta-large-xnli"
    model = AutoModelForSequenceClassification.from_pretrained(model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    scores_forward = compute_entailment_score(
        dataset, model, tokenizer, column1="src_text", column2="mt_text"
    )
    scores_backward = compute_entailment_score(
        dataset, model, tokenizer, column1="mt_text", column2="src_text"
    )
    dataset["score2_xnli"] = -scores_forward * scores_backward


def add_sonar_scores(dataset: pd.DataFrame):
    print("Computing SONAR scores...")
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
    from sonar.models.blaser.loader import load_blaser_model

    t2vec_model = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device="cuda",
    )
    blaser_qe = load_blaser_model("blaser_2_0_qe").eval()

    # We compute embeddings for each language separately, and then reorder them with the original index
    src_embs = []
    tgt_embs = []
    reindices = []
    for (src_lang, tgt_lang), df_part in tqdm(
        dataset.groupby(["src_lang", "tgt_lang"])
    ):
        reindices.extend(df_part.index)
        src_embs.append(
            t2vec_model.predict(df_part.src_text.tolist(), source_lang=src_lang)
        )
        tgt_embs.append(
            t2vec_model.predict(df_part.mt_text.tolist(), source_lang=tgt_lang)
        )

    new_indices = (
        pd.Series(range(len(reindices)), index=reindices).loc[dataset.index].tolist()
    )
    src_embs = torch.concat(src_embs).cpu().numpy()[new_indices]
    tgt_embs = torch.concat(tgt_embs).cpu().numpy()[new_indices]

    src_embs_norm = l2norm(src_embs)
    tgt_embs_norm = l2norm(tgt_embs)

    dataset["score2_sonar_cosine"] = -(src_embs_norm * tgt_embs_norm).sum(-1)
    blaser_scores = (
        blaser_qe(src=torch.tensor(src_embs), mt=torch.tensor(tgt_embs))
        .detach()
        .cpu()
        .numpy()[:, 0]
    )
    dataset["score2_blaser2_qe"] = -blaser_scores


def main(
    data_root: str = "data",
    save_filename: tp.Optional[str] = None,
    sample: bool = False,
    nllb_data_dir: tp.Optional[str] = None,
    nllb_checkpoint: tp.Optional[str] = None,
    nllb_spm_path: tp.Optional[str] = None,
    attn_references_path: tp.Optional[str] = None,
    compute_nllb_scores: bool = True,
    compute_comet: bool = True,
    compute_labse: bool = True,
    compute_laser: bool = True,
    compute_xnli: bool = True,
    compute_sonar: bool = True,
):
    dataset = pd.read_csv(os.path.join(data_root, "halomi_full.tsv"), sep="\t")
    if sample:
        dataset = dataset.sample(30, random_state=1)
    if compute_nllb_scores:
        assert isinstance(nllb_data_dir, str) and os.path.exists(
            nllb_data_dir
        ), f"NLLB data dir does not exist: {nllb_data_dir}"
        assert isinstance(nllb_checkpoint, str) and os.path.exists(
            nllb_checkpoint
        ), f"NLLB checkpoint file does not exist: {nllb_checkpoint}"
        assert isinstance(nllb_spm_path, str) and os.path.exists(
            nllb_spm_path
        ), f"NLLB SPM file does not exist: {nllb_spm_path}"
        add_nllb_scores(
            dataset,
            nllb_data_dir=nllb_data_dir,
            nllb_checkpoint=nllb_checkpoint,
            nllb_spm_path=nllb_spm_path,
            attn_references_path=attn_references_path,
        )
    if compute_comet:
        add_comet_scores(dataset)
    if compute_labse:
        add_labse_scores(dataset)
    if compute_laser:
        add_laser_scores(dataset)
    if compute_xnli:
        add_xnli_scores(dataset)
    if compute_sonar:
        add_sonar_scores(dataset)

    new_columns = [c for c in dataset.columns if c.startswith("score2_")]
    print("Scores computed: ", new_columns)
    old_columns = [c.replace("score2_", "score_") for c in new_columns]
    simple_columns = [c[7:] for c in new_columns]
    new_values = dataset[new_columns].copy()
    new_values.columns = simple_columns
    old_values = dataset[old_columns].copy()
    old_values.columns = simple_columns
    diffs = new_values - old_values
    print("distribution of differences:")
    stats = diffs.describe()
    stats.loc["correlation"] = pd.Series(
        {
            c: old_values[c].corr(new_values[c], method="spearman")
            for c in simple_columns
        }
    )
    print(stats)
    if save_filename:
        print(f"Saving results to {save_filename}")
        dataset.to_csv(
            os.path.join(data_root, save_filename), sep="\t", index_label="idx"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="data",
        type=str,
        help="path to the directory with the dataset",
    )
    parser.add_argument(
        "--save-filename",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--sample", action="store_true", help="run the analysis on a small data sample"
    )

    # Internal scores
    parser.add_argument(
        "--internal", action="store_true", help="Compute ALTI+ and log loss scores"
    )
    parser.add_argument("--nllb-data-dir", type=str, required=False)
    parser.add_argument("--nllb-spm-path", type=str, required=False)
    parser.add_argument("--nllb-checkpoint", type=str, required=False)
    parser.add_argument(
        "--attn-references-path",
        type=str,
        required=False,
        help="path to picled dict of the form {directions: {lengths: [att_map,]}",
    )

    # External scores
    parser.add_argument("--comet", action="store_true")
    parser.add_argument("--labse", action="store_true")
    parser.add_argument("--laser", action="store_true")
    parser.add_argument("--xnli", action="store_true")
    parser.add_argument("--sonar", action="store_true")

    args = parser.parse_args()
    main(
        data_root=args.data_root,
        save_filename=args.save_filename,
        sample=args.sample,
        compute_nllb_scores=args.internal,
        compute_comet=args.comet,
        compute_labse=args.labse,
        compute_laser=args.laser,
        compute_xnli=args.xnli,
        compute_sonar=args.sonar,
        nllb_data_dir=args.nllb_data_dir,
        nllb_spm_path=args.nllb_spm_path,
        nllb_checkpoint=args.nllb_checkpoint,
        attn_references_path=args.attn_references_path,
    )
