# HalOmi dataset

HalOmi is a small corpus of sentence translations between 9 languages,
obtained with a NLLB-200 model and manually annotated for translation hallucinations and omissions.

It is intended for benchmarking methods of detection of hallucinations and omissions at the sentence and word levels.

The dataset is described and applied in the paper [HalOmi: A Manually Annotated Benchmark for Multilingual Hallucination and Omission Detection in Machine Translation](https://arxiv.org/abs/2305.11746).

The dataset includes the following languages:

- High-resource ones:
  - arb_Arab (Modern Standard Arabic)
  - deu_Latn (German)
  - eng_Latn (English)
  - rus_Cyrl (Russian)
  - spa_Latn (Spanish)
  - zho_Hans (Mandarin)
- Lower-resource ones:
  - kas_Deva (Kashmiri)
  - mni_Beng (Manipuri, also known as Meitei)
  - yor_Latn (Yoruba)

For each of the 8 non-English languages, the dataset includes translation to and from English.
Additionally, there is a zero-shot translation direction between Spanish and Yoruba.

The dataset is intended as a test set for benchmarking methods of hallucinations and omissions detection.
We recommend using only the subset of natural translations for this purpose.
We do not provide the data splits because the dataset is small.
If the evaluated method requires tuning some parameters, we recommend using cross-validation.

The code for reproducing all the predicted scores and for computing evaluation metrics on this dataset
is released in the current directory and described below.

The code and most of the data is licensed under [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
However, portions of the dataset are available under separate license terms:
text sourced from [FLORES-200](https://github.com/facebookresearch/flores/tree/main/flores200),
[Jigsaw Toxic Comment Classification](https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification/),
and [Wikipedia](https://dumps.wikimedia.org/)
are licensed under [CC-BY-SA](https://creativecommons.org/licenses/by-sa/4.0/).

The dataset can be downloaded as a zip archive [from this url](https://dl.fbaipublicfiles.com/nllb/halomi_release_v2.zip).

## An example evaluation script

To reproduce the evaluation of all detection methods, please install the packages from `requirements.txt`,
download and unpack the dataset (it will be extracted into the `data` directory):

```bash
pip install -r requirements.txt
wget https://dl.fbaipublicfiles.com/nllb/halomi_release_v2.zip
unzip halomi_release_v2.zip
```

Then you can run the script `reproduce_evaluation.py`. The following output is expected:

```
Reproducing sentence-level scores...

Direction-wise mean score for hallucination detection:
score_log_loss       0.796786
score_alti_mean      0.747689
score_alti_t_mean    0.579903
score_attn_ot        0.533654
score_comet_qe       0.748242
score_labse          0.780582
score_laser          0.753988
score_xnli           0.669939
score_blaser2_qe     0.825663

... (part of the output omitted)

Everything reproduced as expected!
```

Please note that:

- due to a mistake found after releasing [the v1 version of the paper](https://arxiv.org/abs/2305.11746v1),
  the scores for omission detection are going to be slightly different from the ones in that version,
  and will correspond instead to the final version of the paper.
- after releasing the paper, we found that a part of the `score_log_loss` values were computed incorrectly.
  The original scores are stored in the `score_log_loss_legacy` column of the dataset,
  for reproducing the exact numbers from the paper, and the `score_log_loss` columns contain corrected scores,
  which demonstrate slightly higher correlation with human judgements.

## Dataset description

The dataset was created by translating open source sentences with an NLLB model, pre-selecting them with automatic translation quality metrics (e.g. BLEU), and then manually annotating them with sentence- and word-level labels of hallucinations and omissions.

For more details, please read the [accompanying paper](#Citations).

The dataset is released in 4 files:

- `halomi_core.tsv` - the main dataset with the results of human annotation.
- `halomi_full.tsv` - an extended version of the main dataset,
  including more rows (perturbed translations) and more columns (normalized texts and labels, sentence-level predicted scores).
- `halomi_full_source_tokens.tsv` and `halomi_full_target_tokens.tsv` - token-level predictions and labels
  for omissions and hallucinations, respectively.

### Data fields

`halomi_core.tsv` (the first 9 fields) and `halomi_full.tsv` (all fields):

| Field                 | Description                                                                                                                     |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| src_lang              | Source language code                                                                                                            |
| tgt_lang              | Target language code                                                                                                            |
| src_text              | Source text (raw)                                                                                                               |
| mt_text               | Translated text (raw)                                                                                                           |
| omit_spans            | Source text, with the omitted parts enclosed in `<<<>>>`                                                                        |
| hall_spans            | Translation text, with the hallucinated parts enclosed in `<<<>>>`                                                              |
| class_hall            | Human annotation of sentence-level hallucination degree                                                                         |
| class_omit            | Human annotation of sentence-level omission degree                                                                              |
| data_source           | Source of texts, `wiki` or `flores`                                                                                             |
| hall_mask             | Character-level hallucination mask (w.r.t. `mt_text`)                                                                           |
| omit_mask             | Character-level omission mask (w.r.t. `src_text`)                                                                               |
| src_text_normalized   | `src_text` with sentencepiece `nmt_nfkc` normalization used in NLLB                                                             |
| mt_text_normalized    | `mt_text` with sentencepiece `nmt_nfkc` normalization used in NLLB                                                              |
| hall_mask_normalized  | Character-level hallucination mask (w.r.t. `mt_text_normalized`)                                                                |
| omit_mask_normalized  | Character-level omission mask (w.r.t. `src_text_normalized`)                                                                    |
| perturbation          | Translation method, either `natural` or `perturbed`                                                                             |
| direction             | `src_lang` and `tgt_lang` joined by `-`                                                                                         |
| selection             | Pre-selection method, one of `uniform`, `biased` and `worst`                                                                    |
| score_log_loss        | sequence log-probability by the translation model                                                                               |
| score_alti_mean       | Average (over target tokens) total (over source tokens) source-target contribution computed with ALTI+                          |
| score_alti_t_mean     | Average (over target tokens) maximum (over source tokens) source-target contribution computed with ALTI+                        |
| score_attn_ot         | "Wass-Combo" score based on attention map of the NLLB model                                                                     |
| score_comet_qe        | Score predicted by the COMET-QE model                                                                                           |
| score_labse           | Cosine similarity of LaBSE sentence embeddings                                                                                  |
| score_laser           | Cosine similarity of LASER-3 sentence embeddings                                                                                |
| score_xnli            | A product of direct (source=>translation) and reverse (translation=>source) entaiment probabilities, predicted by an XNLI model |
| score_sonar_cosine    | Cosine similarity of SONAR sentence embeddings                                                                                  |
| score_blaser2_qe      | Prediction of the BLASER 2.0-QE model based on SONAR embeddings                                                                 |
| score_log_loss_legacy | A previous, incorrect version of `score_log_loss`, included for reproducibility                                                 |

Note: most of the `score_` columns, except `log_loss` and `attn_ot`, are negated, so that their lower values correspond to better estimated translation quality. The `comet` column is also shifted by 1.

`halomi_full_source_tokens.tsv` and `halomi_full_target_tokens.tsv`:

| Field                      | Description                                                                                            |
| -------------------------- | ------------------------------------------------------------------------------------------------------ |
| token                      | Sentencepiece token from the NLLB model                                                                |
| row_id                     | Index of the sentence in the `halomi_full.tsv` file                                                    |
| direction                  | `src_lang` and `tgt_lang` joined by `-`                                                                |
| perturbation               | Translation method, either `natural` or `perturbed`                                                    |
| label_mask                 | String of human labels (`1`=hallucinated/omitted, `0`=normal) for each character in the token          |
| token_label                | Human label for the whole token (`1`=hallucinated/omitted, `0`=normal)                                 |
| token_weight               | Length of token in characters (0 for added tokens)                                                     |
| start                      | Start position of the token in the corresponding sentence                                              |
| end                        | End position of the token in the corresponding sentence                                                |
| score_log_loss             | Log probability of the token by the NLLB model                                                         |
| score_log_loss_contrastive | Log probability of the token by the NLLB model, minus its probability conditionally on an empty source |
| score_alti_sum             | Sum of ALTI+ contributions for the token                                                               |
| score_alti_max             | Maximum of ALTI+ contributions for the token                                                           |

When reading the files, please make sure that the `"na"` tokens are parsed as strings, not as `NaN`.
Example code:

```Python
source_token_df = pd.read_csv(
    os.path.join(data_root, "halomi_full_source_tokens.tsv"),
    sep="\t",
    keep_default_na=False,
)
```

## On reproducing the dataset creation

### Translations

The translations were produced using the [nllb](https://github.com/facebookresearch/fairseq/tree/nllb) branch of the `fairseq` repo.
The script `example_translation_script.sh` shows all the parameters that we used to create the translations.

### Detection scores

The script `compute_detection_scores.py` reproduces exactly most of the scores, except
`score_attn_ot` that requires a large amount of scored reference translations, which we do not release.
Additionally, our implementation of attention-optimal-transport methods is randomized (but reproducible with a fixed seed).
However, we did not set the seed at the time we produced the scores, so this affects reproducibility and may lead to slightly varying results.

To run this script, you will have to:

- install the requirements from `requirements-detection.txt`;
- install `fairseq` at the [nllb](https://github.com/facebookresearch/fairseq/tree/nllb) branch;
- install [stopes](https://github.com/facebookresearch/stopes);
- from the [NLLB page](https://github.com/facebookresearch/fairseq/tree/nllb), download
  the NLLB-200-600M checkpoint, dictionary and Sentencepiece model;
- rename the dictionary file into `dict.eng_Latn.txt` and put it in the directory that will be used later as `NLLB_DATA_DIR`.

Now you can run the script for reproducing the scores as follows:

```bash
python compute_detection_scores.py \
    --data-root=data \
    --save-filename=scores_reproduction.tsv,
    --nllb-data-dir={PATH TO THE DIRECTORY WITH dict.eng_Latn.txt} \
    --nllb-spm-path={PATH TO THE NLLB SPM MODEL} \
    --nllb-checkpoint={PATH TO THE NLLB PYTORCH MODEL} \
    --attn-references-path={PATH TO THE ATTN-OT REFERENCES, IF YOU HAVE THEM} \
    --internal --comet --labse --laser --xnli --sonar
```

## Citations

To refer to the dataset or evaluation results, please cite:

```bibtex
@article{dale2023halomi,
  title={HalOmi: A Manually Annotated Benchmark for Multilingual Hallucination and Omission Detection in Machine Translation},
  author={Dale, David and Voita, Elena and Lam, Janice and Hansanti, Prangthip and Ropers, Christophe and Kalbassi, Elahe and Gao, Cynthia and Barrault, Lo{\"\i}c and Costa-juss{\`a}, Marta R},
  journal={arXiv preprint arXiv:2305.11746},
  url={https://arxiv.org/abs/2305.11746},
  year={2023}
}
```
