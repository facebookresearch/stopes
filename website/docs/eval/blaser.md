# BLASER: an evaluation metric of translation accuracy for speech and text

BLASER leverages a multilingual multimodal encoder to directly encode the speech segments for source input, translation output and (optionally) reference into a shared embedding space and computes a score of the translation quality that can be used as a proxy to human evaluation.

There are two generations of BLASER models:

1. BLASER is based on [SpeechLASER embeddings](https://github.com/facebookresearch/fairseq/blob/ust/examples/speech_matrix/speech_laser_encoders.md), is intended to use with speech only, and always requires a reference translation.
2. BLASER-2.0 is based on [SONAR embeddings](https://github.com/facebookresearch/SONAR), supports speech and texts interchangeably, and has a reference-free version (BLASER-2.0 QE).

Below, both these model generations are presented.

## BLASER 2.0

BLASER 2.0 is a family of models for automatic evaluation of machine translation quality based on SONAR embeddings, presented in the [SeamlessM4T](https://arxiv.org/abs/2308.11596) paper. They predict [cross-lingual semantic similarity](https://github.com/facebookresearch/fairseq/tree/nllb/examples/nllb/human_XSTS_eval) between the translation and the source (optionally, also using a reference translation).

There are two BLASER 2.0 supervised models: [facebook/blaser-2.0-ref](https://huggingface.co/facebook/blaser-2.0-ref) that requires a reference translation, and [facebook/blaser-2.0-qe](https://huggingface.co/facebook/blaser-2.0-qe) that uses only source and machine translation as inputs. Both can work with text or speech interchangeably. Additionally, we compute unsupervised scores as cosine similarity between the embeddings.

The code for BLASER 2.0, as well as for the underlying SONAR text and speech encoders, is published in the [SONAR](https://github.com/facebookresearch/SONAR) repository.

Here, we present a script `stopes/eval/blaser/blaser2.py`, just to illustrate a possible usage of BLASER 2.0. If you have a `.tsv` file with the columns `src_audio`, `ref_audio`, and `tgt_audio`, with the paths to `.wav` files of English source, reference translation into French, and system translation, you can compute BLASER scores for them with the following code:

```Python
from stopes.eval.blaser.blaser2 import compute_blaser2

(src_embs, ref_embs, tgt_embs), df_with_scores = compute_blaser2(
    data_path=PATH_TO_THE_FILE,
    src_column="src_audio",
    ref_column="ref_audio",
    tgt_column="tgt_audio",
    blaser_path="blaser_2_0_ref",  # or blaser_2_0_qe, if you don't use references
    src_lang="eng",  # lookup language codes on SONAR model cards
    tgt_lang="fra",
    src_is_speech=True,
    tgt_is_speech=True,
)
mean_score = df_with_scores.mean(numeric_only=True)
print(mean_score.unsupervised_scores)  # a number usually between 0 and 1
print(mean_score.supervised_scores)    # a number usually between 1 and 5
```

To run this script, you will need to install [fairseq2](https://github.com/facebookresearch/fairseq2) and [SONAR](https://github.com/facebookresearch/SONAR).

## BLASER: A Text-Free Speech-to-Speech Translation Evaluation Metric

BLASER leverages a multilingual multimodal encoder to directly encode the speech segments for source input, translation output and reference into a shared embedding space and computes a score of the translation quality that can be used as a proxy to human evaluation.

In this folder you can find tools to use BLASER to score speech translation and to train the BLASER supervised model. You can also download a [pre-trained model](http://dl.fbaipublicfiles.com/blaser/blaser.tar.gz).

BLASER relies on [SpeechLASER embeddings](https://github.com/facebookresearch/fairseq/blob/ust/examples/speech_matrix/speech_laser_encoders.md), follow the instructions there to download the embeddings and embed your speech segments.

> [!NOTE] BlaserV2.0 has now been released and lives in the [SONAR repository](https://github.com/facebookresearch/SONAR#predicting-sentence-similarity-with-blaser-20-models). We recommend using this new model and embedding space for your future uses of BLASER evalution

### Files

- `model.py` contains the BLASER supervised model definition.
- `train.py` contains a script that you can use to train the BLASER model.
- `score.py` contains a script that you can use to score speech segments translations.

The train and score scripts are configured using [hydra](https://hydra.cc/), you can look at the base configurations in `blaser/conf/score.yaml` and `blaser/conf/train.yaml`, you will have to specify the `???` fields, either in your own configs, or over the CLI. Both scripts take pre-embedded files.

### Install

```
git clone https://github.com/facebookresearch/stopes.git
cd stopes
pip install -e '.[blaser]'
```

### Using the pipeline

Blaser requires embedded speech segments to compute the evaluation metric. A good way to get these embeddings is to use [SpeechLASER embeddings](https://github.com/facebookresearch/fairseq/blob/ust/examples/speech_matrix/speech_laser_encoders.md).

We provide a pipeline that will compute the embeddings and feed them to the blaser model for you.

** Model **

You can download the model from [here](https://dl.fbaipublicfiles.com/blaser/blaser.tar.gz).

** Requirements **

To be able to use this pipeline, you will need the version of fairseq that has the Wav2VevLaser model implementation, this is currently in the `ust` branch of fairseq. Make sure to clone the repository, switch to that branch and install fairseq in your environment with: `pip install -e .`. It is recommended to do this before install stopes as per above.

** Run the pipeline **

You will need to pass three sets of speech segments to get a blaser score:

- the source audio (`src`)
- the translated audio (`target`)
- the reference audio (`ref`)

The set of speech segments have to be organised in a tsv manifest pointing to the audio files. The format for each input audio data tsv file is:

```
<root_dir>
<audio_path>\t<num_frames>
<audio_path>\t<num_frames>
...
...
```

Rows in each manifest TSV files should align (line 1 of the src manifest should be translated on line 1 of the tgt manifest).

- `root_dir` is on the first line of the TSV, it's a path to where all the following files can be found
- `audio_path` can be either:
  - a filename in `root_dir` for a .npy/.wav/.flac/.ogg file
  - a filename of stored ZIP file, in `root_dir`, with slicing info: "[zip_path]:[offset]:[length]" to find the bytes in the uncompressed zip archive containing that particular file

Check out the `demo/iwslt_blaser_eval/mk_manifest.py` script to see how to generate such a manifest file.

You can then run the pipeline with:

```
python -m stopes.pipelines.eval.eval_blaser output_dir=YOUROUTPUTDIRECTORY src_manifest=PATHTOSOURCEMANIFEST.tsv tgt_manifest=PATHTOTARGETMANIFEST.tsv ref_manifest=PATHTOREFERENCEMANIFEST.tsv src_lang=en tgt_lang=de
```

where `src_lang` is the language of your source audio and tgt_lang is the target language. This is used to lookup the correct encoder model as specified by `stopes/pipelines/eval/conf/eval_blaser.yaml`. You can download pre-trained encoders from the [SpeechMatrix project](https://github.com/facebookresearch/fairseq/blob/ust/examples/speech_matrix/speech_laser_encoders.md). By default, the encoder used for the reference is the same as the target one, you can override this with `ref_lang=..` in the command arguments.

## Citation

If you use `blaser` in your work or any of its models, please cite:

```bibtex
@misc{blaser2022,
  title={BLASER: A Text-Free Speech-to-Speech Translation Evaluation Metric},
  author={Mingda Chen and Paul-Ambroise Duquenne and Pierre Andrews and Justine Kao and Alexandre Mourachko and Holger Schwenk and Marta R. Costa-jussà},
  year={2022}
  doi = {10.48550/ARXIV.2212.08486},
  url = {https://arxiv.org/abs/2212.08486},
  publisher = {arXiv},
}
```

If you use `blaser-2.0`, please cite:

```bibtex
@article{seamlessm4t2023,
  title={SeamlessM4T—Massively Multilingual \& Multimodal Machine Translation},
  author={{Seamless Communication}, Lo\"{i}c Barrault, Yu-An Chung, Mariano Cora Meglioli, David Dale, Ning Dong, Paul-Ambroise Duquenne, Hady Elsahar, Hongyu Gong, Kevin Heffernan, John Hoffman, Christopher Klaiber, Pengwei Li, Daniel Licht, Jean Maillard, Alice Rakotoarison, Kaushik Ram Sadagopan, Guillaume Wenzek, Ethan Ye,  Bapi Akula, Peng-Jen Chen, Naji El Hachem, Brian Ellis, Gabriel Mejia Gonzalez, Justin Haaheim, Prangthip Hansanti, Russ Howes, Bernie Huang, Min-Jae Hwang, Hirofumi Inaguma, Somya Jain, Elahe Kalbassi, Amanda Kallet, Ilia Kulikov, Janice Lam, Daniel Li, Xutai Ma, Ruslan Mavlyutov, Benjamin Peloquin, Mohamed Ramadan, Abinesh Ramakrishnan, Anna Sun, Kevin Tran, Tuan Tran, Igor Tufanov, Vish Vogeti, Carleigh Wood, Yilin Yang, Bokai Yu, Pierre Andrews, Can Balioglu, Marta R. Costa-juss\`{a}, Onur \,{C}elebi,Maha Elbayad,Cynthia Gao, Francisco Guzm\'an, Justine Kao, Ann Lee, Alexandre Mourachko, Juan Pino, Sravya Popuri, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, Paden Tomasello, Changhan Wang, Jeff Wang, Skyler Wang},
  journal={ArXiv},
  year={2023}
}
```

## License

`blaser` is MIT licensed, as found in the LICENSE file in the root directory.
