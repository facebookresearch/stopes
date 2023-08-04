# BLASER: A Text-Free Speech-to-Speech Translation Evaluation Metric

BLASER leverages a multilingual multimodal encoder to directly encode the speech segments for source input, translation output and reference into a shared embedding space and computes a score of the translation quality that can be used as a proxy to human evaluation.

In this folder you can find tools to use BLASER to score speech translation and to train the BLASER supervised model. You can also download a [pre-trained model](http://dl.fbaipublicfiles.com/blaser/blaser.tar.gz).

BLASER relies on [SpeechLASER embeddings](https://github.com/facebookresearch/fairseq/blob/ust/examples/speech_matrix/speech_laser_encoders.md), follow the instructions there to download the embeddings and embed your speech segments.

## Files

- `model.py` contains the BLASER supervised model definition.
- `train.py` contains a script that you can use to train the BLASER model.
- `score.py` contains a script that you can use to score speech segments translations.

The train and score scripts are configured using [hydra](https://hydra.cc/), you can look at the base configurations in `blaser/conf/score.yaml` and `blaser/conf/train.yaml`, you will have to specify the `???` fields, either in your own configs, or over the CLI. Both scripts take pre-embedded files.

## Install

```
git clone https://github.com/facebookresearch/stopes.git
cd stopes
pip install -e '.[blaser]'
```

## Using the pipeline

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

## License
`blaser` is MIT licensed, as found in the LICENSE file in the root directory.
