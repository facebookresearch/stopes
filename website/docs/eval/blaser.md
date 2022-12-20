# BLASER: A Text-Free Speech-to-Speech Translation Evaluation Metric

BLASER leverages a multilingual multimodal encoder to directly encode the speech segments for source input, translation output and reference into a shared embedding space and computes a score of the translation quality that can be used as a proxy to human evaluation.

In this folder you can find tools to use BLASER to score speech translation and to train the BLASER supervised model.

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
