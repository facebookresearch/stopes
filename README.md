![stopes](/website/static/img/banner.png?raw=true "stopes by NLLB.")


# `stopes`: A library for preparing data for machine translation research

As part of the FAIR No Language Left Behind (NLLB) project to drive inclusion through machine translation, a large amount of data was processed to create training data. We provide the libraries and tools we used to:

1. create clean monolingual data from web data
2. mine bitext
3. easily write scalable pipelines for processing data for machine translation

## Examples

checkout the `demo` directory for an example usage with the [WMT22 Shared Task: Large-Scale Machine Translation Evaluation for African
Languages](https://statmt.org/wmt22/large-scale-multilingual-translation-task.html) data.

## Requirements
`stopes` relies on:
* submitit to schedule jobs when ran on clusters
* hydra-core version >= 1.2.0 for configuration
* fairseq to use LASER encoders
* PyTorch version >= 1.5.0
* Python version >= 3.8

## Installing stopes

stopes uses [flit](https://flit.pypa.io/) to manage its setup, you will need a recent version of
pip for the install to work. We recommend that you first upgrade pip:
`python -m pip install --upgrade pip`

You can install stopes with pip:
`pip install -e '.[dev,mono,mining]'`

You can choose what to install. If you are only interested in `mining`, you do not need to install `dev`, and `mono`.

The mining pipeline relies on fairseq to run LASER encoders, pip cannot install fairseq currently, so you will have to do this manually. Check the [fairseq](https://github.com/facebookresearch/fairseq) repo for up to date instructions and requirements:
```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

If you plan to train a lot of NMT model you will also want to setup apex to get a faster training.
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

## How `stopes` works

`stopes` is made of a few different parts:
1. `core` provides a library to write readable piplines
2. `modules` provides a set of modules using the core library and implementing
   common steps in our mining and evaluation pipelines
3. `pipelines` provides pipeline implementation for the data pipelines we use in
   NLLB:
- `monolingual` to preprocess and clean single language data
- `bitext` to run the "global mining" pipeline and extract aligned sentences
  from two monolingual datasets. (inspired by
  [CCMatric](https://ai.facebook.com/blog/ccmatrix-a-billion-scale-bitext-data-set-for-training-translation-models/))

## Full documentation

see https://facebookresearch.github.io/stopes
or the `websites/docs` folder.

## Contributing

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## Contributors

- [Pierre Andrews](https://github.com/Mortimerp9)
- [Onur Çelebi](https://github.com/Celebio)
- [Angela Fan](https://github.com/huihuifan)
- [Vedanuj Goswami](https://github.com/vedanuj)
- [Kevin Heffernan](https://github.com/heffernankevin)
- [Ammar Kamran](https://github.com/AmmarKamran)
- [Jean Maillard](https://github.com/jeanm)
- [Alexandre Mourachko](https://github.com/alexmourachko)
- [Kaushik Ram Sadagopan](https://github.com/kauterry)
- [Holger Schwenk](https://github.com/hoschwenk)
- [Guillaume Wenzek](https://github.com/gwenzek)

(in alphabetical order)

## License
`stopes` is MIT licensed, as found in the LICENSE file.
