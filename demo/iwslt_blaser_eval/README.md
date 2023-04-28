
# Getting started with mining

Welcome to `stopes`, this is a quickstart guide to discover how to run automated pipelines with `stopes`. In this example we describe how to use BLASER to evaluate speech translation as described in the https://iwslt.org/2023/s2s task.

## Installation

We recommend installing all this in a dedicated conda environment to not overwrite your current python install:

```bash
# create conda env
conda create -n blaser python=3.9
conda activate blaser

# upgrade pip
python -m pip install --upgrade pip

# install fairseq
pip install fairseq==0.12.1

# install stopes+blaser
pip install 'stopes[blaser]'
```

# Using BLASER

The main BLASER readme has all the details of how the pipeline works: https://github.com/facebookresearch/stopes/blob/main/stopes/eval/blaser/README.md

In this tutorial we'll check how this works for the IWSLT'23 speech translation track.

## Getting the Data and Encoders

IWSLT'23 provides some test data, we'll also need the LASER encoders for the languages in the track (English and Chinese). You can run `prepare.sh` to download the data. Run this script from the `stopes/demo/iwslt_blaser_eval` folder or adapt it to your need.

```bash
./prepare.sh
```

The script will have down three things:

- download the English eval data into data/en
- download the English and Mandarin encoders in encoders/ (this will take a while as they are big)
- download the BLASER model and config
- create a "manifest" file that lists all the english files to use as source

## Create/Prepare your translation data

Once you are ready to test your system, create one wav file per source audio file and use the same name as the input file (number.wav). Create a manifest file similar to the one created by prepare.sh, you can do this with:

```bash
python mk_manifest.py manifest -i yourdatadir > yourmanifestpath.tsv
```

You should also create such manifest for your reference audio files.

## Run blaser eval:

Most of the config is already preset in `conf/eval_blaser.yaml`, please see the comments in that configuration
file if you need to adjust anything. After that you can run the following command:

```bash
DEMO_DIR=<PATH_TO_IWSLT_EVALDIRECTORY> python -m stopes.pipelines.eval.eval_blaser --config-dir $DEMO_DIR/conf tgt_manifest=<PATH_TARGET_MANIFEST.tsv> ref_manifest=<PATH_REFERENCE_MANIFEST.tsv>
```

Make sure to replace:

- `PATH_TO_IWSLT_EVALDIRECTORY` to the directory where you've run prepare.sh and downloaded everything
- `PATH_TARGET_MANIFEST.tsv` to the manifest you've generated for the translation files
- `PATH_REFERENCE_MANIFEST.tsv` to the manifest you've generated for the reference translations


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
The `blaser` code is MIT licensed, as found in the LICENSE file in the root directory.
