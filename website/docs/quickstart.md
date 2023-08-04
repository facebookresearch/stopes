---
sidebar_position: 1
---

# Getting started with mining

Welcome to `stopes`, this is a quickstart guide to discover how to run automated pipelines with `stopes`. In this example, you'll be running
global mining with the `stopes` toolchain. (Inspired by
[CCMatrix](https://ai.facebook.com/blog/ccmatrix-a-billion-scale-bitext-data-set-for-training-translation-models/)).

## Installation

Follow the installation steps from the [project's README](https://github.com/facebookresearch/stopes/blob/main/README.md), we recommend doing this in a separate [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).

## Getting Data

To run the global mining pipeline, you first need to get some monolingual data.
The [WMT22 Shared Task: Large-Scale Machine Translation Evaluation for African
Languages](https://statmt.org/wmt22/large-scale-multilingual-translation-task.html)
has some interesting monolingual data for some African languages.

You also need some trained encoder, we usually use `stopes` with LASER and we can
find such trained encoders for the languages in the WMT22 shared task too.

The `demo/mining/prepare.sh` script will download the monolingual data and LASER encoders
for you. Start by running this script and wait for the download to finish.

:::tip

`prepare.sh` was built for the quickstart demo, it will only download the encoders and data released as part of the African Languages workshop. The NLLB project has released [many encoders](https://github.com/facebookresearch/LASER/tree/main/nllb/) and data that you can leverage with `stopes`.

:::

## Configuring the pipeline

In `stopes` pipelines, we use [hydra](https://hydra.cc/) to configure the runs.
With hydra, you can configure everything with "overrides" on the cli, but it's
often easier to put the configurations in yaml files as there is a lot of things
to setup.

`stopes/pipelines/bitext/conf/preset/demo.yaml` is a demo configuration for the
data and encoders that we've downloaded in the previous steps. Check out the
comments in that file.

The important parts of that preset config is:
1. we setup the launcher to run on your local computer (no need for a cluster)
2. we setup an alias for a `demo_dir` folder, so you can point to the
   data/models from the cli
3. we setup some information about the `data`:
   - some naming, to get nice file names as outputs
   - where the data is found (with `shard_glob`)
4. we tell the pipeline where to find the encoder and SentencePiece model (SPM) uses
   to embed the text. We do that for each lang in `lang_configs`. Practically,
   if you are only processing a few languages, you don't need so many entries,
   here we preset them for all languages from the WMT22 task

:::tip

Language codes are important, but not standardized everywhere. The `stopes` library does not make any assumptions to what codes you are using. As you will see in the configuration and the prepare script, codes are mostly important in naming data input files. If you want to use a different coding scheme, make sure that the files and config use the same naming conventions.

:::

## Run the Pipeline

You can now start the pipeline with:
```bash
python -m stopes.pipelines.bitext.global_mining_pipeline src_lang=fuv tgt_lang=zul demo_dir=.../stopes-repo/demo/mining +preset=demo output_dir=. embed_text=laser3
```

- `src_lang` and `tgt_lang` specify the pair of languages we want to process,
- `demo_dir` is the new variable we introduce in our preset/demo.yaml file, to
  point to where the `prepare.sh` script downloaded our data; make sure to
  specify an absolute path,
- `+preset=demo` tells hydra to load the demo.yaml preset file to set our
  defaults (the `+` here is because we are telling hydra to append a group that
  doesn't exist in the default config, see the [hydra
  doc](https://hydra.cc/docs/1.0/advanced/override_grammar/basic/#basic-override-syntax)
  for details),
- `output_dir` specifies where we want the output (current run directory),
- `embed_text=laser3` tells the pipeline to use the laser3 encoding code to load
  the models and encode the text.

## Try using a different encoder

In the previous run, we used `embed_text=laser3`, which will encode text with
the language specific laser3, but you can also use other encoders. For instance,
stopes ships with [HuggingFace
sentence-transformers](https://huggingface.co/sentence-transformers), so you can
use different encoders if you want to experiment.

You need to install `sentence-transformers` in your environment:

```bash
pip install --user sentence-transformers
```

Here is an example to run LaBSE:

```bash
python -m stopes.pipelines.bitext.global_mining_pipeline src_lang=fuv tgt_lang=zul demo_dir=.../stopes-repo/demo/mining +preset=demo output_dir=. embed_text=hf_labse lang_configs=null embedding_dimensions=768
```

:::tip

## Explore More

Check out these docs to learn more:
- [Prebuilt Pipelines](category/prebuilt-pipelines)
- [`stopes` Module Framework](stopes)

:::
