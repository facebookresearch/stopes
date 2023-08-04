---
sidebar_position: 1
---

# Global Mining Pipeline

# Basic Usage

You can launch the mining for a pair of languages with the following command:


```bash
python -m stopes.pipelines.bitext.global_mining_pipeline src_lang=fuv tgt_lang=zul demo_dir=.../stopes-repo/demo +preset=demo output_dir=. embed_text=laser2
```
(see the demo doc for a quick understanding of the `+preset` override)


This will run the required steps and try to re-use whatever step outputs has already been computed. So if you run this exact command multiple times (e.g. after a pre-emption in slurm), it will start from where it failed instead of recomputing everything.

Here is an example log:


```
[global_mining][INFO] - output: ....../mining/global_mining/outputs/2021-11-02/08-56-40
[global_mining][INFO] - working dir: ....../mining/global_mining/outputs/2021-11-02/08-56-40
[mining_utils][WARNING] - No mapping for lang bn
[embed_text][INFO] - Number of shards: 55
[embed_text][INFO] - Embed bn (hi), 55 files
[stopes_launcher][INFO] - for encode.bn.55 found 55 already cached array results,0 left to compute out of 55
[train_faiss_index][INFO] - lang=bn, sents=135573728, required=40000000, index type=OPQ64,IVF65536,PQ64
[stopes_launcher][INFO] - index-train.bn.iteration_2 done from cache
[stopes_launcher][INFO] - for populate_index.OPQ64,IVF65536,PQ64.bn found 44 already cached array results,11 left to compute out of 55
[stopes_launcher][INFO] - submitted job array for populate_index.OPQ64,IVF65536,PQ64.bn: ['48535900_0', ..., '48535900_10']
[mining_utils][WARNING] - No mapping for lang hi
[embed_text][INFO] - Number of shards: 55
[embed_text][INFO] - Embed hi (hi), 55 files
[stopes_launcher][INFO] - for encode.hi.55 found 55 already cached array results,0 left to compute out of 55
[train_faiss_index][INFO] - lang=hi, sents=162844151, required=40000000, index type=OPQ64,IVF65536,PQ64
```


We can see that the launcher has found out that it doesn't need to run the encode and train index steps for the bn lang (source language) and can skip straight to populating the index with embeddings, but it also already processed 44 shards for that step, so will only re-schedule jobs for 11 shards. In parallel, it is also processing the target language (hi) and found that it still needs to run the index training step as it also recovered all the encoded shards.

If you are using slurm as the launcher instead of the local setting, the pipeline also takes care of communicating with slurm, waiting for all slurm jobs to finish and synchronizing the consecutive jobs. See below on how to run single steps for debugging.

You can run the whole pipeline locally with:


```bash
python global_mining_pipeline.py src_lang=bn tgt_lang=hi +data=ccg launcher.cluster=local
```



# Understanding the Configuration

The configuration is driven by [Hydra](https://hydra.cc/), this makes it sound way more complicated than it actually is. The first main difference is how the command line arguments are specified. Instead of using the `--arg=foobar` standard notation, Hydra introduces its [own notation](https://hydra.cc/docs/1.0/advanced/override_grammar/basic/#basic-override-syntax) to be able to have a  more complete syntax. This is indeed odd, but once you are used to it, it provides a lot of benefits.

A second big change is that most of the things that can be changed in the pipeline are driven by yaml configuration files instead of having to change the script files. These configuration files are checked in and you can override them on the command line (see the examples above). The pipeline will log the actual full config+overrides in the output folder when you do a run, so that you can always look at the config that was used to generate a particular data folder.

The third major change, and main benefit, is that the configs are split in "groups" (hydra terminology) and you can override a whole group with another yaml file with a very simple syntax. For instance, the embed_text step has a set of pre-made configs in `global_mining/conf/embed_text` and you can swap between them. If you would like to make a new reusable/shared config for embed_text, you could put a new yaml file in that that folder (let say `global_mining/conf/embed_text/foobar.yaml`) and select it from the cli with:


```bash
python global_mining_pipeline.py src_lang=bn tgt_lang=hi +data=ccg embed_text=foobar
```


See the Data and Modules discussion below for more examples.


## Outputs and Working Dir

The output of the pipeline is set in the global_mining.yaml to be ".", which means the current working directory. When running `global_mining_pipeline.py` it will by default create a new folder under `outputs/today_date/timeofrun` and make this your working directory. This means all your logs will be well organized. It also means that the main output of each step will go under that directory given the default configuration of `output_dir: .`

Because you might run the pipeline multiple times for the same "data run" (e.g. if it fails with pre-emption in the middle, etc.), this default config means that you might end up with data spread across multiple date/time directories.

It's therefore a good idea when you are doing a full run (not just testing), to specify a fixed outputs directory when launching the pipeline:


```bash
python global_mining_pipeline.py src_lang=bn tgt_lang=hi +data=ccg output_dir=/myfinal/data/outputs
```


This way logs and other temp files will go to the working directory, but the data will go to a clean central place.


## Data

The current data configuration for the pipeline takes a few parameters:



* data_version
* iteration
* data_shard_dir
* shard_type
* bname

Because you will most often always use the same data for your runs, there is no need to specify this every time on the CLI or in the default config. There is a "group" under `global_mining/conf/data` where you can put common data sources. Checkout the demo config to see how to configure data. You can create a data config folder if you want to switch data without changing all other presets.


# Modules

The pipeline is made of seven main steps:

* split_in_shards (optional)
* embed_text
* train_index
* populate_index
* merge_index
* calculate_distances
* mine_indexes
* mine_sentences
* merge_shards (optional)

Each of them is configured as a "group" and their configurations can be overridden by switching groups on the cli as explained above. This override can also completely switch the code/module that is being used to compute this step, without changing the pipeline itself.


**Embedding Modules**

You can switch the actual encoder being used to choose between multiple encoders. For example, you can choose to use LaBSE, BERT, RoBERTa, or any other model from the sentence-transformers repo within the HuggingFace Model Hub ([https://huggingface.co/sentence-transformers](https://huggingface.co/sentence-transformers)). Here’s an example of how to encode text using LaBSE (with encoder-specific options in blue):


```bash
python global_mining_pipeline.py src_lang=bn tgt_lang=hi +data=ccg  embed_text=hf_roberta_large
```

```bash
python global_mining_pipeline.py src_lang=bn tgt_lang=hi +data=ccg  embed_text=hf_labse
```

or you can choose any huggingface encoder by their name with:
```bash
python global_mining_pipeline.py -c job src_lang=bn tgt_lang=hi +data=ccg  embed_text=huggingface embed_text.encoder_model=sentence-transformers/LaBSE
```

These are shortcuts to common models, but you can switch to any other model in the HuggingFace Model Hub, see `hf_labse.yaml` for an example of how to change config.encoder.encoder_model. To utilise LASER you can use the following example command:

```bash
python global_mining_pipeline.py -c job src_lang=bn tgt_lang=hi +data=ccg embed_text=laser2
embed_text.config.encoder.encoder_model=path_to_laser_model
embed_text.config.encoder.spm_model=path_to_spm_model
```


### Splitting and merging languages
For some large languages, the mining might fail because of out-of-memory errors, especially if the FAISS indexes are stored on GPU. To mitigate this probelm, you can split a language into shards, perform the mining on them in parallel, and then merge the results. 

The first optional module, `split_in_shards`, can randomly split the language (inclusing both text files and metadata files, if they exist) into several shards. 
To use this option, you should specify the parameter `max_shard_size`, and the languages with more total lines than this number will be automatically split into smaller shards. 

Alternatively, you can manually split the data for the language and configure it as several separate "languages", e.g. `eng0,eng1,eng2`. In this case, you can indicate in the mining config that they should be merged into a single language after mining:
```
sharded_langs:
  eng:
    - eng0
    - eng1
    - eng2
```

When you provide `max_shard_size` or `sharded_langs`, the module `merge_shards` is called in the end of the pipeline. It merges the mined bitexts for the sharded languages together and sorts the results (both text and meta files) in the order of decreasing alignment score.

**Note** that the mined sentence pairs are filtered by the [margin score](https://aclanthology.org/P19-1309/) that depends not only on the sentences themselves, but also on their nearest neighbours in the dataset. Therefore, when you mine parallel sentences from subsets of a language and then merge them, you may end up with more sentence pairs than if you mined from the whole language at once. This effect may be countered by adjusting the mining threshold or by filtering the sentence pairs after the mining.

# Sweeping (multi-run)

One of the benefits of the hydra cli override syntax, is that you can ask hydra to try different variations of the configuration with a simple command line. Hydra calls this ["multi-run"](https://hydra.cc/docs/1.0/tutorials/basic/running_your_app/multi-run/) and it lets you specify variations to your config that you would like to try.

For instance, if you would like to run the pipeline on multiple languages, you can do:


```bash
python global_mining_pipeline.py -m src_lang=en tgt_lang=bn,hi +data=ccg
```


The `-m` parameter tells the pipeline to start with multi-run and `tgt_lang=bn,hi` tells it to make two runs, one for en-bn and one for en-hi.

 You could also sweep over the lang and the encoders with:

```bash
python global_mining_pipeline.py -m src_lang=en tgt_lang=bn,hi +data=ccg embed_text=hf_roberta_large,hf_labse
```

# NMT model training on mined bitexts

Once a mined bitext has been produced, `stopes` can then run an end-to-end bilingual NMT system. It follows the following steps:

1. Takes as input a mined bitext (format: alignment-score [tab] text [tab] text)
2. Applies threshold filters based on alignment score and max number of alignments to use for training
3. Applies moses preprocessing (on bitext only)
4. Trains spm (on bitext only)
5. Spm-encodes bitext and chosen evaluation data
6. Binarizes files for fairseq
7. Trains bilingual NMT using `fairseq-train` on binarized data
8. Runs `fairseq-generate` on binarized evaluation data for all model checkpoints
9. Calculates BLEU scores

# Run it

```
python -m stopes.pipelines.bitext.nmt_bitext_eval                   \
src_lang=lin_Latn tgt_lang=eng_Latn                                 \
input_file_mined_data_tsv=/path/to/your/bitext.tsv                  \
preproc_binarize_mined.test_data_dir.dataset_name=flores200         \
preproc_binarize_mined.test_data_dir.base_dir=/path/to/flores200    \
output_dir=/directory/to/store/preprocessed/data/and/checkpoints    \
launcher.cache.caching_dir=/path/to/cache                           \
maximum_epoch=20
```

**NOTE**: In order for the training pipeline to know which column of the bitext corresponds to the selected `src_lang` and `tgt_lang`, it presumes that the two text columns in the bitext are ordered by their sorted language names. For example, for a `eng-lin` bitext, the format is: alignment-score [tab] english-text [tab] lingala-text (not alignment-score [tab] lingala-text [tab] english-text). 

## Outputs

The NMT pipeline will create the following directories in the specified `output_dir`:
- `bin_dir`: moses preprocessed, spm-encoded, and binarized data.
- `trained_models`: checkpoints from `fairseq-train`. **Note**: this directory will also contain files containing the outputs of both `fairseq-generate` (files ending in `.out`) and the corresponding BLEU evaluations for each checkpoint (files ending in `.bleu`).

## Evaluation data

To find the evaluation data for your chosen languages, `stopes` needs to know the relevant path. See `path` in `stopes/pipelines/bitext/conf/preproc_binarize_mined/standard_conf.yaml`. Currently it defaults to the format of the `flores200` dataset. To use this, please [download flores200](https://github.com/facebookresearch/flores/tree/main/flores200). 

## Example overrides

**Spm training**

```spm.train.config.vocab_size=7000```

**Model configuation**

```
train_fairseq.config.params.optimization.lr=[0.0001]
train_fairseq.config.params.optimization.update_freq=[8]
train_fairseq.config.params.model.encoder_layers=6
train_fairseq.config.params.model.encoder_embed_dim=512
train_fairseq.config.params.model.dropout=0.3
train_fairseq.config.num_gpus=8
```
