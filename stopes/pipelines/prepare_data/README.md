# NLLB prepare_data pipeline

This pipeline takes in the filtered corpora text files (can be compressed), trains an SPM model, deduplicates, shards, encodes & binarizes them in the format required by fairseq. An array of jobs is scheduled wherever possible, in particular for validate, retrieve_data, dedup_sharding and binarizing. The pipeline uses the caching feature of Stopes.

## Input Config:

* fold: train, train_mining, train_mmt_bt, train_smt_bt, valid, test are possible options
* lang_dir: language direction

corpora: `CorporaConfig`
    <FOLD>:
        <LANG_DIR>:
            <CORPUS_NAME>:
                src: <SRC_FILE_PATH>
                tgt: <TGT_FILE_PATH>
                metadata: <METADATA_FILE_PATH> (optional)
            ...
        ...
    ...
Specify paths to src, tgt, and optionally metadata files per (fold, lang_dir) for each corpus.

preprocessing: `PreprocessingConfig`
Specify boolean values for MOSES preprocessing.

vocab: `VocabConfig`
Specify vocab params for src, tgt vocab. By default vocab is trained jointly, in which case we only use the src_vocab config for the joint vocab.

dedup: `DedupConfig`
How to deduplicate? Whether across folds and how to use individual sentences for deduplication.

sharding: `ShardingConfig`
How to shard? Total sentences per shard and the minimum number of sentences for each lang_dir per shard. Also the number of workers to binarize sharded files in a Multiproc way.

launcher:
How to launch your jobs? locally or submitit

## Run Command:

Please override the default config options as required.
```
python stopes/pipelines/prepare_data/prepare_data.py output_dir=<OUTPUT_DIR>
```

## Pipeline Breakdown

* validate: Counts the number of lines for all parallel corpora and makes sure they're the same for src & tgt and stores train line counts statistics.
* retrieve_data: Concatenates all corpora for each (fold, lang_dir), runs Moses preprocessing over each of them as per preprocessing config and saves them to the `retrieved_data` directory.
* build_vocab: Samples a corpus as per sampling_config and trains an SPM on the sampled corpus. We need to sample a corpus since training an SPM on all of the corpora is time consuming. This is done jointly for src, tgt directinos by default but can be done separately as well. The trained SPM, the model file and vocab file are saved in the `vocab_bin` directory
* dedup_sharding: Deduplicates training corpora across eval corpora (valid, test) & optionally across folds as per dedup_config and shards training corpora.
* binarize: Binarizes all the sharded files (train, eval) using `MultiProcFairSeqBinarizerEncoder` and writes them to the sharded directories in the `data_bin` directory.

## Caveat

This pipeline doesn't work if metadata is not specified for all corpora for a (fold, lang_dir) because we concatenate all corpora files for each (fold, lang_dir) into one file and shard them. So we need metadata information for every one of these lines, if specified at all for the (fold, lang_dir).
