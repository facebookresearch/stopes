# NLLB Data Preparation

*NB*: This is legacy code that is older than the rest of `stopes`. It has not been
ported yet -- do not depend on it as it will eventually be refactored.

This pipeline binarizes training data and takes care of sharding as well as of SPM
training, where applicable.

It may be run as follows:
```
python stopes/pipelines/prepare_data/prepare_data.py \
  --data-config $CONFIG_PATH \
  --data-path $DATA_PATH \
  --output-dir $OUTPUT_PATH
```

See below for the format of the configuration file as well as the expected directory
structure of `$DATA_PATH`.

## Config format

The `--data-config` argument should point to a YAML file with the following format:
```
train_corpora:
  - <LANGUAGE_DIRECTION>/<CORPORA_NAME>
  ...
train_mining_corpora:
  - <LANGUAGE_DIRECTION>/<CORPORA_NAME>
  ...
train_smt_bt_corpora:
  - <LANGUAGE_DIRECTION>/<CORPORA_NAME>
  ...
train_mmt_bt_corpora:
  - <LANGUAGE_DIRECTION>/<CORPORA_NAME>
  ...
valid_corpora:
  - <LANGUAGE_DIRECTION>/<CORPORA_NAME>
  ...
test_corpora:
  - <LANGUAGE_DIRECTION>/<CORPORA_NAME>
  ...

source_vocab_config:
    pretrained:
        model_file: <PATH_TO_SRC_MODEL_FILE>
        vocab_file: <PATH_TO_SRC_VOCAB_FILE>

target_vocab_config:
    pretrained:
        model_file: <PATH_TO_TGT_MODEL_FILE>
        vocab_file: <PATH_TO_TGT_VOCAB_FILE>

binarization_config:
  max_examples_per_shard: <MAX_EXAMPLES_PER_SHARD>
  smallest_shard: <MINIMUM_NUMBER_OF_EXAMPLES_FOR_EACH_SHARD>

preprocessing_config:
  moses_config:
    script_directory: <PATH_TO_MOSES_SCRIPTS>
    lowercase: false
    normalize_punctuation: true
    remove_non_printing_chars: false
    deescape_special_chars: false

executor_config:
  cluster: <CLUSTER_CONFIG>
 ```

The option `preprocessing_config.moses_config.script_directory` will need to be set to
the path which contains the following MOSES perl scripts: `clean-corpus-n.perl`,
`deescape-special-chars.perl`, `lowercase.perl`, `normalize-punctuation.perl`,
`remove-non-printing-char.perl`. They are provided by the
[MOSES system](https://github.com/moses-smt/mosesdecoder).

## Data path format

The `--data-path` argument should point to the directory that contains all corpora. This
directory is expected to have subdirectories corresponding to training directions in
the `$src-$tgt` format, each containing the corpora. Corpora are expected to be made up
of two gzipped files, `$corpus_name.$src.gz` and `$corpus_name.$tgt.gz`.

Here is an example directory structure:
```
$ tree $DATA_PATH
my_corpora
├── arb_Arab-eng_Latn
│   ├── mycorpus.arb_Arab.gz
│   └── mycorpus.eng_Latn.gz
└── eng_Latn-lij_Latn
    ├── nllbseed.eng_Latn.gz
    ├── nllbseed.lij_Latn.gz
    ├── tatoeba.eng_Latn.gz
    └── tatoeba.lij_Latn.gz
```

## Output file format

These files live in the subdirectory `$OUTPUT_PATH/shard$SHARD_ID/data_bin`.  The dict
files and the sample_train files for each train corpora are excluded for succinctness:

```
train.<LANGUAGE_DIRECTION>.<SRC>.bin
train.<LANGUAGE_DIRECTION>.<SRC>.idx
train.<LANGUAGE_DIRECTION>.<TGT>.bin
train.<LANGUAGE_DIRECTION>.<TGT>.idx
...
train_mining.<LANGUAGE_DIRECTION>.<SRC>.bin
train_mining.<LANGUAGE_DIRECTION>.<SRC>.idx
train_mining.<LANGUAGE_DIRECTION>.<TGT>.bin
train_mining.<LANGUAGE_DIRECTION>.<TGT>.idx
...
train_smt_bt.<LANGUAGE_DIRECTION>.<SRC>.bin
train_smt_bt.<LANGUAGE_DIRECTION>.<SRC>.idx
train_smt_bt.<LANGUAGE_DIRECTION>.<TGT>.bin
train_smt_bt.<LANGUAGE_DIRECTION>.<TGT>.idx
...
train_mmt_bt.<LANGUAGE_DIRECTION>.<SRC>.bin
train_mmt_bt.<LANGUAGE_DIRECTION>.<SRC>.idx
train_mmt_bt.<LANGUAGE_DIRECTION>.<TGT>.bin
train_mmt_bt.<LANGUAGE_DIRECTION>.<TGT>.idx
...
test.<LANGUAGE_DIRECTION>.<SRC>.bin
test.<LANGUAGE_DIRECTION>.<SRC>.idx
test.<LANGUAGE_DIRECTION>.<TGT>.bin
test.<LANGUAGE_DIRECTION>.<TGT>.idx
...
valid.<LANGUAGE_DIRECTION>.<SRC>.bin
valid.<LANGUAGE_DIRECTION>.<SRC>.idx
valid.<LANGUAGE_DIRECTION>.<TGT>.bin
valid.<LANGUAGE_DIRECTION>.<TGT>.idx
```

## How to add more sources of training data?

Let's assume `train_fold_corpora` is the new training corpora you're adding. You will need to adhere to the nomenclature of calling any additional training corpora `training_<FOLD_NAME>_corpora`. Following these steps, you can add as many additional training corpora as possible:

* Create a file `train_fold.yaml` inside `stopes/pipelines/data` which is where you will place all the corpus source & target files for all the language directions for your `train_fold_corpora`. The prepare_data.py script looks for this specific file. Format for `train_fold.yaml`:
    ```
    <LANGUAGE_DIRECTION>:
        <CORPORA_NAME>:
            is_gzip: <BOOLEAN_VALUE>
            source: <PATH_TO_SOURCE_CORPORA_FILE>
            target: <PATH_TO_TARGET_CORPORA_FILE>
        ...
    ...
    ```
* Modify the `DataConfig` dataclass definition in `stopes/pipelines/prepare_data/data_types.py` to include:
    ```
    train_fold_corpora: Optional[Dict[str, CorporaMap]] = None
    ```
* Add the following to your data_config with all the corpus names for all the language directions you wish to prepare data for:
    ```
    train_fold_corpora:
    - <LANGUAGE_DIRECTION>/<CORPORA_NAME>
    ...
    ```
