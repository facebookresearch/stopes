---
sidebar_position: 3
---

# NLLB Distillation Pipeline

Welcome to `stopes`, and thanks for checking out our sequence-level knowledge distillation pipeline. This is a quick start guide which walks through how to run the pipeline yourself and what the expected outputs will be from each step. The logic of the pipeline is at a high level as follows:
1. cleans pre-downloaded monolingual data (see [STOPES monolingual pipeline](https://github.com/fairinternal/nllb/blob/main/website/docs/pipelines/monolingual.md#nllb-monolingual-pipeline)) - results in one merged file of data for each source language
2. shards each source language file from previous step into as many shards as number of specified target languages
3. generates target language translations for each shard from previous step using Fairseq Generate
4. cleans generated target language data and removes corresponding sentences from source language file used to generate translation
5. binarizes and encodes the cleaned bitext data from previous step
6. trains student model using the binarized distilled data from previous step

## To run:

First, fill out any missing fields in distillation.yaml (labeled ???). Then,
`python stopes/pipelines/distillation/distillation_pipeline.py` should be enough to get it running.

You can also override distillation.yaml fields manually through the CLI as such:
`python stopes/pipeliens/distillation/distillation_pipeline.py src_langs="[eng,mai]" tgt_langs="[fra,deu]" mono_data_dir=<path_to_predownloaded_mono_data> output_dir=<path_to_output_dir>`.

For internal FAIR users, feel free to add the `+fb_preset=nllb` argument to the CLI command to use some preset config settings.

Note: Testing performance can be done with a separate STOPES module, `/stopes/modules/evaluation/generate_multi_bleu_detok_module.py`.

## Useful overrides
- `src_langs` is an array of source languages you have pre-downloaded monolingual data for
- `tgt_langs` is an array of target languages you want to train the student model to translate to
- `mono_data_dir` is the path to pre-downloaded monolingual data
- `output_dir` is the path to the desired output directory of this pipeline run

- `skip_dedup=true launcher.cluster=local` if you want to run this locally instead of on the slurm
- `launcher.cluster=slurm` if you want to run this on the slurm instead of locally

See `distillation.yaml` and it's associated config sub groups for more possible configuration options.

## Pipeline outputs

Please be aware that at every intermediate step, the program will overwrite files with the same name (such as output from previous runs) so be sure to change the specified `output_dir` or rename past outputs between runs.

The run will be started with a custom working directory that follows the pattern: `outputs/{date}/{start_time}`, all the logs will go there (including executor_logs from slurm jobs). By default, the data output is set in `distillation.yaml` to be `output_dir: .` this means that the outputs will go to the working directory and will go to different places depending on the day/time you start the run. This is useful for testing, but if you want to output somewhere else (like a central clean monolingual repo), override the `output_dir=/somethingstable/` when starting the run.

### Raw input monolingual file:
```
~/test_inputs/eng
 % cat test.eng
Appealing Accent Chair And Ottoman and Petra Fabric Accent Chair With Ottoman Furniture Home Decoration - Lilangels Furniture	http://www.sample_placeholder_website.url/	placeholder.gz	placeholder_sha	0
no down payment auto insurance in Scottsdale AZ	http://www.sample_placeholder_website.url/	placeholder.gz	placeholder_sha    847
A Question of Priorities: Democratic Reform and Economic Recovery in Postwar Germany Auteur: Rebecca L. Boehling TiersD'occasion8,25€202,25€	50
202-458-1769 Joie Olverson - Spring House Ln, Washington, District of Columbia	http://www.sample_placeholder_website.url/	placeholder.gz	placeholder_sha	8125
BlackBerry Z10 To Launch In South Africa Tomorrow - Blackberry Empire	http://www.sample_placeholder_website.url/	placeholder.gz	placeholder_sha   0
```

### Example file output of monolingual_pipeline before dedup:
Parsed in column format:
                    1. self.corpus,  # the original corpus name
                    2. self.offset_start,  # skip that many bytes (use dd)
                    3. line_id,  # after skipping, go to line
                    4. line_hash,  # xxhash.xxh3_64 of the original line/paragrph
                    5. f"{prob_lang:.5f}",  # lid score
                    6. clean, # sentence
                    # config
                    sep="\t"

```
~/test_outputs/mono_data/eng
 % cat test.eng.000.sorted (processed and kept lines)
test	1056	0	4426603632439174366	0.71947	202-458-1769 Joie Olverson - Spring House Ln, Washington, District of Columbia
test	692	0	8327890826167111651	0.83095	A Question of Priorities: Democratic Reform and Economic Recovery in Postwar Germany Auteur: Rebecca L. Boehling TiersD'occasion8,25€202,25€
test	0	0	12930410217004390762	0.90479	Appealing Accent Chair And Ottoman and Petra Fabric Accent Chair With Ottoman Furniture Home Decoration - Lilangels Furniture
test	443	0	3451732902557484365	0.83896	no down payment auto insurance in Scottsdale AZ
```

```
~/test_outputs/mono_data/eng
% cat _discarded.test.eng.000.sorted (discarded lines)
test	0	__label__eng	0.37420	BlackBerry Z10 To Launch In South Africa Tomorrow - Blackberry Empire
```

### Example file output of dedup
```
% cat eng_all_dedup
test	1056	0	4426603632439174366	0.71947	202-458-1769 Joie Olverson - Spring House Ln, Washington, District of Columbia
test	692	0	8327890826167111651	0.83095	A Question of Priorities: Democratic Reform and Economic Recovery in Postwar Germany Auteur: Rebecca L. Boehling TiersD'occasion8,25€202,25€
test	0	0	12930410217004390762	0.90479	Appealing Accent Chair And Ottoman and Petra Fabric Accent Chair With Ottoman Furniture Home Decoration - Lilangels Furniture
test2	0	0	5374428323341487497	1.00001	He has a cat.
test2	0	0	5374428323341487497	0.99987	Hello the president is here!
test	443	0	3451732902557484365	0.83896	no down payment auto insurance in Scottsdale AZ
```

### Example file output of shard
```
% cat shard.000
test	1056	0	4426603632439174366	0.71947	202-458-1769 Joie Olverson - Spring House Ln, Washington, District of Columbia
test	692	0	8327890826167111651	0.83095	A Question of Priorities: Democratic Reform and Economic Recovery in Postwar Germany Auteur: Rebecca L. Boehling TiersD'occasion8,25€202,25€
test	0	0	12930410217004390762	0.90479	Appealing Accent Chair And Ottoman and Petra Fabric Accent Chair With Ottoman Furniture Home Decoration - Lilangels Furniture
test2	0	0	5374428323341487497	1.00001	He has a cat.
test2	0	0	5374428323341487497	0.99987	Hello the president is here!
test	443	0	3451732902557484365	0.83896	no down payment auto insurance in Scottsdale AZ
```

### Example file output of generate
Target generated data:
```
test 1056 0 4426603632439174366 0.71947 202-458-1769 Joie Olverson - Spring House Ln, Washington, District de Columbia
test 692 0 8327890826167111651 0.83095 Une question de priorités: réforme démocratique et reprise économique en Allemagne d'après-guerre Auteur: Rebecca L. Boehling TiersD'occasion8,25€202,25€
test 0 0 12930410217004390762 0.90479 Chaise d'accent attrayante et chaise d'accent du tissu ottoman et du tissu de Petra avec meubles ottomans décoration de la maison - Lilangels meubles
test2 0 0 5374428323341487497 1.00001 Il a un chat.
test2 0 0 5374428323341487497 0.99987 Bonjour le président est ici !
test 443 0 3451732902557484365 0,83896 aucun acompte d'assurance automobile à Scottsdale AZ
```

### Example file output of bitext clean
The contents of the filtered `clean.eng-fra.eng.000.xz` and `clean.eng-fra.fra.000.xz` files are respectively:
```
test	692	0	8327890826167111651	0.83095	A Question of Priorities: Democratic Reform and Economic Recovery in Postwar Germany Auteur: Rebecca L. Boehling TiersD'occasion8,25€202,25€
test	0	0	12930410217004390762	0.90479	Appealing Accent Chair And Ottoman and Petra Fabric Accent Chair With Ottoman Furniture Home Decoration - Lilangels Furniture
test2	0	0	5374428323341487497	1.00001	He has a cat.
test2	0	0	5374428323341487497	0.99987	Hello the president is here!
test	443	0	3451732902557484365	0.83896	no down payment auto insurance in Scottsdale AZ
```
```
target_data	0	15	9889559120183218255	0.97691	Une question de priorités: réforme démocratique et reprise économique en Allemagne d'après-guerre Auteur: Rebecca L. Boehling TiersD'occasion8,25€202,25€
target_data	28	39	7358542291591603186	0.98684	Chaise d'accent attrayante et chaise d'accent du tissu ottoman et du tissu de Petra avec meubles ottomans décoration de la maison - Lilangels meubles
target_data	56	61	4587081072824752671	0.81006	Il a un chat.
target_data	56	68	13648239048374052831	0.99998	Bonjour le président est ici !
target_data	56	78	15942782228027469307	0.97898	aucun acompte d'assurance automobile à Scottsdale AZ
```

Meanwhile, the contents of the two discarded output files `discarded.eng-fra.eng.000.xz` and `discarded.eng-fra.fra.000.xz` are respectively:
```
test	1056	0	4426603632439174366	0.71947	202-458-1769 Joie Olverson - Spring House Ln, Washington, District of Columbia
```
```
gen_shard	0	__label__eng	0.32102	202-458-1769 Joie Olverson - Spring House Ln, Washington, District de Columbia
```

### Example file output of binarizing and encoding
```
train.eng-fra.eng.000.bin  train.eng-fra.eng.001.idx  train.eng-fra.eng.003.bin
train.eng-fra.eng.000.idx  train.eng-fra.eng.002.bin  train.eng-fra.eng.003.idx
train.eng-fra.eng.001.bin  train.eng-fra.eng.002.idx
```

### Example file output of train
```
-rw-rw-r-- 1 $USER $USER 4.2G Aug  3 12:05 checkpoint_best.pt
-rw-rw-r-- 1 $USER $USER 4.2G Aug  3 12:05 checkpoint_last.pt
```
