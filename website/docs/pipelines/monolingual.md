---
sidebar_position: 2
---

# NLLB Monolingual Pipeline

This is the monolingual "cleaning" pipeline, it does a few things:

1. split paragraphs in sentences
2. run some moses normalization+cleaning on the sentences
3. filter the sentences that do not match some criteria (length, character ratios, etc.)
4. run script detection at the sentence level, if this doesn't match the expected lang, throw the sentence out
5. run lid detection at the sentence level, if this doesn't match the expected lang, throw the sentence out
6. deduplicate sentences (this is done by sorting sentences)

The core filtering is in `monolingual_line_processor.py` and `utils/text_filter.py`

## Run it

`python monolingual_pipeline.py data_dir=yourdatahere langs='[umb,ssw]'`

should be enough to get it running.
- `data_dir` is where the raw data is, should have subfolders per lang and files named with the pattern corpus_name.lang.xz
- `langs` an array of langs to process in this run

## Usefull overrides

- `launcher.cluster=local local_tmp_dir=/tmp/monolingual` if you want to run this locally instead of on the slurm
- `preproces_requirements.cpus_per_task=40` this is the number of CPUs used to process each lang file in a slurm job. Higher means it will go faster, but you'll have a harder time to get a machine from the queue
- `corpus_filter=yourcorpus` filter the lang files you'll process to only work on a specific corpus
- `input_file_glob_template` replace this if the files in your data_dir do not follow the expected template

See `monolingual.yaml` for more possible configurations.

## Outputs

The run will be started with a custom working directory that follows the pattern: `outputs/{date}/{start_time}`, all the logs will go there (including executor_logs from slurm jobs). By default, the data output is set in `monolingual.yaml` to be `output_dir: .` this means that the outputs will go to lang dirs in the working directory and will go to different places depending on the day/time you start the run. This is useful for testing, but if you want to output somewhere else (like a central clean monolingual repo), override the `output_dir=/somethingstable/` when starting the run.

## Logging

the run will log to wandb monolingual dashboard. Go to wanddb and make sure to enable grouping. Choose to group by "group" and "lang". There will be one sub-run per process (see num_cpu above) per lang + a global run for the root script. The global run will have a funny name and will only report data at the end of everything. You can check progress in each subrun.
