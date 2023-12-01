# Overview

Stopes pipeline to parallelize and evaluate M4T models on all supported tasks. The pipeline is based on the [evaluation script](https://github.com/facebookresearch/seamless_communication/blob/main/src/seamless_communication/cli/m4t/evaluate/evaluate.py) from seamless_communication repo.

# Quick Start

```bash
python3 /stopes/pipelines/m4t_eval/m4t_eval.py output_dir=<output_dir> launcher.partition=<slurm_partition>
```

## Description of variables used in the config files

1. output_dir: Path of the folder where the generated metric files are to be saved
2. data_dir: Root folder to find the input tsv manifests
3. task: task to evaluate on
4. dataset_split: Dataset split and prefix of the input manifest to evaluate on. For ex: test_fleurs
5. audio_root_dir: Root folder where the audio zips are located.
6. lang_dirs: List of language directions to evaluate on
7. kwargs: Additional arguments to the evaluate.py script can be added to this dictionary
