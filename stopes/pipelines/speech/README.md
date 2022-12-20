# Speech Pipelines

## compute_laser_embeddings

This pipeline takes in audio data tsv files for multiple language directions and computes LASER embeddings for each language direction using the SpeechLASER model used for mining SpeechMatrix data. For each language direction, the pipeline first splits the audio data tsv file into chunks and computes the laser embeddings for each chunk separately on a node with 1 GPU asynchronously and saves a `.embeddings` file for each chunk.

The format for each input audio data tsv file is:
```
<root_dir>
<filename>:<offset>:<length>\t<num_frames>
<filename>:<offset>:<length>\t<num_frames>
...
...
```

We run the pipeline using the command:
```
python stopes/pipelines/speech/compute_laser_embeddings.py
```

The input config:
```
@dataclass
class LaserEmbeddingConfig:
    launcher: DictConfig
    max_tokens: int
    checkpoint_dir: Path = MISSING
    data_dir: Path = MISSING
    num_chunks: int = MISSING
    lang_dirs: str = MISSING
    out_dir: Path = MISSING
```

Parameters:
* `launcher`: Config for the Stopes launcher, either `submitit` or `local`. Make sure you specify the partition for the launcher if you're using the `submitit` launcher.
* `max_tokens`: Determines the effective batch size for feeding in the audio waveforms. Needs to be tuned to make sure we don't OOM on the GPU.
* `checkpoint_dir`: Path to the checkpoint directory of the SpeechLASER models.
* `data_dir`: Path to the audio data tsv files in the format `<lang_dir>_<lang>.tsv`.
* `num_chunks`: number of chunks to split the audio data tsv files.
* `lang_dirs`: comma separated string of language directions. Ex: `hr-en,ro-en,es-en`
* `out_dir`: Path to the output directory to save the embedding files.
