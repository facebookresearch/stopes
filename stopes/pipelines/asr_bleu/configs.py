import typing as tp
from dataclasses import dataclass


@dataclass
class Dataset:
    lang: str
    config_key: str
    audio_dirpath: str
    reference_path: str
    reference_format: str
    reference_tsv_column: str
    results_dirpath: str
    transcripts_path: str
    asr_version: str
    audio_format: str = "n_pred.wav"


@dataclass
class CorporaConfig:
    datasets: tp.Dict[str, Dataset]


@dataclass
class AsrBleuConfig:
    launcher: tp.Dict[str, tp.Any]
    output_dir: str
    corpora: CorporaConfig
