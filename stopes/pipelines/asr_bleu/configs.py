import typing as tp
from dataclasses import dataclass


@dataclass
class Dataset:
    lang_dir: str
    num_data_pairs: int


@dataclass
class DatasetsConfig:
    datasets: tp.Dict[str, Dataset]


@dataclass
class AsrBleuConfig:
    output_dir: str
    split: str
    model_name: str
    eval_first_pass: bool
    dataset: str
    audio_format: str
    launcher: tp.Dict[str, tp.Any]
    datasets: DatasetsConfig
