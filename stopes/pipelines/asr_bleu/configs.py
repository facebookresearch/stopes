import typing as tp
from dataclasses import dataclass


@dataclass
class CorporaConfig:
    lang: str
    audio_dirpath: str
    reference_path: str
    reference_format: str
    reference_tsv_column: str
    results_dirpath: str
    transcripts_path: str
    audio_format: str = "n_pred.wav"
    asr_version: str = "oct22"


@dataclass
class AsrBleuConfig:
    launcher: tp.Dict[str, tp.Any]
    output_dir: str
    corpora: CorporaConfig
