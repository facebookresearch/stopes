from dataclasses import dataclass


@dataclass
class CorporaConfig:
    lang: str
    audio_dirpath: str
    reference_path: str
    reference_format: str
    reference_tsv_column: str = None
    audio_format: str = "n_pred.wav"
    results_dirpath: str = None
    transcripts_path: str = None


@dataclass 
class AsrBleuConfig:
    corpora: CorporaConfig
    asr_version: str = "oct22"
