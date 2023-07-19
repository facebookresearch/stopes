from dataclasses import dataclass

@dataclass
class AsrBleuConfig:
    lang: str
    audio_dirpath: str
    reference_path: str
    reference_format: str