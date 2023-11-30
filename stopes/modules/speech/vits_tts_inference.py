# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import importlib
import io
import logging
import re
import string
import typing as tp
import zipfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from omegaconf.omegaconf import MISSING

from stopes.core.stopes_module import Requirements, StopesModule
from stopes.ust_common.generation.tts_utils import VITS_Synthesizer
from stopes.ust_common.lib.manifests import create_zip_manifest
from stopes.utils.tts_preprocessing.cmn import CHINESE_PUNC_LIST

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
PUNCTUATIONS = set(CHINESE_PUNC_LIST + string.punctuation)


@dataclass
class ProsodyLabel:
    action: str = MISSING
    value: int = MISSING
    text: str = MISSING

    def split_by(self, action):
        pattern = re.compile(rf"\s?\<{action}=(\d+)(?:ms|%)\>\s?")
        labels = set(pattern.findall(self.text))
        splits = pattern.split(self.text)
        ret = []
        for text in splits:
            if text in labels:
                ret.append(
                    ProsodyLabel(
                        text="",
                        action=action,
                        value=int(text),
                    )
                )
            else:
                ret.append(
                    ProsodyLabel(
                        text=text,
                        action=self.action,
                        value=self.value,
                    )
                )
        return ret


def parse_prosody_labels(text) -> tp.List[ProsodyLabel]:
    """
    Parse a string of prosody-annotated text* into a list of prosody labels
    * Definition follows AI speech's ctts format, same action couldn't be nested
    """
    pattern = re.compile(r"{([^\{\}]+)}\<(\w+)=(\d+)(?:ms|%)\>")
    prosody_labels = pattern.findall(text)
    prosody_labels = list(
        map(
            lambda x: ProsodyLabel(
                text=x[0],
                action=x[1],
                value=int(x[2]),
            ),
            prosody_labels,
        )
    )
    ret = []
    for prosody_label in prosody_labels:
        ret += prosody_label.split_by("silence")

    return ret


@dataclass
class VITSInferenceConfig:
    model_path: Path = MISSING
    input_tsv: Path = MISSING
    output_dir: Path = MISSING
    text_col: str = MISSING
    id_col: str = "id"
    lang: str = "cmn"
    nshards: int = 10
    text_cleaner: str = ""
    prosody_manipulate: bool = False
    silero_vad_prob: float = 0.3
    torch_hub_cache: str = ""


class VITSInferenceModule(StopesModule):
    def __init__(self, config: tp.Any):
        super().__init__(config, VITSInferenceConfig)
        self.input_tsv = pd.read_csv(self.config.input_tsv, quoting=3, sep="\t")
        self.synthesizer = VITS_Synthesizer(
            self.config.model_path.as_posix(),
            dict_size_shift=0,
            skip_text_cleaners=True,
        )
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.text_cleaner = []
        if len(self.config.text_cleaner):
            self.text_cleaner = self.config.text_cleaner.strip().split()
            cleaner_package = importlib.import_module("stopes.ust_common.text.cleaners")
            for i, cleaner in enumerate(self.text_cleaner):
                self.text_cleaner[i] = getattr(cleaner_package, cleaner)

    def requirements(self) -> Requirements:
        return Requirements(gpus_per_node=1, constraint="volta32gb")

    def init_silero_vad(self):
        """
        copied from stopes/ust_common/lib/audio.py,
        to remove global variable "_vad", since it breaks slurm jobs
        """
        torch.set_num_threads(1)
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.vad_model, utils = torch.hub.load(
            repo_or_dir=f"{self.config.torch_hub_cache}/snakers4_silero-vad_master/",
            model="silero_vad",
            source="local",
            onnx=True,
        )
        self.get_speech_timestamps, self.collect_chunks = utils[0], utils[-1]

    def apply_silero_vad_audio(self, audio, sr):
        for thr in torch.arange(self.config.silero_vad_prob, 0.0, -0.02):
            speech_tss = self.get_speech_timestamps(
                audio, self.vad_model, threshold=thr, sampling_rate=sr
            )
            if len(speech_tss) > 0:
                break

        # if VAD failed, we return None here and treat that audio as silence only
        if len(speech_tss) > 0:
            vad_audio = self.collect_chunks(speech_tss, audio.view(-1)).unsqueeze(0)
        else:
            vad_audio = torch.tensor([[0.0]], device="cpu")

        return (vad_audio, sr)

    def array(self):
        if self.config.nshards > 1:
            return np.array_split(self.input_tsv, self.config.nshards)
        return None

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> tp.Any:
        if self.config.prosody_manipulate:
            self.init_silero_vad()

        input_tsv = self.input_tsv
        if iteration_value is not None:
            input_tsv = iteration_value

        output_zip = self.config.output_dir / (f"shard{iteration_index}" + ".zip")
        with contextlib.ExitStack() as stack:
            # append to worker's zip file, to avoid writing small files into NFS
            zip_obj = stack.enter_context(
                zipfile.ZipFile(
                    output_zip, "w", compression=zipfile.ZIP_STORED, compresslevel=0
                )
            )
            for _, row in input_tsv.iterrows():
                text = row[self.config.text_col]
                logger.info(text)
                if not self.config.prosody_manipulate:  # input is pure text
                    for cleaner in self.text_cleaner:
                        text = cleaner(text)

                    logger.info(text)
                    audio = self.synthesizer.synthesize(text=text, need_waveform=True)

                else:  # input is prosody-annotated markup-ed text
                    audios = []
                    prosody_labels = parse_prosody_labels(text)
                    for prosody_label in prosody_labels:
                        if prosody_label.action == "silence":
                            audios.append(
                                torch.zeros(
                                    1, SAMPLE_RATE * prosody_label.value // 1000
                                )
                            )

                        elif prosody_label.action == "rate":
                            for cleaner in self.text_cleaner:
                                prosody_label.text = cleaner(prosody_label.text)

                            logger.info(prosody_label.text)
                            if all(t in PUNCTUATIONS for t in prosody_label.text):
                                continue

                            audio = self.synthesizer.synthesize(
                                text=prosody_label.text, need_waveform=True
                            )
                            audio = self.apply_silero_vad_audio(
                                torch.FloatTensor(audio), SAMPLE_RATE
                            )[0]
                            audio = torchaudio.sox_effects.apply_effects_tensor(
                                audio.to(torch.int16),
                                SAMPLE_RATE,
                                [["tempo", str(prosody_label.value / 100)]],
                            )[0]
                            audios.append(audio)

                        else:
                            raise NotImplementedError

                    if len(audios) == 0:
                        logger.info("ERROR", row[self.config.id_col], text)
                        continue

                    audio = torch.cat(audios, dim=-1).squeeze(0).to(torch.int16)

                # transform audio tensor into bytes string
                buffer = io.BytesIO()
                sf.write(buffer, audio, SAMPLE_RATE, format="wav")
                buffer.seek(0)
                audio_bytes = buffer.read()
                zip_obj.writestr(row[self.config.id_col] + ".wav", audio_bytes)

        # convert audio zip into a manifest
        manifest = create_zip_manifest(output_zip, is_audio=True, n_workers=10)
        manifest = pd.DataFrame.from_dict(manifest, orient="index").reset_index(
            names="id"
        )
        manifest["id"] = manifest["id"].apply(lambda x: ".".join(x.split(".")[:-1]))
        manifest = pd.merge(manifest, input_tsv, on="id")
        manifest["audio"] = (
            output_zip.as_posix()
            + ":"
            + manifest["offset"].astype(str)
            + ":"
            + manifest["size"].astype(str)
        )
        manifest["duration"] = manifest["length"] / SAMPLE_RATE
        return manifest
