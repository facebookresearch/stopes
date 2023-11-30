# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
A module to apply Demucs denoising, i.e. separation of voice from background music.
Currently, it uses the demucs==4.0.0 implementation: https://github.com/facebookresearch/demucs.
"""

import logging
import typing as tp
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import demucs.apply
import demucs.pretrained
import demucs.separate
import torch
import torchaudio
import xxhash
from omegaconf import MISSING
from tqdm import tqdm

import stopes.modules.speech.utils as speech_utils
from stopes.core import utils
from stopes.core.stopes_module import Requirements, StopesModule

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000

TDemucsModel = tp.Union[demucs.apply.Model, demucs.apply.BagOfModels]


def calc_snr(lbl, est, eps=1e-8):
    """Compute signal-to-noise ratio of two waveforms: denoised and original."""
    y = 10.0 * torch.log10(
        torch.sum(lbl**2, dim=-1) / (torch.sum((est - lbl) ** 2, dim=-1) + eps) + eps
    )
    return y


def write_audio(wav, filename, sample_rate=16_000):
    """Normalize audio (if it prevents clipping) and save to disk."""
    wav = wav / max(wav.abs().max().item(), 1)
    torchaudio.save(
        filename, wav.cpu(), sample_rate, encoding="PCM_S", bits_per_sample=16
    )


@dataclass
class DenoiserConfig:
    """Configuration for a DenoiserModule.
    shards: list of tsv files with an input audio in each line, or a single such file.
    output_dir: the directory to put the resulting manifest and denoised audios
    dry_wet: proportion of the original audio (as opposed to the denoised one) in the output
    use_cuda: whether to run denoising on GPU
    compute_snr: whether to compute signal-to-noise ratio after denoising
    """

    shards: tp.Any = MISSING  # it is tp.Union[Path, tp.List[Path]], but such annotation is not supported by OmegaConf
    output_dir: Path = MISSING
    model_name: str = "mdx_extra"
    source_id: tp.Union[str, int] = "vocals"
    dry_wet: float = 0.01
    use_cuda: bool = True
    compute_snr: bool = False


class DenoiserModule(StopesModule):
    """A module to apply Demucs denoising (i.e. non-speech removal) to audios.
    Based on https://github.com/facebookresearch/demucs.
    """

    def __init__(self, config: DenoiserConfig = DenoiserConfig()):
        super().__init__(config, DenoiserConfig)
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = (
            "cuda" if self.config.use_cuda and torch.cuda.is_available() else "cpu"
        )
        self.model: tp.Optional[TDemucsModel] = None
        self.downsampler: tp.Optional[tp.Callable] = None
        self.source_id: tp.Optional[int] = None

    def load_model(self):
        """Load the Demucs model"""
        if self.model is None:
            self.model = demucs.pretrained.get_model(self.config.model_name)
            if self.model.samplerate != SAMPLE_RATE:
                self.downsampler = torchaudio.transforms.Resample(
                    orig_freq=self.model.samplerate, new_freq=SAMPLE_RATE
                )
        if isinstance(self.config.source_id, int):
            self.source_id = self.config.source_id
        else:
            self.source_id = self.model.sources.index(self.config.source_id)
            error = f"Source_id {self.config.source_id} not found among model sources {self.model.sources}"
            assert self.source_id is not None, error

    def array(self):
        if isinstance(self.config.shards, str):
            return None
        return self.config.shards

    def requirements(self) -> Requirements:
        return Requirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=1,
            cpus_per_task=1,
            timeout_min=180,
        )

    def load_audios(self, file: tp.IO) -> tp.Iterator[tp.Tuple[int, str, torch.Tensor]]:
        """
        Read audios from file and return resampled waveforms.
        """

        @lru_cache(maxsize=1024)
        def load_audio(audio: str) -> tp.Tuple[torch.Tensor, str]:
            audio_obj = speech_utils.parse_audio(audio, sampling_factor=16)
            # get string-based representation of audio segment
            audio_information = str(audio_obj)
            wav = audio_obj.load(average_channels=True)
            if wav.ndim > 1:
                wav = wav.squeeze(0)  # remove channel dimension
            return wav, audio_information

        for line_no, line in tqdm(enumerate(file)):
            line = line.rstrip()
            wav, audio_information = load_audio(line)
            yield line, audio_information, wav

    def denoise(
        self, audio: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, tp.Optional[float]]:
        """Apply denoising and potentially compute signal-to-noise ratio."""
        assert self.model is not None, "The model must be loaded first."
        wav = demucs.separate.convert_audio(
            audio.unsqueeze(0),
            SAMPLE_RATE,
            self.model.samplerate,
            self.model.audio_channels,
        )
        # estimate = self.model(audio)
        ref = wav.mean(0)
        wav_init = wav
        wav = (wav - ref.mean()) / ref.std()
        sources = demucs.separate.apply_model(
            self.model,
            wav[None],
            device=self.device,
            shifts=1,
            split=True,
            overlap=0.25,
            progress=False,
            num_workers=0,
        )[0]
        sources = sources * ref.std() + ref.mean()
        estimate = sources[self.source_id]
        estimate = (1 - self.config.dry_wet) * estimate + self.config.dry_wet * wav_init
        estimate = estimate.mean(0)  # averaging over the channels
        if self.downsampler:
            estimate = self.downsampler(estimate)

        snr = None
        if self.config.compute_snr:
            snr = calc_snr(audio, estimate)
            snr = snr.cpu().detach()[0][0].item()
        return (estimate.unsqueeze(0), snr)

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        if iteration_value is None:
            iteration_value = self.config.shards
        assert isinstance(
            iteration_value, (str, Path)
        ), "Input value must be a path to the manifest file"
        self.load_model()

        digest = xxhash.xxh3_64_intdigest(str(iteration_value))
        filename = f"denoiser.{iteration_index:05d}.{self.config.dry_wet}.{digest}.gz"
        output_manifest_filename = self.output_dir / filename

        with utils.open(
            output_manifest_filename, mode="wt", encoding="utf-8"
        ) as out_file:
            with utils.open(iteration_value) as in_file:
                for line, audio_information, audio in self.load_audios(in_file):
                    estimate, snr = self.denoise(audio)
                    # extract filename of audio segment and replace special characters for filesystem
                    audio_information = (
                        Path(audio_information).name.replace(":", "_").replace("|", "_")
                    )
                    output_wav_filename = (
                        self.output_dir / f"denoised_{audio_information}.wav"
                    )
                    write_audio(estimate, str(output_wav_filename), SAMPLE_RATE)
                    print(
                        f"{line}\t{output_wav_filename}\t{snr}",
                        file=out_file,
                    )
                    out_file.flush()

        return output_manifest_filename
