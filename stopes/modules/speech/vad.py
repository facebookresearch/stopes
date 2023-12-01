# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
This module offers a wrapper around Silero VAD models.
There are two ways in which you could use it directly:

```Python
from stopes.modules.speech.vad import VAD
root = "$DATA_ROOT/seamless/nllb/modules_test_data"
vad = VAD(
    model_path=f"{root}/silero_vad.jit",
    hard_limit_min_length=2.0,
)
audio_path = f"{root}/thai_example.mp3"

# Usage 1: predicting contiguous segments (with potential overlap)
batch_sampler, dataset = vad.build_segments_dataset(audio_path)
print([dataset.timestamps[i] for ind in batch_sampler for i in ind])
# [(2592, 45024), (2592, 84960), ..., (147456, 201696)], 10 segments in total

# Usage 2: just predicting speech probabilities at each moment of time
proba, n_frames = vad.get_speech_probs(vad.read_audio(audio_path), vad.model, window_size_samples=1536)
print(proba)
# [0.0024, 0.0003, 0.9172, ..., 0.0009]
# 136 predictions in total, where 136 = ceil(n_frames / 1536)
```

Alternatively, you can use VADSegmentAudioModule based on this module.
"""

import typing as tp
import warnings

import numpy as np
import torch

from stopes.modules.speech.utils import parse_audio
from stopes.utils.web import cached_file_download

SAMPLING_RATE = 16000
# To freeze the version of the VAD model, we re-host it on our server.
# Originally, it was downloaded from https://pytorch.org/hub/snakers4_silero-vad_vad
MODEL_URL = (
    "https://dl.fbaipublicfiles.com/speech_expressivity_evaluation/silero_vad.jit"
)


def init_jit_model(model_path: str, device=torch.device("cpu")):
    """Load a torch.jit model for inference"""
    torch.set_grad_enabled(False)
    if model_path == "auto":
        model_path = str(cached_file_download(MODEL_URL, "silero_vad.jit"))
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


class VAD:
    def __init__(
        self, model_path, min_length=1, max_length=20, hard_limit_min_length=1
    ):
        self.model = init_jit_model(model_path=model_path).cuda()
        self.min_length = min_length * SAMPLING_RATE
        self.max_length = max_length * SAMPLING_RATE
        self.hard_limit_min_length = hard_limit_min_length * SAMPLING_RATE

    def read_audio(self, audio: str) -> torch.Tensor:
        wav = parse_audio(audio, sampling_factor=16).load(average_channels=True)
        if wav.ndim > 1:
            wav = wav.squeeze(0)  # remove channel dimension)
        return wav

    @classmethod
    def get_speech_timestamps(
        cls,
        audio: torch.Tensor,
        model,
        threshold: float = 0.5,
        sampling_rate: int = SAMPLING_RATE,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        window_size_samples: int = 1536,
        speech_pad_ms: int = 30,
        return_seconds: bool = False,
    ) -> tp.List[tp.Dict]:
        """
        This method is used for splitting long audios into speech chunks using silero VAD
        Parameters
        ----------
        audio: torch.Tensor, one dimensional
            One dimensional float torch.Tensor, other types are casted to torch if possible
        model: preloaded .jit silero VAD model
        threshold: float (default - 0.5)
            Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
            It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.
        sampling_rate: int (default - 16000)
            Currently silero VAD models support 8000 and 16000 sample rates
        min_speech_duration_ms: int (default - 250 milliseconds)
            Final speech chunks shorter min_speech_duration_ms are thrown out
        min_silence_duration_ms: int (default - 100 milliseconds)
            In the end of each speech chunk wait for min_silence_duration_ms before separating it
        window_size_samples: int (default - 1536 samples)
            Audio chunks of window_size_samples size are fed to the silero VAD model.
            WARNING! Silero VAD models were trained using 512, 1024, 1536 samples for 16000 sample rate and 256, 512, 768 samples for 8000 sample rate.
            Values other than these may affect model perfomance!!
        speech_pad_ms: int (default - 30 milliseconds)
            Final speech chunks are padded by speech_pad_ms each side
        return_seconds: bool (default - False)
            whether return timestamps in seconds (default - samples)
        Returns
        ----------
        speeches: list of dicts
            list containing ends and beginnings of speech chunks (samples or seconds based on return_seconds)
        """

        speech_probs, audio_length_samples = cls.get_speech_probs(
            audio=audio,
            model=model,
            sampling_rate=sampling_rate,
            window_size_samples=window_size_samples,
        )

        return cls.segment_speech(
            speech_probs=speech_probs,
            audio_length_samples=audio_length_samples,
            threshold=threshold,
            sampling_rate=sampling_rate,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            window_size_samples=window_size_samples,
            speech_pad_ms=speech_pad_ms,
            return_seconds=return_seconds,
        )

    @staticmethod
    def get_speech_probs(
        audio: torch.Tensor,
        model,
        sampling_rate: int = SAMPLING_RATE,
        window_size_samples: int = 1536,
    ) -> tp.Tuple[np.ndarray, int]:
        """Get a list of speech probabilities computed with sliding window over the audio using the model."""
        if not torch.is_tensor(audio):
            try:
                audio = torch.Tensor(audio)
            except:
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")

        if len(audio.shape) > 1:
            for _ in range(audio.ndim):  # trying to squeeze empty dimensions
                audio = audio.squeeze(0)
            assert (
                audio.ndim == 1
            ), "More than one dimension in audio. Are you trying to process audio with 2 channels?"

        if sampling_rate > 16000 and (sampling_rate % 16000 == 0):
            step = sampling_rate // 16000
            sampling_rate = 16000
            audio = audio[::step]
            warnings.warn(
                "Sampling rate is a multiply of 16000, casting to 16000 manually!"
            )
        else:
            step = 1

        if sampling_rate == 8000 and window_size_samples > 768:
            warnings.warn(
                "window_size_samples is too big for 8000 sampling_rate! Better set window_size_samples to 256, 512 or 768 for 8000 sample rate!"
            )
        if window_size_samples not in [256, 512, 768, 1024, 1536]:
            warnings.warn(
                "Unusual window_size_samples! Supported window_size_samples:\n - [512, 1024, 1536] for 16000 sampling_rate\n - [256, 512, 768] for 8000 sampling_rate"
            )

        model.reset_states()

        audio_length_samples = len(audio)

        speech_probs = []
        for current_start_sample in range(0, audio_length_samples, window_size_samples):
            chunk = audio[
                current_start_sample : current_start_sample + window_size_samples
            ]
            if len(chunk) < window_size_samples:
                chunk = torch.nn.functional.pad(
                    chunk, (0, int(window_size_samples - len(chunk)))
                )
            if next(model.parameters()).is_cuda:
                chunk = chunk.cuda()
            speech_prob = model(chunk, sampling_rate).item()
            speech_probs.append(speech_prob)

        return np.array(speech_probs), audio_length_samples

    @staticmethod
    def segment_speech(
        speech_probs: tp.Union[tp.List[float], np.ndarray],
        audio_length_samples: int,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        window_size_samples: int = 1536,
        speech_pad_ms: int = 30,
        return_seconds: bool = False,
    ) -> tp.List[tp.Dict]:
        """Given a list of speech probability for each window, predict contiguous segments of speech."""
        if sampling_rate > 16000 and (sampling_rate % 16000 == 0):
            step = sampling_rate // 16000
            sampling_rate = 16000
        else:
            step = 1

        min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
        min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        speech_pad_samples = sampling_rate * speech_pad_ms / 1000

        triggered = False
        speeches = []
        current_speech = {}
        neg_threshold = threshold - 0.15
        temp_end = 0

        for i, speech_prob in enumerate(speech_probs):
            if (speech_prob >= threshold) and temp_end:
                temp_end = 0

            if (speech_prob >= threshold) and not triggered:
                triggered = True
                current_speech["start"] = window_size_samples * i
                continue

            if (speech_prob < neg_threshold) and triggered:
                if not temp_end:
                    temp_end = window_size_samples * i
                if (window_size_samples * i) - temp_end < min_silence_samples:
                    continue
                else:
                    current_speech["end"] = temp_end
                    if (
                        current_speech["end"] - current_speech["start"]
                    ) > min_speech_samples:
                        speeches.append(current_speech)
                    temp_end = 0
                    current_speech = {}
                    triggered = False
                    continue

        if (
            current_speech
            and (audio_length_samples - current_speech["start"]) > min_speech_samples
        ):
            current_speech["end"] = audio_length_samples
            speeches.append(current_speech)

        for i, speech in enumerate(speeches):
            if i == 0:
                speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
            if i != len(speeches) - 1:
                silence_duration = speeches[i + 1]["start"] - speech["end"]
                if silence_duration < 2 * speech_pad_samples:
                    speech["end"] += int(silence_duration // 2)
                    speeches[i + 1]["start"] = int(
                        max(0, speeches[i + 1]["start"] - silence_duration // 2)
                    )
                else:
                    speech["end"] += int(speech_pad_samples)
            else:
                speech["end"] = int(
                    min(audio_length_samples, speech["end"] + speech_pad_samples)
                )

        if return_seconds:
            for speech_dict in speeches:
                speech_dict["start"] = round(speech_dict["start"] / sampling_rate, 1)  # type: ignore
                speech_dict["end"] = round(speech_dict["end"] / sampling_rate, 1)  # type: ignore
        elif step > 1:
            for speech_dict in speeches:
                speech_dict["start"] *= step
                speech_dict["end"] *= step

        return speeches

    def get_timestamps(self, wav, **kwargs):
        with torch.no_grad():
            timestamps = self.get_speech_timestamps(
                wav, self.model, sampling_rate=SAMPLING_RATE, **kwargs
            )
        res = np.array(
            [
                [timestamps[i]["start"], timestamps[i]["end"]]
                for i in range(len(timestamps))
            ]
        )
        return res

    def get_segmentations(self, wav):
        timestamps = self.get_timestamps(wav)
        segments = []
        indices = []
        sizes = []
        for i in range(timestamps.shape[0]):
            start = timestamps[i, 0]
            for j in range(i, timestamps.shape[0]):
                end = timestamps[j, 1]
                length = end - start
                if length > self.min_length and length < self.max_length:
                    segments.append(wav[start:end])
                    indices.append((start, end))
                    sizes.append(end - start)
                if j + 1 < timestamps.shape[0]:
                    next_start = timestamps[j + 1, 0]
                    if next_start - end > self.hard_limit_min_length:
                        break
        return segments, indices, sizes

    def get_net_speech_duration(
        self,
        audio: torch.Tensor,  # shape [channel, time]
        threshold: tp.Optional[float] = None,
        window: int = 512,
    ) -> tp.Tuple[float, np.ndarray, int]:
        """Compute net duration as sum of durations of the voiced frames in an audio."""
        speech_probs, audio_length_samples = self.get_speech_probs(
            audio=audio, model=self.model, window_size_samples=window
        )
        if threshold is not None:
            avg = (speech_probs >= threshold).mean()
        else:
            avg = speech_probs.mean()
        return avg * audio.size(-1) / SAMPLING_RATE, speech_probs, audio_length_samples
