# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
This module provides a Wav2Vec2ForcedAligner:
a wrapper around Wav2Vec2ForCTC models capable of aligning transcription with speech.
Potentially, it can be enhanced with VAD predictions to detect pauses more correctly.
"""


import re
import typing as tp
from dataclasses import dataclass

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from stopes.eval.local_prosody.forced_aligner import BaseSpeechForceAligner
from stopes.eval.local_prosody.utterance import Utterance


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


class Wav2Vec2ForcedAligner(BaseSpeechForceAligner):
    """
    A forced aligner based on https://huggingface.co/docs/transformers/v4.31.0/en/model_doc/wav2vec2.
    The code is heavily copy-pasted from https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html.
    """

    def __init__(
        self,
        model_name="voidful/wav2vec2-xlsr-multilingual-56",
        device="cuda",
        blank_id=None,
        model=None,
        processor=None,
        sep_char="|",
    ):
        self.model_name = model_name
        self.device = device
        self.model = model or Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
        self.processor = processor or Wav2Vec2Processor.from_pretrained(model_name)
        self.blank_id = (
            self.processor.tokenizer.pad_token_id if blank_id is None else blank_id
        )

        self.id2label = self.processor.tokenizer.convert_ids_to_tokens(
            range(len(self.processor.tokenizer))
        )
        self.label2id = {c: i for i, c in enumerate(self.id2label)}

        self.frame_reduction = np.prod(
            [
                layer.conv.stride[0]
                for layer in self.model.wav2vec2.feature_extractor.conv_layers
            ]
        )
        self.sampling_rate = self.processor.feature_extractor.sampling_rate
        self.fps = self.sampling_rate / self.frame_reduction
        self.sep_char = sep_char

    def set_lang(self, lang):
        """Set lang for a model with an adapter, such as MMS-ASR."""
        self.processor.tokenizer.set_target_lang(lang)
        self.model.load_adapter(lang)
        self.id2label = self.processor.tokenizer.convert_ids_to_tokens(
            range(len(self.processor.tokenizer))
        )
        self.label2id = {c: i for i, c in enumerate(self.id2label)}

    def transcribe(
        self, waveforms: tp.List[np.ndarray]
    ) -> tp.Tuple[tp.List[torch.Tensor], tp.List[str]]:
        """
        Generate lists of emissions (log-probabilities) and of transcripts for a list of waveforms.
        The waveforms are expected to be sampled at self.sampling_rate.
        """
        features = self.processor(
            waveforms,
            padding=True,
            return_tensors="pt",
            sampling_rate=self.sampling_rate,
        )
        input_values = features.input_values.to(self.device)
        attention_mask = features.attention_mask.to(self.device)
        emissions = []
        with torch.no_grad():
            logits = self.model(input_values[0], attention_mask=attention_mask).logits
            decoded_results = []
            for logit in logits:
                pred_ids = torch.argmax(logit, dim=-1)
                mask = pred_ids.ge(1).unsqueeze(-1).expand(logit.size())
                vocab_size = logit.size()[-1]
                voice_prob = torch.nn.functional.softmax(
                    (torch.masked_select(logit, mask).view(-1, vocab_size)), dim=-1
                )
                comb_pred_ids = torch.argmax(voice_prob, dim=-1)
                decoded_results.append(self.processor.decode(comb_pred_ids))
                emissions.append(torch.log_softmax(logits[0], dim=-1).cpu())
        return emissions, decoded_results

    def modify_emission_with_vad(
        self, emission: torch.Tensor, vad_pred: np.ndarray, coef=1.0
    ) -> torch.Tensor:
        """Boost the silences predicted by the VAD model."""
        vad_aligned = smooth_resample(vad_pred, new_size=emission.shape[0])
        sep_id = self.processor.tokenizer.convert_tokens_to_ids(self.sep_char)
        silence_ids = [self.blank_id, sep_id]
        non_silence = [
            i
            for i in range(len(self.processor.tokenizer))
            if i != self.blank_id and i != sep_id
        ]
        new_emission = emission.clone()
        speech_proba = torch.tensor(vad_aligned, dtype=new_emission.dtype).unsqueeze(-1)
        new_emission[:, silence_ids] += torch.log(1 - speech_proba) * coef
        emission[:, non_silence] += torch.log(speech_proba) * coef
        new_emission = torch.log_softmax(new_emission, dim=-1)
        return new_emission

    def force_align_emission(
        self, transcript: str, emission: torch.Tensor
    ) -> Utterance:
        """Given a transcript and a tensor of per-frame probabilities, produce timestamps for each word."""
        if len(transcript) > emission.shape[0]:
            print(
                f"Cannot align {emission.shape[0]} frames to {len(transcript)} tokens."
            )
            result = Utterance(
                text=None,
                words=[""],
                ends=[-1],
                starts=[-1],
            )
            return result
        tokens = [self.label2id[c] for c in transcript]
        trellis = get_trellis(emission, tokens, blank_id=self.blank_id)
        path = backtrack(trellis, emission, tokens, blank_id=self.blank_id)
        if path is None:
            result = Utterance(
                text=None,
                words=[""],
                ends=[-1],
                starts=[-1],
            )
            return result
        segments = merge_repeats(path=path, transcript=transcript)
        result = segments_to_words(segments, fps=self.fps, sep=self.sep_char)
        result.text = transcript
        return result

    def preprocess_words(self, words: tp.List[str]) -> str:
        """Turn words into a text of characters that are recognizeable by the model (e.g. lowercase)."""
        # TODO: make it more model-dependent
        words = [re.sub("[^\\w]", "", w.lower()) for w in words]
        return self.sep_char.join(w for w in words if w)

    def process_utterance(
        self,
        waveform: torch.Tensor,
        text: tp.Optional[str] = None,
        words: tp.Optional[tp.List[str]] = None,
        **kwargs,
    ) -> Utterance:
        """
        Convert a waveform to a prosodically annotated Utterance,
        potentially using a fixed text or list of words as a forced transcription.
        """
        emissions, decoded_results = self.transcribe([waveform.numpy()])  # type: ignore
        if words is None:
            if text is None:
                text = decoded_results[0]
            words = text.split()
        transcript = self.preprocess_words(words)
        utterance = self.force_align_emission(
            emission=emissions[0], transcript=transcript
        )
        return utterance


def get_trellis(
    emission: torch.Tensor, tokens: tp.List[int], blank_id: int = 0
) -> torch.Tensor:
    """
    Compute scores for transcript labels occuring at each time frame.
    """
    # TODO: This method processes blank tokens naively; consider computing a grid interleaved by blanks instead
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, blank_id], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token (and emitting a blank OR emitting it once more)
            torch.maximum(
                trellis[t, 1:] + emission[t, blank_id],
                trellis[t, 1:] + emission[t, tokens],
            ),
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


def backtrack(
    trellis: torch.Tensor,
    emission: torch.Tensor,
    tokens: tp.List[int],
    blank_id: int = 0,
) -> tp.Optional[tp.List[Point]]:
    """Compute the most likely transcription path"""
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = int(torch.argmax(trellis[:, j]).item())

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        print(RuntimeWarning("Failed to align"))
        return None
    return path[::-1]


def merge_repeats(path: tp.List[Point], transcript: str) -> tp.List[Segment]:
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


def segments_to_words(
    segments: tp.List[Segment], fps: float = 1.0, sep="|"
) -> Utterance:
    result = Utterance(
        text=None,
        words=[""],
        ends=[-1],
        starts=[-1],
    )
    for s in segments:
        if s.label == sep:
            result.words.append("")
            result.starts.append(-1)
            result.ends.append(-1)
        else:
            if result.starts[-1] == -1:
                result.starts[-1] = s.start / fps
            result.ends[-1] = s.end / fps
            result.words[-1] = result.words[-1] + s.label
    return result


def smooth_resample(signal: np.ndarray, new_size: int, window_radius=2) -> np.ndarray:
    """
    Resample a signal by computing its weighted moving average.
    This has an advantage over Fourier-based resampling of not surpassing previous minimum and maximim values.
    """
    n = len(signal)
    id_diffs = (
        np.arange(n)[:, np.newaxis] - np.arange(new_size)[np.newaxis, :] * n / new_size
    )
    weights = np.maximum(0, window_radius - np.abs(id_diffs))
    weights /= weights.sum(0, keepdims=True)
    return np.dot(signal, weights)
