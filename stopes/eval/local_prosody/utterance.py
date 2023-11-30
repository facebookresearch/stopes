# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import typing as tp
from dataclasses import dataclass, field

PRECISION_DECIMALS = 4


@dataclass
class Utterance:
    """A representation of spoken utterance with start and end timestamps (in seconds) for each word.
    Potentially, other prosodic annotations could be added to this class in the future.
    """

    id: tp.Optional[str] = None
    # the text field might be different from " ".join(words)
    # because it may contain punctuation or other inaudible tokens,
    # whereas each word is supposed to be audible
    text: tp.Optional[str] = None
    words: tp.List[str] = field(default_factory=list)
    starts: tp.List[float] = field(default_factory=list)
    ends: tp.List[float] = field(default_factory=list)
    pauses: tp.List[float] = field(default_factory=list)
    emphasis_scores: tp.Optional[tp.List[float]] = None
    pause_speech_probs: tp.Optional[tp.List[float]] = None
    pause_vad_durations: tp.Optional[tp.List[float]] = None
    total_duration: tp.Optional[float] = None
    vad_duration: tp.Optional[float] = None
    sampling_rate: int = 16000
    lang: tp.Optional[str] = None

    def serialize(self) -> str:
        """Convert this instance to string representation."""
        return json.dumps(self.__dict__, ensure_ascii=False)

    @classmethod
    def deserialize(cls, data: str):
        """Create an instance from string representation."""
        return cls(**json.loads(data))

    @property
    def net_duration(self) -> float:
        """Total duration of the utterance in seconds, exlcluding the pauses"""
        return sum(self.ends) - sum(self.starts)

    @property
    def trimmed_duration(self) -> float:
        """Total duration of the utterance in seconds, including the pauses, but excluding silence before and after."""
        return self.ends[-1] - self.starts[0]

    def convert_sec_to_sample(self, seconds: float) -> int:
        return int(seconds * self.sampling_rate)

    def compute_speech_prob_between_words(
        self, sample_vad_speech_probs, vad_threshold=0.5
    ) -> None:
        """Given VAD speech probabilities (frame level), set `pause_speech_probs` and `pause_vad_durations` properties"""
        pause_speech_probs = []
        pause_vad_durations = []
        for word_i in range(len(self.starts) - 1):
            vad_segment = sample_vad_speech_probs[
                self.convert_sec_to_sample(
                    self.ends[word_i]
                ) : self.convert_sec_to_sample(self.starts[word_i + 1])
            ]
            pause_speech_probs.append(
                1 if vad_segment.numel() == 0 else vad_segment.mean().item()
            )
            pause_vad_durations.append(
                (vad_segment < vad_threshold).sum().item() / self.sampling_rate
            )
        pause_speech_probs.append(0)
        pause_vad_durations.append(0)

        self.pause_speech_probs = [
            round(x, PRECISION_DECIMALS) for x in pause_speech_probs
        ]
        self.pause_vad_durations = [
            round(x, PRECISION_DECIMALS) for x in pause_vad_durations
        ]

    def compute_aligner_based_pause_lengths(self) -> None:
        """Set `pauses` property based on word start and end moments"""
        pauses = []
        for word_i in range(len(self.starts) - 1):
            # first compute pause by aligner
            aligner_pause = self.starts[word_i + 1] - self.ends[word_i]
            pauses.append(aligner_pause)
        pauses.append(0)
        self.pauses = [round(x, PRECISION_DECIMALS) for x in pauses]

    def get_pauses_after_words(
        self,
        min_duration=0.1,
    ) -> tp.List[float]:
        """
        Compute pauses after each word.
        Pauses below min_duration (in seconds) are rounded to 0.
        Pause after the last word is always set to 0.
        """
        pauses = []
        for word_i in range(len(self.starts) - 1):
            # first compute pause by aligner
            pause = self.starts[word_i + 1] - self.ends[word_i]
            if self.pause_vad_durations is not None:
                pause = min(pause, self.pause_vad_durations[word_i])
            pauses.append(pause)
        pauses.append(0)

        if min_duration:
            pauses = [p if p >= min_duration else 0 for p in pauses]
        return pauses

    def get_text_with_markup(self, min_pause_duration=0.1, min_emph_score=0.5) -> str:
        """
        Return words and significant pauses, in the format like 'the cat [pause x 0.2] sat on the mat'.
        If a word is emphasized, it will be formatted like "*sat*".
        """
        parts = []
        pause_durations = self.get_pauses_after_words(min_duration=min_pause_duration)
        emphasis_scores = self.emphasis_scores or [0] * len(self.words)
        for (word, pause, emph_score) in zip(
            self.words, pause_durations, emphasis_scores
        ):
            if emph_score > min_emph_score:
                word = f"*{word}*"
            parts.append(word)
            if pause > min_pause_duration:
                parts.append(f"[pause x {pause:2.2f}]")
        # with adjascent emphasized words, we don't want repeated asterisks
        result = " ".join(parts).replace("* *", " ")
        return result
