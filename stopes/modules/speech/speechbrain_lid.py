# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import dataclasses
import logging
import typing as tp
from collections import defaultdict
from pathlib import Path

import omegaconf
import torch
import torchaudio

from stopes.core import utils
from stopes.modules.preprocess.line_processor import LineProcessorCallback
from stopes.modules.speech.utils import Audio, parse_audio_deprecated

log = logging.getLogger("speechbrain_lid")


class LidResult(tp.NamedTuple):
    score: float
    audio: str
    lang: str
    extra_info: str


@dataclasses.dataclass
class SpeechbrainLIDConfig:
    model: Path = omegaconf.MISSING
    # if the segment is longer than `max_seconds`, it will be clipped
    max_seconds: float = 600.0

    # if the segment is shorter than `min_seconds`, it will be extended
    # `extend_seconds`/2 on the left and `extend_seconds`/2 on the right
    min_seconds: float = 1.1
    extend_seconds: float = 1.0
    max_tokens: int = 10_000_000
    # Will create one output file per input
    split_to_mono: bool = True


class SpeechbrainLidCallback(LineProcessorCallback):
    """
    Run LID using a speechbrain model (https://github.com/speechbrain/speechbrain)

    python -m stopes.modules \
        +speech_preproc=speechbrain_lid \
        speech_preproc.output_dir=test-lid-thai \
        speech_preproc.shards="thai_example.mp3" \
        speech_preproc.line_processor.config.model=$MODEL \
        launcher.cluster=debug
    """

    def __init__(self, config: SpeechbrainLIDConfig = SpeechbrainLIDConfig(), **kwargs):
        super().__init__(**kwargs)
        if isinstance(config, omegaconf.DictConfig):
            config = SpeechbrainLIDConfig(**config)  # type: ignore
        self.config = config
        self.model_path = Path(config.model)
        assert (
            self.model_path.exists()
        ), "Please provide a valid local path to downloaded model"
        self.gpu = torch.cuda.is_available()

        # Delay speechbrain import to the config parsing.
        # That make sure that only people using speechbrain need to have it installed.
        from speechbrain.pretrained import EncoderClassifier  # type: ignore

        self.input_file_hash = utils.sha_key(kwargs["input_file"])
        self.tmp_folder = self.output_dir / ("tmp_speechbrain_" + self.input_file_hash)
        try:
            self.model = EncoderClassifier.from_hparams(
                source=str(self.model_path),
                savedir=self.tmp_folder,
                run_opts={"device": "cuda" if self.gpu else "cpu"},
            )
            ind2lab = self.model.hparams.label_encoder.ind2lab
            for ind in ind2lab:
                lang = ind2lab[ind]
                ind2lab[ind] = self.parse_lang(lang)
        except Exception:
            log.exception(f"Not able to load speechbrain model from {self.model_path}")
            raise

        self.sample_rate = self.model.audio_normalizer.sample_rate
        self.max_frames = int(self.sample_rate * self.config.max_seconds)
        self.min_frames = int(self.sample_rate * self.config.min_seconds)
        self.extend_frames_each = int(
            self.sample_rate * self.config.extend_seconds * 0.5
        )
        self.audio_cache: tp.Dict[str, torch.Tensor] = {}
        self._unk_count = 0
        self._ok_count = 0
        self._out: tp.IO[tp.Any] = None  # type: ignore
        self._out_per_lang: tp.Dict[str, tp.TextIO] = {}

    def name_output_file(self, lang: str) -> Path:
        return (
            Path(self.output_dir)
            / f"{self.outfile_prefix}.{self.input_file_idx:05d}.{self.input_file_hash}.{lang}.tsv.gz"
        )

    def __enter__(self) -> "SpeechbrainLidCallback":
        if not self.config.split_to_mono:
            self._out = utils.open(self.final_result(), "w")
        return self

    def __exit__(self, *args: tp.Any) -> None:
        unk, ok = self._unk_count, self._ok_count
        log.info(f"Analyzed {ok} audio samples.")
        if unk:
            log.error(f"Wasn't able to analyze {unk} audio samples.")
        if self._out:
            self._out.close()

        for lang, out_file in self._out_per_lang.items():
            out_file.close()

        # remove symlinks created by speechbrain
        if self.tmp_folder.exists():
            for file in self.tmp_folder.iterdir():
                if file.is_symlink():
                    file.unlink()

    def get_outfile_for_lang(self, lang: str) -> tp.TextIO:
        if not lang in self._out_per_lang:
            self._out_per_lang[lang] = utils.open(self.name_output_file(lang), "wt")  # type: ignore
        return self._out_per_lang[lang]

    def get_audio(self, line: str) -> torch.Tensor:
        infile, ts_start, ts_end, _ = parse_audio_deprecated(line)
        if infile in self.audio_cache:
            signal = self.audio_cache[infile]
        else:
            signal, sr = torchaudio.load(infile, channels_first=False)
            if sr != self.sample_rate:
                log.info(
                    f"model expects sample rate of {self.sample_rate}. "
                    f"however, {infile} uses: {sr}. Converting audio.."
                )
            # convert sample rate and channel selection (if needed)
            signal = self.model.audio_normalizer(signal, sr)
            self.audio_cache[infile] = signal
        if ts_start is not None and ts_end is not None:
            if (
                ts_end != -1
                and self.min_frames is not None
                and ts_end - ts_start < self.min_frames
            ):
                ts_start = max(0, ts_start - self.extend_frames_each)
                ts_end = min(len(signal) - 1, ts_end + self.extend_frames_each)
            signal = signal[ts_start:ts_end]
        if len(signal) > self.max_frames:
            log.info(
                f"{infile} ({ts_start}-{ts_end}) longer than {self.config.max_seconds} seconds. clipping.."
            )
            signal = signal[: self.max_frames]
        return signal

    def append_unknown(
        self, results: tp.List[LidResult], e: Exception, line: str
    ) -> None:
        if self._unk_count % 100 == 0:
            log.exception(
                f"could not get prediction for input: {line!r}. "
                f"Defaulting to UNK ({self._unk_count})"
            )
        self._unk_count += 1
        results.append(LidResult(score=0, audio=line, lang="UNK", extra_info=str(e)))

    def get_predictions(
        self, padded_signals: torch.Tensor, lines: tp.List[str]
    ) -> tp.List[LidResult]:
        results: tp.List[LidResult] = []
        try:
            all_out_probs, *_ = self.model.classify_batch(padded_signals)
        except torch.cuda.OutOfMemoryError as oom:  # type: ignore
            log.exception(
                "Cuda OutOfMemoryError: probably batch too big. "
                f"Try to decrease max_tokens (currently {self.config.max_tokens})"
            )
            raise oom
        except Exception as e:
            for line in lines:
                self.append_unknown(results, e, line)
            return results

        for out_probs, line in zip(all_out_probs, lines):
            try:
                log_scores, _labels = torch.topk(out_probs, k=3)
                scores = torch.exp(log_scores)
                labels = self.model.hparams.label_encoder.decode_torch(_labels)
                self._ok_count += 1
                results.append(
                    LidResult(
                        score=scores[0].item(),
                        audio=line,
                        lang=labels[0],
                        extra_info=f"{labels[1]}: {scores[1]:.3f}, {labels[2]}: {scores[2]:.3f}",
                    )
                )
            except Exception as e:
                self.append_unknown(results, e, line)
        return results

    def final_result(self) -> Path:
        if self.config.split_to_mono:
            return Path(self.output_dir)
        else:
            return Path(self.output_dir) / f"{Path(self.input_file).stem}.tsv.gz"

    def batch_signals(
        self,
        line_signals: tp.Iterable[tp.Tuple[torch.Tensor, str]],
        max_tokens: int,
    ) -> tp.Iterator[tp.List[tp.Tuple[torch.Tensor, str]]]:
        """
        We assume that `signals` is a list of tensors ordered by decreasing length.
        This function will create batches so that padding them will result to
        batches size less than `max_tokens`.
        """
        current_batch: tp.List[tp.Tuple[torch.Tensor, str]] = []
        current_batch_token_sum = 0
        longest_size_in_batch = 0
        for signal, line in line_signals:
            if (
                current_batch_token_sum + longest_size_in_batch > max_tokens
                and current_batch
            ):
                yield current_batch
                current_batch = []
                longest_size_in_batch = 0
                current_batch_token_sum = 0
            if longest_size_in_batch == 0:
                longest_size_in_batch = len(signal)

            current_batch_token_sum += longest_size_in_batch
            current_batch.append((signal, line))

        if current_batch:
            yield current_batch

    def process_lines(
        self,
        lines_with_number: tp.Iterator[tp.Tuple[int, str]],
    ) -> None:
        # Note: this is making a full in memory copy of the input.
        # this may be problematic if fed big input files
        samples_per_lang: tp.Dict[str, tp.List[LidResult]] = defaultdict(list)
        split_to_mono = self.config.split_to_mono

        line_signals = []
        for line_no, line in lines_with_number:
            line = line.rstrip("\n")
            line_signals.append((self.get_audio(line), line))

        # for optimal batching with padded tensors, we order by length (decreasing)
        line_signals.sort(key=lambda x: len(x[0]), reverse=True)

        for signal_line_batch in self.batch_signals(
            line_signals, max_tokens=self.config.max_tokens
        ):
            signal_batch = [s for s, l in signal_line_batch]
            lines_batch = [l for s, l in signal_line_batch]
            padded_signals = torch.nn.utils.rnn.pad_sequence(
                signal_batch, batch_first=True
            )
            predictions = self.get_predictions(padded_signals, lines_batch)

            for prediction in predictions:
                if split_to_mono:
                    samples_per_lang[prediction.lang].append(prediction)
                else:
                    self.write_result(prediction, self._out)

        if split_to_mono:
            self.split_to_mono(samples_per_lang)

    def parse_lang(self, lang: str) -> str:
        # parse model-specific outputs
        if self.model_path.name == "lang-id-voxlingua107-ecapa":
            return lang.split(":")[0]
        return lang

    def split_to_mono(
        self, predictions: tp.Mapping[str, tp.Iterable[LidResult]]
    ) -> None:
        for lang, results in predictions.items():
            lang = self.parse_lang(lang)
            # NOTE: lang-specific files are outputted for each input shard idx
            out = self.get_outfile_for_lang(lang)
            for result in results:
                self.write_result(result, out)
            out.flush()

    def write_result(self, result: LidResult, out: tp.IO[tp.Any]) -> None:
        # We use the same format as bitext mining results:
        # a tsv with 3 columns (score, source, target).
        # The predicted lang can still be easily parsed, and output can be opened by seamlisten.
        print(
            result.score,
            result.audio,
            f"{result.lang} ({result.extra_info})",
            sep="\t",
            file=out,
        )
