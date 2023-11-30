# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import gc
import logging
import os
import tempfile
import typing as tp
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import whisper
from omegaconf import MISSING
from sentence_transformers import SentenceTransformer
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
from tqdm.auto import tqdm

from stopes.core.stopes_module import Requirements, StopesModule
from stopes.modules.speech.utils import read_audio
from stopes.modules.speech.video_alignement.video_utils import (
    VideoShard,
    add_metadata_to_table,
    folder_size,
    generate_ngrams,
    numpy_to_fixed_size_pyarrow_array,
    wav_to_arrow_array,
)

logger = logging.getLogger(__name__)


lang_map_3to2 = {
    "eng": "en",
    "deu": "de",
    "spa": "es",
    "rus": "ru",
    "kor": "ko",
    "fra": "fr",
    "por": "pt",
    "tur": "tr",
    "pol": "pl",
    "cat": "ca",
    "nld": "nl",
    "swe": "sv",
    "ita": "it",
    "ind": "id",
    "hin": "hi",
    "fin": "fi",
    "vie": "vi",
    "ukr": "uk",
    "ces": "cs",
    "ron": "ro",
    "dan": "da",
    "tam": "ta",
    "tha": "th",
    "urd": "ur",
    "cym": "cy",
    "slk": "sk",
    "tel": "te",
    "ben": "bn",
    "kan": "kn",
    "est": "et",
    "swh": "sw",
    "arb": "af",
    "mlt": "mt",
    "tgl": "tl",
    "cmn": "zh",
}


@dataclass
class WhisperSegmentorConfig:
    shards: tp.Any = MISSING
    """
    We expect here a `csv` file with the following schema :
      'sibling_id' (string) - a video_id that is the same for all language versions of the same video
      'lang' (string) - video language in 3 chars code (as for SONAR)
      'audio_path' (string) - path to readable video audio (various formats accepted here)
    """
    langs: tp.Optional[tp.List[str]] = None
    """
    Filtering option. If not None, it represents a list of languages to restrict on.
    """
    sibling_ids: tp.Optional[
        tp.List[str]
    ] = None  # specific videos ids that we want to segment
    """
    Filtering option. If not None, it represents a list of ids to restrict on.
    """
    output_dir: Path = MISSING  # path to output parquet file
    parquet_file_name: str = "segments"
    whisper_model: str = "large-v2"
    text_model: tp.Optional[str] = "LaBSE"
    pause_ngram_depth: int = 12
    min_segment_length: float = (
        0.2  # in seconds; to drop the small segments whose len < min_segment_length
    )
    max_segment_length: float = 20
    max_pause_duration: float = 1.5
    segment_padding: float = (
        0.1  # number of seconds added before and after each segment
    )
    batch_size_: int = 5  # internal batch size used for sonar model inference


class WhisperSegmentorModule(StopesModule):

    """Extract utterances from an audio and embed them
    This module is multi-lingual : different languages can be processed with the same pipeline

    Example:
    >>> test_config = WhisperSegmenterConfig(shards=".../all_langs_videos.csv",
                                             output_dir="...",
                                             parquet_file_name = "segments",
                                             langs=["fra", "eng"],
                                             sibling_ids=None,
                                             whisper_model="base",
                                             text_model="LaBSE",
                                             pause_ngram_depth=10,
                                             min_segment_length=0.2,
                                             max_segment_length=15,
                                             max_pause_duration=1.5,
                                             segment_padding=0.1)
    >>> wsm = WhisperSegmentorModule(test_config)
    >>> for ch in wsm.array(): wsm.run(ch)

    Outputs:
        a path to the following segments partitioned dataset (parquet):
            partitioning keys:
                * sibling_id
                * lang
                * ts (timestamp of the pipeline start, used for the deduplication in case of multiple launches)
            other columns:
                * source: [categorical] - either "whisper" (original segment, used for debugging)
                                        - either "ngrams" (our pause based upsample method)
                                        - either "sentence" (segments looking more like a sentence)
                * start: [double] - segment's start in the original audio  (in seconds)
                * end: [double] - segment's end in the original audio  (in seconds)
                * duration: [double] - segment's duration in seconds
                * probability: [float] - segment speech probability (1 - no_speech_prob) given by whisper
                * text: [string] - segment transcription
                * sampled_wav [list[float32*]] - segment audio waveform array (sampled at SR=16k)
                * test_embeddings [list[float32[1024]]] - segment transcribed text embedding (not normalized)
                * speech_embeddings [list[float32[1024]]] - segment audio SONAR embedding (not normalized)
    """

    SR = 16000

    def __init__(self, config: WhisperSegmentorConfig = WhisperSegmentorConfig()):
        super().__init__(config, WhisperSegmentorConfig)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        self.shards: tp.List[VideoShard] = self.parse_shards()
        self.launch_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def requirements(self) -> Requirements:
        return Requirements(
            nodes=1,
            mem_gb=42,
            tasks_per_node=1,
            gpus_per_node=1,
            cpus_per_task=2,
            timeout_min=680,
            constraint="volta32gb",
        )

    def array(self):
        return self.shards

    def run(
        self, iteration_value: tp.Optional[VideoShard] = None, iteration_index: int = 0
    ) -> pa.Table:
        shard: VideoShard = iteration_value or self.shards[0]
        wav, transcript_df = self.get_transcription(shard.audio_path, shard.lang)

        logger.info(f"Total whisper segment generated {len(transcript_df)}")
        words_df = self.get_word_dataframe(transcript_df)

        sentence_df = self.get_sentence_segments_dataframe(
            words_df, max_duration_expected=self.config.max_segment_length * 0.6
        )
        pauses_df = self.get_pause_segments_dataframe(words_df)
        ngrams_df = self.build_ngrams(pauses_df)

        merged_df = self.concat_segments_and_ngrams_and_filter(
            transcript_df, sentence_df, ngrams_df
        )
        logger.info(f"Total sub-segements generated {len(merged_df)}")

        # sort by duration to make batch padding more efficient
        merged_df = merged_df.sort_values("duration")

        # Adding partition data
        merged_df["sibling_id"] = pd.Series(
            [shard.sibling_id] * len(merged_df), dtype="category"
        )
        merged_df["lang"] = pd.Series([shard.lang] * len(merged_df), dtype="category")
        merged_df["ts"] = pd.Series([self.launch_ts] * len(merged_df), dtype="category")
        merged_df["audio_path"] = pd.Series(
            [shard.audio_path] * len(merged_df), dtype="category"
        )
        # Transform to pyarrow Table
        pa_table = pa.Table.from_pandas(merged_df)
        metadata = {"config": str(self.config), "launch_time": self.launch_ts}
        pa_table = add_metadata_to_table(pa_table, metadata)

        wav_segments = self._cut_as_segments_and_add_padding(
            wav,
            torch.as_tensor(merged_df["start"].values),
            torch.as_tensor(merged_df["end"].values),
        )

        pa_table = pa_table.append_column(
            "sampled_wav", wav_to_arrow_array(wav_segments)
        )

        speech_embeds_sonar = self.compute_sonar_speech_segment_embedding(
            wav_segments, shard.lang
        )
        pa_table = pa_table.append_column(
            "speech_embeddings", numpy_to_fixed_size_pyarrow_array(speech_embeds_sonar)
        )

        if self.config.text_model:
            text_embeds = self.compute_text_segment_embedding(merged_df["text"].values)
            pa_table = pa_table.append_column(
                "text_embeddings", numpy_to_fixed_size_pyarrow_array(text_embeds)
            )

        # sort back
        pa_table = pa_table.sort_by([("start", "ascending"), ("end", "ascending")])
        # dump to partition parquet dataset
        logger.info(
            f"Dumping parquet DataSet if size (in memory) {pa_table.nbytes / (1024 * 1024)} M"
        )
        output_ds_path = self.config.output_dir / self.config.parquet_file_name
        pq.write_to_dataset(
            pa_table,
            output_ds_path,
            partition_cols=["sibling_id", "lang", "ts"],
        )

        result_path = (
            output_ds_path
            / f"sibling_id={shard.sibling_id}"
            / f"lang={shard.lang}"
            / f"ts={self.launch_ts}"
        )
        logger.info(
            f"Producing a parquet partition of size {folder_size(result_path) / (1024 * 1024)} M"
        )
        return result_path

    @staticmethod
    def get_sentence_segments_dataframe(
        words_df: pd.DataFrame, max_duration_expected: float
    ) -> pd.DataFrame:
        words_df = words_df.copy()

        words_df["new_segment_split"] = (
            (words_df["segment_split"] > 0)
            & (words_df["capital_case"] | words_df["punctuation_nonstops"])
        ) | words_df["punctuation_stops"]

        words_df[
            "new_segment_index"
        ] = WhisperSegmentorModule._segments_with_duration_limit(
            words_df, max_duration_expected
        )

        grouper_ = words_df.groupby("new_segment_index")
        segment_df = grouper_.agg(
            {"text": "sum", "start": "min", "end": "max", "probability": "mean"}
        )
        segment_df.reset_index(inplace=True, drop=True)
        segment_df["source"] = pd.Series(
            ["sentence"] * len(segment_df), dtype="category"
        )
        segment_df = segment_df.loc[
            segment_df["end"] - segment_df["start"] <= 2 * max_duration_expected
        ]
        return segment_df.reset_index()

    def parse_shards(self) -> tp.List[VideoShard]:
        input_df = pd.read_csv(self.config.shards)
        if self.config.langs is not None:
            df_lang = input_df.loc[input_df["lang"].isin(self.config.langs)]
        else:
            df_lang = input_df

        if self.config.sibling_ids is not None:
            df_lang = df_lang.loc[df_lang["sibling_id"].isin(self.config.sibling_ids)]
            logger.info(
                f"Found {len(df_lang)} specific sibling_ids of langs in {self.config.langs}"
            )

        shards = [
            VideoShard(
                sibling_id=row["sibling_id"],
                audio_path=row["audio_path"],
                lang=row["lang"],
            )
            for _, row in df_lang.iterrows()
        ]
        logger.info(
            f"Loading {len(shards)} chards of lang={self.config.langs} out of {len(input_df)}"
        )
        return shards

    @staticmethod
    @lru_cache(maxsize=3)
    def _load_sonar_speech2vec(lang: str) -> torch.nn.Module:
        encoder = f"sonar_speech_encoder_{lang}"
        sonar_s2vec_model = SpeechToEmbeddingModelPipeline(encoder=encoder)
        return sonar_s2vec_model

    @staticmethod
    @lru_cache(maxsize=3)
    def _load_whisper(path: str) -> whisper.model.Whisper:
        return whisper.load_model(path, device="cpu").cpu()

    @staticmethod
    @lru_cache(maxsize=3)
    def _load_text2vec_model(path: str) -> SentenceTransformer:
        return SentenceTransformer(path, device="cpu").cpu()

    @staticmethod
    def free_memory() -> None:
        gc.collect()
        torch.cuda.empty_cache()

    def compute_sonar_speech_segment_embedding(
        self, nested_tensor: tp.List[torch.Tensor], lang: str
    ) -> np.ndarray:
        sonar_model: torch.nn.Module = self._load_sonar_speech2vec(lang).cuda()
        sonar_model.device = torch.device("cuda")  # type: ignore
        vec = sonar_model.predict(  # type: ignore
            nested_tensor,
            batch_size=self.config.batch_size_,
            n_prefetched_batches=2,
            n_parallel=2,
        )
        sonar_model.cpu()
        sonar_model.device = torch.device("cpu")  # type: ignore
        del sonar_model
        self.free_memory()
        return vec.cpu().numpy()

    @torch.inference_mode()
    def compute_text_segment_embedding(
        self, nested_tensor: tp.List[tp.List[str]]
    ) -> np.ndarray:
        text_model = self._load_text2vec_model(self.config.text_model).cuda()
        text_embeds = text_model.encode(nested_tensor).astype("float32")
        text_model.cpu()
        del text_model
        self.free_memory()
        return text_embeds

    def _cut_as_segments_and_add_padding(
        self, wav: torch.Tensor, start: torch.Tensor, end: torch.Tensor
    ) -> tp.List[torch.Tensor]:
        start = ((start - self.config.segment_padding).clip_(0) * self.SR).long()
        end = ((end + self.config.segment_padding) * self.SR).long()
        return [wav[s:e] for s, e in zip(start, end)]

    @torch.inference_mode()
    def compute_whisper_segmentation(
        self, wav: torch.Tensor, lang: str
    ) -> pd.DataFrame:
        logger.info(
            f"Starting Whisper segmentation on wav of length {round(len(wav) / self.SR / 60, 3)} minuntes in lang = {lang}"
        )
        with tempfile.TemporaryDirectory() as data_gym_cache:  # attempt to fixe loading issue
            os.environ["DATA_GYM_CACHE_DIR"] = str(data_gym_cache)
            whisper_model = self._load_whisper(self.config.whisper_model).cuda()
            wav = wav.cpu()
            full_transcript = whisper_model.transcribe(
                wav,
                verbose=False,
                language=lang_map_3to2.get(lang, lang),
                word_timestamps=True,
            )
            # wav = wav.cpu()
            logger.info(f"Found {len(full_transcript['segments'])} raws segments")
            whisper_model = whisper_model.cpu()
            del whisper_model, wav
            self.free_memory()
        # Output Schema :
        # start | end | text | tokens | avg_logprob | compression_ratio | no_speech_prob | words

        transcript_df = pd.DataFrame(full_transcript["segments"])
        transcript_df = transcript_df.reset_index()
        transcript_df.rename(columns={"index": "segment_index"}, inplace=True)
        transcript_df.drop(columns=["id", "seek", "temperature"], inplace=True)
        return transcript_df

    def get_transcription(self, audio_path: str, lang: str):
        wav = read_audio(audio_path, self.SR)
        transcript_df = self.compute_whisper_segmentation(wav, lang)
        transcript_df["source"] = pd.Series(
            ["whisper"] * len(transcript_df), dtype="category"
        )
        transcript_df["probability"] = 1.0 - transcript_df["no_speech_prob"]
        return wav, transcript_df

    @staticmethod
    def get_word_dataframe(
        transcript_df: pd.DataFrame,
        stops_punctuations=list(".。!！?？"),
        nonstop_punctuations=list(",:"),
    ) -> pd.DataFrame:
        words_df = (
            transcript_df[["segment_index", "words"]]
            .loc[transcript_df["words"].apply(len) > 0]
            .explode("words")
        )
        word_flat = pd.DataFrame(words_df["words"].values.tolist())
        word_flat.rename(columns={"word": "text"}, inplace=True)
        words_df.drop(columns=["words"], inplace=True)
        words_df.reset_index(inplace=True, drop=True),
        word_flat.reset_index(inplace=True, drop=True)
        words_df = pd.concat([words_df, word_flat], axis=1).sort_values(
            ["start", "end"]
        )

        words_df["punctuation_stops"] = words_df["text"].apply(
            lambda ss: any(x in stops_punctuations for x in ss)
        )
        words_df["punctuation_stops"] = words_df["punctuation_stops"].shift(
            periods=1, fill_value=False
        )
        words_df["capital_case"] = words_df["text"].apply(lambda ss: ss[0].isupper())

        words_df["punctuation_nonstops"] = (
            words_df["text"]
            .str.strip()
            .apply(lambda ss: any(ss.endswith(x) for x in nonstop_punctuations))
        )

        words_df["pause"] = (
            words_df["start"] - words_df.shift(periods=1, fill_value=0)["end"]
        )
        words_df["segment_split"] = (
            words_df["segment_index"]
            - words_df.shift(periods=1, fill_value=0)["segment_index"]
        )
        words_df["duration"] = words_df["end"] - words_df["start"]
        return words_df

    @staticmethod
    def get_pause_segments_dataframe(
        words_df: pd.DataFrame, min_pause_duration=0.0
    ) -> pd.DataFrame:
        words_df = words_df.copy()
        words_df["new_segment_index"] = (
            (words_df["segment_split"] > 0)
            | (words_df["pause"] > min_pause_duration)
            | words_df["punctuation_stops"]
        ).cumsum()

        grouper_ = words_df.groupby("new_segment_index")
        segment_df = grouper_.agg(
            {"text": "sum", "start": "min", "end": "max", "probability": "mean"}
        )
        segment_df.reset_index(inplace=True, drop=True)
        return segment_df

    @staticmethod
    def _segments_with_duration_limit(
        words_df: pd.DataFrame, max_duration_expected: float
    ) -> np.ndarray:
        new_segment_index = words_df["new_segment_split"].values
        duration = words_df["duration"].values
        segment_split = words_df["segment_split"].values

        groups = np.zeros(len(duration), dtype=bool)
        dynamic_duration = np.zeros(len(duration), dtype=np.float32)
        for i in range(1, len(words_df)):
            dynamic_duration[i] = dynamic_duration[i - 1] + duration[i]
            groups[i] = new_segment_index[i]

            if dynamic_duration[i] >= max_duration_expected and segment_split[i]:
                groups[i] = 1
                dynamic_duration[i] = 0
        return groups.cumsum()

    def build_ngrams(self, words_df):
        _grams = []
        ngrams_index = generate_ngrams(
            list(range(len(words_df) + 1)), max_depth=self.config.pause_ngram_depth
        )
        for id_start, id_stop in tqdm(ngrams_index, desc="Ngrams Formating"):
            seg_df = words_df[id_start:id_stop]
            _grams.append(
                [
                    seg_df["text"].sum(),
                    seg_df["start"].min(),
                    seg_df["end"].max(),
                    seg_df["probability"].mean(),
                    (seg_df["end"] - seg_df["start"]).sum(),
                ]
            )

        df_grams = pd.DataFrame(
            _grams,
            columns=["text", "start", "end", "probability", "words_total_duration"],
        )
        df_grams["pause_duration"] = (
            df_grams["end"] - df_grams["start"] - df_grams["words_total_duration"]
        )
        df_grams["source"] = pd.Series(["ngrams"] * len(df_grams), dtype="category")
        df_grams = df_grams.loc[
            df_grams["pause_duration"] < self.config.max_pause_duration
        ]
        df_grams = df_grams.loc[
            df_grams["end"] - df_grams["start"] <= self.config.max_segment_length
        ]
        df_grams.reset_index(inplace=True, drop=True)
        return df_grams

    def concat_segments_and_ngrams_and_filter(self, *dfs: pd.DataFrame) -> pd.DataFrame:
        columns = ["start", "end", "text", "probability", "source"]
        for df in dfs:
            df.reset_index(inplace=True, drop=True)

        merged_df = pd.concat([df[columns] for df in dfs], axis=0)
        merged_df["duration"] = merged_df["end"] - merged_df["start"]
        merged_df = merged_df.loc[
            merged_df["duration"] > self.config.min_segment_length
        ]
        merged_df.reset_index(inplace=True)
        return merged_df
