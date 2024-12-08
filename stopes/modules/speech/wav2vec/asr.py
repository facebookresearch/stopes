# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Different modules for speech recognition / speech-to-text
import dataclasses
import logging
import typing as tp
from pathlib import Path

import fairseq  # type: ignore[import]
import numpy as np
import submitit
import torch
from fairseq.models.fairseq_model import FairseqModel
from fairseq.tasks.fairseq_task import FairseqTask
from omegaconf import MISSING

from stopes.core import Requirements, StopesModule, utils
from stopes.modules.speech.speech_units import parallel_audio_read
from stopes.modules.speech.wav2vec.utils import WavesDataset
from stopes.speech.asr.wav2vec import decoder as wave2vec_decoder
from stopes.speech.asr.wav2vec.base_decoder import BaseDecoder
from stopes.speech.asr.wav2vec.decoder_config import FlashlightDecoderConfig
from stopes.utils.sharding.text_shards import (
    TextShard,
    make_text_file_shards,
    parse_header,
    resolve_output,
)


@dataclasses.dataclass
class ASRConfig(FlashlightDecoderConfig):
    # shards can be a string, or a list of strings.
    # In case of a string and it is a single file, `nshards` will be used
    # to shard automatically the file
    shards: tp.Any = MISSING
    # column index (0,1) or column name ("src_audio", "tgt_audio",..)
    column: tp.Union[int, str] = MISSING
    output_dir: Path = MISSING

    # Checkpoint to the Wav2vec encoder
    encoder_model: Path = MISSING

    # Post process the sentence to remove special tokens from the tokenizer
    # Permitted values:
    # 'sentencepiece', 'wordpiece', 'letter', 'silence', '_EOW', 'subword_nmt', '@@', "@@ ", 'none'
    post_process_symbol: str = "letter"

    # max tokens to process the CTC batch
    max_tokens: int = 1_280_000

    # How many segments to be loaded into memory at once. This value should
    # be set accordingly to the memory capacity of the cluster node
    buffer_size: int = 50

    # runtime requirements:
    # nshards = no. of shards to split the inputs
    nshards: int = 1
    gpu: bool = True
    cpus_per_task: int = 4


class Wav2vecASR(StopesModule):
    """
    Read a manifest file and apply `fairseq.models.wav2vec.wav2vec_asr.Wav2VecCtc`
    on each segment.
    """

    config: ASRConfig

    def __init__(self, config: ASRConfig, **kwargs):
        super().__init__(config, ASRConfig)
        self.output_dir = Path(self.config.output_dir).resolve()
        self.output_dir.mkdir(exist_ok=True)
        self.header = (
            isinstance(self.config.column, str) and not self.config.column.isdecimal()
        )
        self.logger = logging.getLogger("stopes.asr")
        self.kwargs = kwargs
        self._current_progress: tp.Optional[TextShard] = None

    def load_model_and_task(self) -> tp.Tuple[tp.List[FairseqModel], FairseqTask]:
        """Load a Wav2vec Encoder and the ASR task"""
        models, _, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [str(self.config.encoder_model)], strict=False
        )
        assert len(models) > 0, "No model found in {self.config.encoder_model}"

        # First task should be ASR / audio finetuning
        model = models[0]
        model.make_generation_fast_()
        if self.config.gpu and torch.cuda.is_available():
            model = model.cuda()

        # `fairseq.examples.speech_recognition.new.decoders.base_decoder.BaseDecoder.generate()`
        # expects a list of models (mult-task setting)
        return [model], task

    def requirements(self) -> Requirements:
        # Rule of thumbs: Use maximum 4 cpus per task, 1 task per node
        return Requirements(
            gpus_per_node=int(self.config.gpu),
            cpus_per_task=int(self.config.cpus_per_task),
        )

    def array(self) -> tp.List[TextShard]:
        return list(
            make_text_file_shards(
                self.config.shards,
                nshards=self.config.nshards,
                header=self.header,
                sep="\t",
                col=self.config.column,
            )
        )

    def _generate_sentences(
        self,
        hypos: tp.List[tp.Dict[str, torch.LongTensor]],
        task: FairseqTask,
    ) -> tp.List[str]:
        """
        Extract the words and tokens from the decoded results (best hypotheses)
        """
        sentences = []
        for hypo in hypos:
            if "words" in hypo:
                sentence = " ".join(hypo["words"])
            else:
                tokens = task.target_dictionary.string(hypo["tokens"].int().cpu())
                sentence = fairseq.data.data_utils.post_process(
                    tokens, self.config.post_process_symbol
                )
            sentences.append(sentence)
        return sentences

    def _infer_batch(
        self,
        signals: tp.Iterable[tp.Union[np.ndarray, torch.Tensor]],
        encoders: tp.List[FairseqModel],
        task: FairseqTask,
        decoder: BaseDecoder,
    ) -> tp.List[str]:
        """
        Perform the inference on one batch of auio waveforms:
        - Convert waveforms into fbank features and wrap in an fairseq RauAudioFormat dataset
        - Encode the fbank using fairseq.models.wav2vec.wav2vec_asr.Wav2VecCtc
        - Generate hypotheses from the encodings using fairseq.examples.speech_recognition.new.decoders
        (either by a simple HMM or "viterbi", or by a LM ("KenLM" or "FairseqLM"))
        """
        sizes = [signal.size for signal in signals]
        dataset = WavesDataset(
            signals, sizes, fbank_features=getattr(task, "fbank_features", 80)
        )

        # Shuffle the batch to have the items with (almost) same lengths in one sub-batch
        batch_sampler = dataset.batch_by_size(
            dataset.ordered_indices(),
            max_tokens=self.config.max_tokens,
            max_sentences=None,
            required_batch_size_multiple=1,
        )
        batch_sentences: tp.List[str] = []
        batch_segment_ids = []
        for segment_ids in batch_sampler:
            batch = dataset.collater([dataset[i] for i in segment_ids])
            batch["id"] = torch.IntTensor(segment_ids)
            for k in batch["net_input"].keys():
                if self.config.gpu and torch.cuda.is_available():
                    batch["net_input"][k] = batch["net_input"][k].cuda()

            hypos = decoder.generate(encoders, batch)
            best_hypos = [h[0] for h in hypos]
            sentences = self._generate_sentences(best_hypos, task)
            batch_sentences.extend(sentences)
            batch_segment_ids.append(segment_ids)

        # Sort sentences by segment ids
        sorted_ids = np.concatenate(batch_segment_ids, axis=0)
        return [batch_sentences[i] for i in np.argsort(sorted_ids)]

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> Path:
        shard = iteration_value
        # Shard is None when the run() is called directly, e.g. when running the
        # module from command line: python -m stopes.modules.speech.audio_zip ...
        # In this case we create a dummy Shard object with index = None
        if shard is None:
            assert Path(
                self.config.shards
            ).is_file(), "Direct call of run() only works with a single shard"
            cols = parse_header(self.config.shards, self.header, "\t")
            shard = TextShard(
                input_file=self.config.shards, columns=cols, sep="\t", filter=None
            )
        self._current_progress = shard

        # Set up I/O variables
        output_file = resolve_output(shard, Path(self.config.output_dir), suffix=".tsv")
        assert (
            output_file
        ), f"Cannot determine the output file name for {shard.input_file} (shard #{shard.index})"
        column_offset = shard.resolve_column_index(self.config.column)

        # Setup models variables
        models: tp.Optional[tp.List[FairseqModel]] = None
        task: tp.Optional[FairseqTask] = None
        decoder: tp.Optional[BaseDecoder] = None
        with utils.measure("loading encoder and decoder", logger=self.logger):
            models, task = self.load_model_and_task()
            decoder = wave2vec_decoder(
                self.config, task.target_dictionary, gpu=self.config.gpu
            )

        with shard as f, open(output_file, "a+") as o:
            for lines in utils.batch(iter(f), self.config.buffer_size):
                # lines should be a list now
                audio_signals = []
                with utils.measure("reading batch", logger=self.logger):
                    for _, audio in parallel_audio_read(
                        iter(lines),
                        column_offset=column_offset,
                        gpu=self.config.gpu,
                        # We cannot use fp16 in this module because `torchaudio.compliance.kaldi``
                        # does not support Half tensors yet.
                        fp16=False,
                        num_process=int(self.config.cpus_per_task),
                    ):
                        audio_signals.append(audio)
                sentences = self._infer_batch(audio_signals, models, task, decoder)
                if len(sentences) > 0:
                    o.write("\n".join(sentences))
                    o.write("\n")
        return output_file

    def checkpoint(
        self,
        iteration_value: TextShard,
        iteration_index: int,
        **kwargs: tp.Any,
    ) -> submitit.helpers.DelayedSubmission:
        progress = self._current_progress or iteration_value
        # resubmit the module with updated progress
        return submitit.helpers.DelayedSubmission(
            self, progress, iteration_index, **kwargs
        )
