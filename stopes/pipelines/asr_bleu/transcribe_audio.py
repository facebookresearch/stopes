import torch
import typing as tp
import torchaudio
import fairseq
import logging
from tqdm import tqdm
from dataclasses import dataclass

from fairseq.data.data_utils import lengths_to_padding_mask
from omegaconf.omegaconf import MISSING

from stopes.core.launcher import Launcher
from stopes.core.stopes_module import Requirements, StopesModule
from stopes.pipelines.asr_bleu.utils import ASRContainer


@dataclass
class TranscribeAudioJob:
    eval_manifest: tp.Dict[str, tp.List] = MISSING
    asr_config: tp.Dict = MISSING


@dataclass
class TranscribeAudioConfig:
    transcribe_audio_jobs: tp.List[TranscribeAudioJob] = MISSING


class TranscribeAudio(StopesModule):
    def __init__(self, config: TranscribeAudioConfig):
        super().__init__(config=config, config_class=TranscribeAudioConfig)

    def array(self):
        return self.config.transcribe_audio_jobs

    def requirements(self) -> Requirements:
        return Requirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=1,
            timeout_min=720,
        )

    @torch.inference_mode()
    def _load_audiofile(
        self,
        audio_path: str,
        sample_rate: int,
        normalize_input: bool
    ) -> torch.Tensor:
        """
        Load the audio files and apply resampling and normalization

        Args:
            audio_path: the audio file path
            sample_rate: the target sampling rate for the ASR model
            normalize_input: the bool value for the ASR model

        Returns:
            audio_waveform: the audio waveform as a torch.Tensor object
        """
        audio_waveform, sampling_rate = torchaudio.load(audio_path)
        if audio_waveform.dim == 2:
            audio_waveform = audio_waveform.mean(-1)
        if sample_rate != sampling_rate:
            audio_waveform = torchaudio.functional.resample(
                audio_waveform,
                sampling_rate,
                sample_rate
            )
        if normalize_input:
            # following fairseq raw audio dataset
            audio_waveform = torch.nn.functional.layer_norm(
                audio_waveform,
                audio_waveform.shape
            )

        return audio_waveform

    @torch.inference_mode()
    def _compute_emissions(
        self,
        audio_input: torch.Tensor,
        asr_model: fairseq.models.FairseqEncoder,
        cuda
    ) -> torch.Tensor:
        """
        Compute the emissions for either fairseq or huggingface asr model

        Args:
            audio_input: the input audio waveform
            asr_model: the ASR model to use for computing emissions

        Returns:
            emissions: the logits of the encoded prediction.
        """

        if cuda:
            audio_input = audio_input.to("cuda")
        if isinstance(
            asr_model, fairseq.models.wav2vec.wav2vec2_asr.Wav2VecCtc
        ):
            padding_mask = lengths_to_padding_mask(
                torch.tensor([audio_input.numel()])
            )
            emissions = asr_model.w2v_encoder(
                audio_input,
                padding_mask
            )["encoder_out"].transpose(0, 1)
        else:
            emissions = asr_model(audio_input).logits

        return emissions

    def _decode_emissions(
        self,
        emissions: torch.Tensor,
        decoder,
        post_process_fn
    ) -> str:
        """
        Decode the emissions and apply post-process functions

        Args:
            emissions: the input Tensor object
            decoder: ASR decoder
            post_process_fn: post-process function for the ASR output

        Returns:
            hypo: the str as the decoded transcriptions
        """

        emissions = emissions.cpu()
        results = decoder(emissions)

        # assuming the lexicon-free decoder and working with tokens
        hypo = decoder.idxs_to_tokens(results[0][0].tokens)
        hypo = post_process_fn(hypo)

        return hypo

    def _merge_tailo_init_final(self, text):
        """
        Hokkien ASR hypothesis post-processing.
        """
        sps = text.strip().split()
        results = []
        last_syllable = ""
        for sp in sps:
            if sp == "NULLINIT" or sp == "nullinit":
                continue
            last_syllable += sp
            if sp[-1].isnumeric():
                results.append(last_syllable)
                last_syllable = ""
        if last_syllable != "":
            results.append(last_syllable)
        return " ".join(results)

    def _transcribe_audiofile(
        self,
        asr_model,
        audio_path: str,
        lower=True
    ) -> str:
        """
        Transcribe the audio into a string

        Args:
            model_cfg: the dict of the asr model config
            audio_path: the input audio waveform
            lower: the case of the transcriptions with lowercase as the default

        Returns:
            hypo: the transcription result
        """

        asr_input = self._load_audiofile(
            audio_path,
            asr_model.sampling_rate,
            asr_model.normalize_input
        )
        emissions = self._compute_emissions(
            asr_input,
            asr_model.model,
            asr_model.use_cuda
        )
        hypo = self._decode_emissions(
            emissions,
            asr_model.decoder,
            asr_model.post_process_fn
        )

        return hypo.strip().lower() if lower else hypo.strip()

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> tp.List[str]:

        assert iteration_value is not None, "Iteration value is null"
        self.logger = logging.getLogger("stopes.asr_bleu.transcribe_audio")
        asr_model = ASRContainer(iteration_value.asr_config)

        prediction_transcripts = []

        for prediction in tqdm(
            iterable=iteration_value.eval_manifest["prediction"],
            desc="Transcribing predictions",
            total=len(iteration_value.eval_manifest["prediction"]),
        ):
            self.logger.info(f"Transcribing {prediction}")
            transcription = self._transcribe_audiofile(asr_model, prediction)
            prediction_transcripts.append(transcription.lower())

        if iteration_value.asr_config.lang == "hok":
            prediction_transcripts = [
                self._merge_tailo_init_final(
                    text
                ) for text in prediction_transcripts
            ]

        return prediction_transcripts


async def transcribe_audio(
    eval_manifests: tp.List[tp.Dict[str, tp.List]],
    launcher: Launcher,
    asr_config,
):
    """
    Transcribes audio from a list of audio files
    Returns a list of lists of transcriptions
    Datasets are lists of audio file paths
    """
    transcribe_audio_jobs = [
        TranscribeAudioJob(
            eval_manifest=eval_manifest,
            asr_config=asr_config,
        ) for eval_manifest in eval_manifests
    ]
    transcribe_audio_module = TranscribeAudio(
        TranscribeAudioConfig(
            transcribe_audio_jobs=transcribe_audio_jobs,
        ),
    )
    transcribed_audio = await launcher.schedule(transcribe_audio_module)

    return transcribed_audio
