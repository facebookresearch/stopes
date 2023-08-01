import asyncio
import torch
import torchaudio
import fairseq
from fairseq.data.data_utils import lengths_to_padding_mask

@torch.inference_mode()
async def load_audiofile(audio_path: str, sample_rate: int, normalize_input: bool) -> torch.Tensor:
    """
    Load the audio files and apply resampling and normalization

    Args:
        audio_path: the audio file path
        sample_rate: the target sampling rate for the ASR model
        normalize_input: the bool value for the ASR model

    Returns:
        audio_waveform: the audio waveform as a torch.Tensor object
    """
    audio_waveform, sampling_rate = await asyncio.to_thread(torchaudio.load, audio_path)
    if audio_waveform.dim == 2:
        audio_waveform = audio_waveform.mean(-1)
    if sample_rate != sampling_rate:
        audio_waveform = torchaudio.functional.resample(audio_waveform, sampling_rate, sample_rate)
    if normalize_input:
        # following fairseq raw audio dataset
        audio_waveform = torch.nn.functional.layer_norm(audio_waveform, audio_waveform.shape)

    return audio_waveform


@torch.inference_mode()
def compute_emissions(audio_input: torch.Tensor, asr_model: fairseq.models.FairseqEncoder, cuda) -> torch.Tensor:
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
    if isinstance(asr_model, fairseq.models.wav2vec.wav2vec2_asr.Wav2VecCtc):
        padding_mask = lengths_to_padding_mask(torch.tensor([audio_input.numel()]))
        emissions = asr_model.w2v_encoder(audio_input, padding_mask)["encoder_out"].transpose(0, 1)
    else:
        emissions = asr_model(audio_input).logits

    return emissions


def decode_emissions(emissions: torch.Tensor, decoder, post_process_fn) -> str:
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


async def transcribe_audiofile(asr_model,audio_path: str, lower=True) -> str:
    """
    Transcribe the audio into a string

    Args:
        model_cfg: the dict of the asr model config
        audio_path: the input audio waveform
        lower: the case of the transcriptions with lowercase as the default

    Returns:
        hypo: the transcription result
    """

    asr_input = await load_audiofile(audio_path, asr_model.sampling_rate, asr_model.normalize_input)
    emissions = compute_emissions(asr_input, asr_model.model, asr_model.use_cuda)
    hypo = decode_emissions(emissions, asr_model.decoder, asr_model.post_process_fn)

    return hypo.strip().lower() if lower else hypo.strip()