# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import pandas as pd
import sacrebleu
import torch
import whisper
from tqdm import tqdm
from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer

from stopes.modules.speech.utils import parse_audio

LANG2_LANG3 = {
    "en": "eng",
    "ar": "arb",
    "as": "asm",
    "be": "bel",
    "bg": "bul",
    "bn": "ben",
    "ca": "cat",
    "ckb": "ckb",
    "cs": "ces",
    "cy": "cym",
    "da": "dan",
    "de": "deu",
    "el": "ell",
    "es": "spa",
    "et": "est",
    "fa": "pes",
    "fi": "fin",
    "fr": "fra",
    "ga": "gle",
    "hi": "hin",
    "hu": "hun",
    "id": "ind",
    "it": "ita",
    "ja": "jpn",
    "ka": "kat",
    "ky": "kir",
    "lg": "lug",
    "lt": "lit",
    "lv": "lvs",
    "mn": "khk",
    "mr": "mar",
    "mt": "mlt",
    "nl": "nld",
    "pa": "pan",
    "pl": "pol",
    "pt": "por",
    "ro": "ron",
    "ru": "rus",
    "sk": "slk",
    "sl": "slv",
    "sw": "swh",
    "ta": "tam",
    "th": "tha",
    "tr": "tur",
    "uk": "ukr",
    "ur": "urd",
    "uz": "uzn",
    "vi": "vie",
    "yue": "yue",
    "zh": "cmn",
    "af": "afr",
    "is": "isl",
    "lb": "ltz",
    "no": "nob",
    "gl": "glg",
    "kea": "kea",
    "bs": "bos",
    "hr": "hrv",
    "mk": "mkd",
    "sr": "srp",
    "az": "azj",
    "kk": "kaz",
    "ko": "kor",
    "gu": "guj",
    "kn": "kan",
    "or": "ory",
    "sd": "snd",
    "te": "tel",
    "ceb": "ceb",
    "jv": "jav",
    "ms": "zlm",
    "ml": "mal",
    "tl": "tgl",
    # "tl": "fil",
    "my": "mya",
    "km": "khm",
    "lo": "lao",
    "he": "heb",
    "ps": "pbt",
    "tg": "tgk",
    "am": "amh",
    "ig": "ibo",
    "ln": "lin",
    "nso": "nso",
    "so": "som",
    "xh": "xho",
    "yo": "yor",
    "zu": "zul",
    "kam": "kam",
    "luo": "luo",
    "ny": "nya",
    "om": "gaz",
    "sn": "sna",
    "umb": "umb",
    "ga-IE": "gle",
    "sv": "swe",
    "ast": "ast",
    "ff": "ful",
    "mi": "mri",
    "ha": "hau",
    "wo": "wol",
    "oc": "oci",
    "ilo": "ilo",
    "ba": "bak",
    "br": "bre",
    "fy": "fry",
    "yi": "yid",
    "tn": "tsn",
    "gd": "gla",
    "ht": "hat",
    "mg": "mlg",
    "ns": "nso",
    "si": "sin",
    "sq": "sqi",
    "ss": "ssw",
    "su": "sun",
    # "zh-HK": "zh-HK",
    "ab": "abk",
    "bas": "bas",
    "cnh": "cnh",
    "cv": "chv",
    "dv": "div",
    "eo": "epo",
    "eu": "eus",
    "fy-NL": "fry",
    "gn": "grn",
    "hsb": "hsb",
    "hy": "hye",
    "ia": "ina",
    "kab": "kab",
    "kmr": "kmr",
    "mdf": "mdf",
    "mhr": "mhr",
    "myv": "myv",
    "nan-tw": "hbl",
    "ne": "npi",
    "nn-NO": "nno",
    "rm-sursilv": "rm-sursilv",
    "rm-vallader": "rm-vallader",
    "rw": "kin",
    "sah": "sah",
    "sat": "sat",
    "sc": "srd",
    "tig": "tig",
    "tok": "tok",
    "tt": "tat",
    "ug": "uig",
    "vot": "vot",
    "mrj": "mrj",
    "skr": "skr",
    "ti": "tir",
    "tw": "twi",
    "bo": "bod",
    "fo": "fao",
    "gv": "glv",
    "haw": "haw",
    "la": "lat",
    "sa": "san",
    "sco": "sco",
    "war": "war",
    "jw": "jav",
    "nn": "nno",
    "tk": "tuk",
}
LANG3_LANG2 = {v: k for k, v in LANG2_LANG3.items()}


def init_whisper_model(whisper_model_tag: str = "large"):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    asr_model = whisper.load_model(name=whisper_model_tag, device=device)
    return asr_model


def transcribe_series(
    audio_paths_series: pd.Series,
    asr_model,
    audio_lang: str,
    beam_size: int = 1,
    temperature: float = 0.0,
):
    """Transcribes each audio filepath from series and returns series of transcriptions

    Args:
        audio_paths_series (pd.Series): each line contains path to audio that could be
        asr_model: ASR model to do the transcribing process e.g. Whisper
        audio_lang str: what language is used in the given audio, used by ASR model
        beam_size int: whisper beam size. Defaults to 1
        temperature float: whisper temperature. Defaults to 0.0 to avoid fallback decoding (see details below).

    Returns:
        pd.Series: Series where each line has a transcription of corresponding audio from audio_paths_series


    Whisper model implements decoding with fallback: https://github.com/openai/whisper/blob/main/whisper/transcribe.py#L147
    The core idea is that decoding at each time step might happen multiple times if at least one criterion to "fall back" i.e.
    start over is fired. Number of fallback iterations is determined by the schedule of temperature values:
    https://github.com/openai/whisper/blob/main/whisper/transcribe.py#L41
    By default this schedule is active and temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0) i.e. even with beam_size 5 it might fell back and
    turn on sampling by using temperature > 0, in this case the beam search is not used in the fall back iteration.
    Explicit setting of temperature=0.0 overwrites the schedule and fall back decoding has only one for loop iteration i.e. no fall backs.
    This allows us to do reproducible evaluation without sample variations. Beware that this might introduce the repetition loops in
    the transcriptions and lead to worse ASR-BLEU score in the end.
    """

    if len(audio_lang) == 3:
        # to make it work with whisper
        audio_lang = LANG3_LANG2[audio_lang]

    transcriptions = {}

    for idx, audio_path in tqdm(
        audio_paths_series.items(),
        desc=f"Transcribing {audio_paths_series.name} column",
        total=len(audio_paths_series),
    ):
        if ".zip" in audio_path:
            sampling_factor = (
                16  # hardcoded as most of our audiozips are 16kHz sampling rate
            )
        else:
            sampling_factor = None
        audio = parse_audio(audio_path, sampling_factor=sampling_factor).load()[0]
        hypo = asr_model.transcribe(
            audio, temperature=temperature, beam_size=beam_size, language=audio_lang
        )["text"].strip()
        transcriptions[idx] = hypo

    transcriptions_series = pd.Series(transcriptions)
    transcriptions_series.name = f"{audio_paths_series.name}_transcribed"

    return transcriptions_series


def whisper_normalize_series(transcription_series: pd.Series, text_lang: str):
    """Normalizes the text series using whisper noramlizer. English has a specific one in whisper package.

    Args:
        transcription_series (pd.Series): Each line contains arbitrary text written in text_lang
        text_lang (str): Language of the text in series

    Returns:
        pd.Series: Series with normalized text
    """
    if text_lang == "eng":
        normalizer = EnglishTextNormalizer()
    else:
        normalizer = BasicTextNormalizer()

    norm_transcriptions = {}

    for idx, text in transcription_series.items():
        if text_lang == "cmn":
            try:
                import chinese_converter
            except ImportError:
                raise ImportError(
                    "Please install chinese_converter: pip install chinese_converter"
                )
            text = chinese_converter.to_simplified(text)

        norm_transcriptions[idx] = normalizer(text)

    norm_transcriptions_series = pd.Series(norm_transcriptions)
    norm_transcriptions_series.name = transcription_series.name

    return norm_transcriptions_series


def compute_asr_bleu(
    audio_paths_series: pd.Series,
    ref_text_series: pd.Series,
    lang: str,
    asr_model,
    whisper_normalize_text=True,
    return_transcriptions=False,
):
    """Wraps functions above to compute corpus-level ASR-BLEU

    ASR decoding hyper-parameters are hard coded to ensure reproducibility across evaluations

    Args:
        audio_paths_series (pd.Series): each line contains path to audio
        ref_text_series (pd.Series): each line contains the text reference to compare audio with
        lang (str): the language of both audio and ref_text
        asr_model: whisper ASR model
        whisper_normalize_text: normalize both text hypotheses and reference if True. Defaults to True.
        return_transcriptions (bool): return a dataframe with the ASR transcripts and references. Defaults to False.
    """

    beam_size = 1
    temperature = 0.0
    audio_transcriptions = transcribe_series(
        audio_paths_series,
        asr_model,
        audio_lang=lang,
        beam_size=beam_size,
        temperature=temperature,
    )

    asr_bleu, asr_bleu_signature = compute_corpus_bleu(
        audio_transcriptions, ref_text_series, lang, whisper_normalize_text
    )
    asr_bleu_signature.info["whisper_asr_beam_size"] = beam_size
    asr_bleu_signature.info["whisper_asr_temperature"] = temperature
    asr_bleu_signature.info["whisper_asr_language"] = lang

    transcript_df = None
    if return_transcriptions:
        transcript_df = pd.concat(
            [
                audio_paths_series,
                audio_transcriptions,
                ref_text_series,
            ],
            axis=1,
            keys=["audio", "transcript", "reference"],
        )
    return asr_bleu, asr_bleu_signature, transcript_df


def get_bleu_tokenizer(lang):
    lang_tok_map = {
        "cmn": "char",
        "jpn": "char",
        "tha": "char",
        "lao": "char",
        "mya": "char",
    }
    tok = lang_tok_map.get(lang, "13a")  # 13a is the default tokenizer
    return tok


def compute_corpus_bleu(
    hyp_text_series: pd.Series,
    ref_text_series: pd.Series,
    lang: str,
    whisper_normalize_text=True,
):
    """Wraps normalization functions and compute corpus-level BLEU score

    Args:
        hyp_text_series (pd.Series): each line contains s2t model prediction or first pass prediction
        ref_text_series (pd.Series): _description_
        lang (str): _description_
        whisper_normalize_text (bool, optional): normalize both text hypotheses and reference if True. Defaults to True.

    Returns:
        (BLEUScore, BLEUScoreSignature)
    """
    if whisper_normalize_text:
        hyp_text_series = whisper_normalize_series(hyp_text_series, lang)
        ref_text_series = whisper_normalize_series(ref_text_series, lang)

    tokenizer_name = get_bleu_tokenizer(lang)
    corpus_bleu_metric = sacrebleu.metrics.bleu.BLEU(
        lowercase=whisper_normalize_text, tokenize=tokenizer_name
    )  # lowercase applied if we use whisper_normalize_text

    corpus_bleu = corpus_bleu_metric.corpus_score(
        hyp_text_series.to_list(), [ref_text_series.to_list()]
    )
    corpus_bleu_signature = corpus_bleu_metric.get_signature()
    corpus_bleu_signature.info["whisper_normalize"] = whisper_normalize_text

    return corpus_bleu, corpus_bleu_signature


def compute_translation_quality_expressivity(
    output_manifest_tsv_path: str,
    output_dir: str,
    tgt_lang: str,
    whisper_model_tag="large",
    print_to_stdout=True,
):
    """Wraps asr and s2t bleu functions to call it with TSV manifest composed on expressivity side

    Args:
        output_manifest_tsv_path (str): output manifest which has "ref_text", "hypo_audio", "s2t_out" column names
        tgt_lang (str): what language we evaluate on
        whisper_model (str, optional): Whisper model tag. Defaults to "large".
    """
    df = pd.read_csv(
        output_manifest_tsv_path,
        sep="\t",
        quoting=3,
    )

    whisper_model = init_whisper_model(whisper_model_tag)

    # S2T first pass evaluation with normalization
    s2t_bleu_normalized, s2t_bleu_normalized_signature = compute_corpus_bleu(
        hyp_text_series=df["s2t_out"],
        ref_text_series=df["ref_text"],
        lang=tgt_lang,
        whisper_normalize_text=True,
    )
    s2t_bleu_normalized_json = s2t_bleu_normalized.format(
        signature=s2t_bleu_normalized_signature.format(), is_json=True
    )

    # S2TT evaluation without (!) normalization
    s2t_bleu, s2t_bleu_signature = compute_corpus_bleu(
        hyp_text_series=df["s2t_out"],
        ref_text_series=df["ref_text"],
        lang=tgt_lang,
        whisper_normalize_text=False,
    )
    s2t_bleu_json = s2t_bleu.format(signature=s2t_bleu_signature.format(), is_json=True)

    asr_bleu_normalized, asr_bleu_normalized_signature, _ = compute_asr_bleu(
        audio_paths_series=df["hypo_audio"],
        ref_text_series=df["ref_text"],
        lang=tgt_lang,
        asr_model=whisper_model,
        whisper_normalize_text=True,
    )
    asr_bleu_normalized_signature.info["whisper_asr_model"] = whisper_model_tag

    asr_bleu_normalized_json = asr_bleu_normalized.format(
        signature=asr_bleu_normalized_signature.format(), is_json=True
    )

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(
        (Path(output_dir) / "s2st_asr_bleu_normalized.json").as_posix(), "w"
    ) as f:
        f.write(asr_bleu_normalized_json)

    with open((Path(output_dir) / "s2tt_bleu_normalized.json").as_posix(), "w") as f:
        f.write(s2t_bleu_normalized_json)

    with open((Path(output_dir) / "s2tt_bleu.json").as_posix(), "w") as f:
        f.write(s2t_bleu_json)

    if print_to_stdout:
        print(f"S2ST ASR Normalized BLEU:\n{asr_bleu_normalized_json}")
        print(f"S2TT Normalized BLEU:\n{s2t_bleu_normalized_json}")
        print(f"S2TT BLEU:\n{s2t_bleu_json}")


def test_corpus_bleu_manual():
    whisper_model = "large"
    whisper_large_model = init_whisper_model(whisper_model)
    output_tsv = Path("~/asr_eval_test/output_manifest.tsv").expanduser()
    df = pd.read_csv(output_tsv, sep="\t")
    tgt_lang = "eng"
    asr_bleu, asr_bleu_signature, transcripts = compute_asr_bleu(
        audio_paths_series=df["hypo_audio"],
        ref_text_series=df["ref_text"],
        lang=tgt_lang,
        asr_model=whisper_large_model,
        return_transcriptions=True,
    )
    asr_bleu_signature.info["whisper_asr_model"] = whisper_model
    asr_bleu_json = asr_bleu.format(signature=asr_bleu_signature.format(), is_json=True)
    print(transcripts)
    print(asr_bleu_json)
    return asr_bleu_json
