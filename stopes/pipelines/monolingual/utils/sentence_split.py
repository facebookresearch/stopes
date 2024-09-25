# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
import typing as tp
from pathlib import Path

# --- sentence splitters
# Moses-style
from sentence_splitter import SentenceSplitter

from stopes.pipelines.monolingual.language_utils.bod.bod_sentenizer import BodSentenizer
from stopes.pipelines.monolingual.language_utils.urd.urdu_sentenizer import (
    urdu_sentence_tokenizer,
)

# Indicp NLP


INDIC_NLP_RESOURCES = None  # apparently not needed for splitting and normalization

# pythainlp for Thai
# Seahorse for Indonesian, Thai, Vietnamese
# botok for tibetan
# Spacy for various tool-kits

logger = logging.getLogger("sentence_split")


def map_lang(lang: str, equivalence_file: Path) -> str:
    assert (
        equivalence_file.is_file()
    ), f"Language equivalence file {equivalence_file} does not exist"
    map_equiv = {}
    with equivalence_file.open("r", encoding="utf-8") as fin:
        for line_num, line in enumerate(fin):
            if line.startswith("#"):
                continue
            fields = line.strip().split(maxsplit=2)
            assert (
                len(fields) >= 2
            ), f"""
                Format error in Language equivalence file {equivalence_file}:{line_num}
                Line: {line}
            """
            map_equiv[fields[0]] = fields[1]

    logger.info(
        f"loaded {len(map_equiv)} entries in language mapping file '{equivalence_file}'"
    )
    if lang in map_equiv:
        mapped_lang = map_equiv[lang]
        logger.info(f"mapping language '{lang}' to '{mapped_lang}'")
        return mapped_lang

    return lang


# ----------------------------------
# Supported tokenization algorithms
# List of supported languages and mapping ISO3 - > ISO2

LANGS_MOSES = {
    "cat": "ca",
    "ces": "cs",
    "dan": "da",
    "nld": "nl",
    "eng": "en",
    "fin": "fi",
    "fra": "fr",
    "deu": "de",
    "ell": "el",
    "hun": "hu",
    "isl": "is",
    "ita": "it",
    "lav": "lv",
    "lit": "lt",
    "nob": "no",
    "pol": "pl",
    "por": "pt",
    "ron": "ro",
    "rus": "ru",
    "slk": "sk",
    "slv": "sl",
    "spa": "es",
    "swe": "sv",
    "tur": "tr",
}

LANGS_THAINLP = {"tha": "tha"}
LANGS_LAONLP = {"lao": "lao"}
LANGS_KHMER = {"khm": "khm"}
LANGS_BOTOK = {
    "bod": "bod",
    "dzo": "dzo",
}  # languages with tibetan script
LANGS_URDU = {"urd": "urd"}
LANGS_HASAMI = {"jpn": "jpn"}

# ----------------------------------------------
LANGS_CASELESS = {"kat": "kat"}


def make_splitter_caseless(
    base_splitter: tp.Callable[[str], tp.Iterable[str]]
) -> tp.Callable[[str], tp.Iterable[str]]:
    """
    Try splitting an uppercase version of the texts.
    Works for languages like Georgian, where the texts are written in lowercase,
    where otherwise the default splitter ignores full stops, because they are not followed by uppercase letters.
    The splitter is applied by upper-casing the text, then running a base splitter on it,
    but returning the corresponding substrings of the original text.
    Two conditions must hold:
    - After upper-casing, the text doesn't change lenght;
    - The sentences returned by the base splitter are substrings of the text being split.
    If they do not hold, falling back to just running the base splitter.
    """

    def caseless_splitter(text: str) -> tp.List[str]:
        upper = text.upper()
        if len(text) != len(upper):
            logger.warning(
                f"The text changes length at uppercasing, so caseless splitter could not be applied."
                f"Falling back to the case-dependend splitter for `{text[:20]}...`"
            )
            return list(base_splitter(text))
        upper_sents = base_splitter(upper)
        sents = []
        start_id = 0
        for upper_sent in upper_sents:
            start_id = upper.find(upper_sent, start_id)
            if start_id == -1:
                logger.warning(
                    f"A sentence `{upper_sent}...` was not found in the text in a caseless-splitter."
                    f"Falling back to the case-dependend splitter for `{upper[:20]}...`"
                )
                return list(base_splitter(text))
            sents.append(text[start_id : start_id + len(upper_sent)])
        return sents

    return caseless_splitter


# ----------------------------------------------
LANGS_INDIC = {
    "asm": "as",
    "ben": "bn",
    "brx": "bD",
    "gom": "xx",
    "guj": "gu",
    "hin": "hi",
    "kan": "kn",
    "kok": "kK",
    "mni": "bn",  # our meitei is in bengali script, so swapped it to bengali here
    "mal": "ml",
    "mar": "mr",
    "npi": "ne",
    "ory": "or",
    "pan": "pa",
    "san": "sa",
    "snd": "sd",
    "tam": "ta",
    "tel": "te",
    # "urd": "ur", # For Urdu, a special splitter is applied instead
}

# ----------------------------------------------
LANGS_GEEZ = {"amh": "amh", "tir": "tir"}


def split_geez(line: str) -> tp.Iterable[str]:
    """Split Amharic text into sentences."""
    line = line.replace("፡፡", "።")
    # remove "•" if there's already EOS marker before
    line = (
        line.replace("። •", "።")
        .replace("? •", "?")
        .replace("! •", "!")
        .replace(". •", ".")
    )
    for sent in re.findall(r"[^።•!?\!\?\.]+[።•!?।৷\?\!\.]?", line, flags=re.U):
        yield sent


# ----------------------------------------------
LANGS_OLCHIKI = {"san": "san"}


def split_olchiki(line: str) -> tp.Iterable[str]:
    """Split Santali text into sentences."""
    for sent in re.findall(r"[^᱾|᱿!?\!\?]+[᱾|᱿!?\?\!]?", line, flags=re.U):
        yield sent


# test sentence: ᱱᱤᱭᱟᱹ ᱣᱤᱠᱤᱯᱤᱰᱤᱭᱟ ᱫᱚ ᱥᱟᱱᱛᱟᱲᱤ ᱛᱮ ᱚᱞ ᱟᱠᱟᱱᱟ᱾ ᱚᱨᱦᱚᱸ ᱮᱴᱟᱜ ᱯᱟᱹᱨᱥᱤᱛᱮ ᱦᱚᱸ ᱟᱭᱢᱟ ᱣᱤᱠᱤᱯᱤᱰᱤᱭᱟ ᱢᱮᱱᱟᱜᱼᱟ ᱾ ᱱᱚᱸᱰᱮ ᱠᱤᱪᱷᱩ ᱛᱟᱹᱞᱠᱟᱹ ᱮᱢ ᱦᱩᱭᱱᱟ ᱾
# splits three times


# ----------------------------------------------
LANGS_BURMESE = {"mya": "mya", "shn": "shn"}


def split_burmese(line: str) -> tp.Iterable[str]:
    """Split Amharic text into sentences."""
    # remove "•" if there's already EOS marker before
    line = line.replace("။”", "APOS။")
    for sent in re.findall(r"[^။!?\!\?\.]+[။!?।৷\?\!\.]?", line, flags=re.U):
        yield sent.replace("APOS။", "။”")


# ----------------------------------------------
LANGS_CHINESE = {"zho": "zho", "zho_Hans": "zho_Hans"}


def split_chinese(line: str) -> tp.Iterable[str]:
    """
    Split Chinese text into sentences.
    From https://stackoverflow.com/questions/27441191/splitting-chinese-document-into-sentences
    Special question/exclamation marks were added upon inspection of our raw data
    """
    for sent in re.findall(r"[^!?。\.\!\?\！\？\．]+[!?。\.\!\?\！\？\．]?", line, flags=re.U):
        yield sent


# ----------------------------------------------
LANGS_ARMENIAN = {"hye": "hye"}


def armenian_tokenize_naive(text):
    prev_end = 0
    results = []
    for found in re.finditer("[.:։՜՞] ", text):
        sentence = text[prev_end : max(prev_end, found.end() - 1)].strip()
        if sentence:
            results.append(sentence)
        prev_end = found.end()
    sentence = text[prev_end:]
    if sentence:
        results.append(sentence)
    return results


# ----------------------------------


def get_split_algo(lang: str, split_algo: str) -> tp.Callable[[str], tp.Iterable[str]]:
    # get default algorithm if requested
    if split_algo == "default":
        # use best algorithm in function of language
        if lang in LANGS_MOSES:
            split_algo = "moses"
        elif lang in LANGS_INDIC:
            split_algo = "indic"
        elif lang in LANGS_GEEZ:
            split_algo = "geez"
        elif lang in LANGS_KHMER:
            split_algo = "khmer"
        elif lang in LANGS_BURMESE:
            split_algo = "burmese"
        elif lang in LANGS_CHINESE:
            split_algo = "chinese"
        elif lang in LANGS_THAINLP:
            split_algo = "thai"
        elif lang in LANGS_URDU:
            split_algo = "urduhack"
        elif lang in LANGS_HASAMI:
            split_algo = "hasami"
        elif lang in LANGS_BOTOK:
            split_algo = "botok"
        elif lang in LANGS_ARMENIAN:
            split_algo = "armenian"
        elif lang in LANGS_CASELESS:
            split_algo = "moses_caseless"
        else:
            # use Moses by default (which likely will fall-back to English)
            split_algo = "moses"
        logger.info(f" - default algorithm for {lang} is {split_algo}")

    if split_algo == "none" or lang == "TODO":
        logger.info(" - no sentence splitting")
        return lambda line: [line]

    elif split_algo == "moses" or split_algo == "moses_caseless":
        if lang in LANGS_MOSES:
            lang = LANGS_MOSES[lang]
            logger.info(f" - Moses sentence splitter: using rules for '{lang}'")
        else:
            lang = "en"
            logger.info(
                f" - Moses sentence splitter for {lang}: falling back to {lang} rules"
            )
        splitter = SentenceSplitter(language=lang)
        if split_algo == "moses_caseless":
            return make_splitter_caseless(splitter.split)
        # non_breaking_prefix_file=non_breaking_prefix_file
        return splitter.split

    elif split_algo == "indic":
        # initialize toolkit (apparently not needed for sentence segmentation)
        from indicnlp import common as indic_common
        from indicnlp import loader as indic_loader
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
        from indicnlp.tokenize import sentence_tokenize as indic_sent_tok

        if INDIC_NLP_RESOURCES:
            logger.info(" - Initialize Indic NLP toolkit")
            indic_common.set_resources_path(INDIC_NLP_RESOURCES)
            indic_loader.load()
        if lang in LANGS_INDIC:
            lang = LANGS_INDIC[lang]
            logger.info(f" - Indic sentence splitter: using rules for '{lang}'")
        else:
            lang = "hi"
            logger.info(
                f" - Indic sentence splitter for {lang}: falling back to {lang} rules"
            )

        # setup normalizer
        factory = IndicNormalizerFactory()
        indic_normalizer = factory.get_normalizer(lang)

        def split_indic(line: str) -> tp.Iterable[str]:
            """Split Indian text into sentences using Indic NLP tool."""
            line = indic_normalizer.normalize(line)
            for sent in indic_sent_tok.sentence_split(line, lang=lang):
                yield sent

        return split_indic

    elif split_algo == "laonlp":
        logger.info(f" - LaoNLP sentence splitter applied to '{lang}'")
        from laonlp.tokenize import sent_tokenize as lao_sent_tok

        return lao_sent_tok

    elif split_algo == "khmer":
        from khmernltk import sentence_tokenize as khm_sent_tok

        logger.info(f" - Khmer NLTK sentence splitter applied to '{lang}'")
        return khm_sent_tok

    elif split_algo == "botok":
        logger.info(f" - Tibetan NLTK sentence splitter applied to '{lang}'")
        sentenizer = BodSentenizer()

        return sentenizer

    elif split_algo == "geez":
        logger.info(f" - Ge'ez rule-based sentence splitter applied to '{lang}'")
        return split_geez

    elif split_algo == "burmese":
        logger.info(f" - Burmese rule-based sentence splitter applied to '{lang}'")
        return split_burmese

    elif split_algo == "chinese":
        logger.info(f" - Chinese rule-based sentence splitter applied to '{lang}'")
        return split_chinese

    elif split_algo == "thai":
        logger.info(f" - PyThaiNLP sentence splitter applied to '{lang}'")
        from pythainlp import sent_tokenize

        return sent_tokenize  # type: ignore[no-any-return]

    elif split_algo == "urduhack":
        logger.info(f" - Urdu Hack sentence splitter applied to '{lang}'")
        return urdu_sentence_tokenizer

    elif split_algo == "hasami":
        logger.info(f" - Hasami sentence splitter applied to '{lang}'")
        from hasami import segment_sentences

        return segment_sentences

    elif split_algo == "armenian":
        logger.info(f" - Armenian sentence splitter applied to '{lang}'")

        return armenian_tokenize_naive

    raise ValueError(f"Unknown splitting algorithm {split_algo}")
