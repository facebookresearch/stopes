# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

import os
import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import sentencepiece as spm
from holistic_bias.src.sentences import HolisticBiasSentenceGenerator

# HolisticBias params (ideally we wouldn't hardcode these in)
NUMBERS_OF_NOUNS = {"female": 10, "male": 11, "total": 30}
# The current number of nouns in the dataset
NONE_STRING = "(none)"

# Analysis params
LANG_ALLOWLIST = [
    "ace_Latn",
    "acm_Arab",
    "acq_Arab",
    "aeb_Arab",
    "afr_Latn",
    "ajp_Arab",
    "aka_Latn",
    "als_Latn",
    "amh_Ethi",
    "apc_Arab",
    "arb_Arab",
    "ars_Arab",
    "ary_Arab",
    "arz_Arab",
    "ast_Latn",
    "ayr_Latn",
    "azb_Arab",
    "azj_Latn",
    "bak_Cyrl",
    "bam_Latn",
    "ban_Latn",
    "bel_Cyrl",
    "bem_Latn",
    "bjn_Arab",
    "bjn_Latn",
    "bos_Latn",
    "bug_Latn",
    "bul_Cyrl",
    "cat_Latn",
    "ceb_Latn",
    "ces_Latn",
    "cjk_Latn",
    "ckb_Arab",
    "crh_Latn",
    "cym_Latn",
    "dan_Latn",
    "deu_Latn",
    "dik_Latn",
    "dyu_Latn",
    "dzo_Tibt",
    "ell_Grek",
    "epo_Latn",
    "est_Latn",
    "eus_Latn",
    "ewe_Latn",
    "fao_Latn",
    "fij_Latn",
    "fin_Latn",
    "fon_Latn",
    "fra_Latn",
    "fur_Latn",
    "fuv_Latn",
    "gaz_Latn",
    "gla_Latn",
    "gle_Latn",
    "glg_Latn",
    "grn_Latn",
    "hat_Latn",
    "hau_Latn",
    "heb_Hebr",
    "hrv_Latn",
    "hye_Armn",
    "ilo_Latn",
    "ind_Latn",
    "isl_Latn",
    "ita_Latn",
    "jav_Latn",
    "kab_Latn",
    "kac_Latn",
    "kam_Latn",
    "kas_Arab",
    "kat_Geor",
    "kaz_Cyrl",
    "kbp_Latn",
    "kea_Latn",
    "khk_Cyrl",
    "kik_Latn",
    "kin_Latn",
    "kir_Cyrl",
    "kmb_Latn",
    "kmr_Latn",
    "knc_Arab",
    "knc_Latn",
    "kon_Latn",
    "lij_Latn",
    "lim_Latn",
    "lin_Latn",
    "lit_Latn",
    "lmo_Latn",
    "ltg_Latn",
    "ltz_Latn",
    "lua_Latn",
    "lug_Latn",
    "luo_Latn",
    "lus_Latn",
    "lvs_Latn",
    "min_Latn",
    "mkd_Cyrl",
    "mlt_Latn",
    "mos_Latn",
    "mri_Latn",
    "nld_Latn",
    "nno_Latn",
    "nob_Latn",
    "nso_Latn",
    "nus_Latn",
    "nya_Latn",
    "oci_Latn",
    "pap_Latn",
    "pbt_Arab",
    "pes_Arab",
    "plt_Latn",
    "pol_Latn",
    "por_Latn",
    "prs_Arab",
    "quy_Latn",
    "ron_Latn",
    "run_Latn",
    "rus_Cyrl",
    "sag_Latn",
    "scn_Latn",
    "slk_Latn",
    "slv_Latn",
    "smo_Latn",
    "sna_Latn",
    "snd_Arab",
    "som_Latn",
    "sot_Latn",
    "spa_Latn",
    "srd_Latn",
    "srp_Cyrl",
    "ssw_Latn",
    "sun_Latn",
    "swe_Latn",
    "swh_Latn",
    "szl_Latn",
    "tat_Cyrl",
    "tgk_Cyrl",
    "tgl_Latn",
    "tir_Ethi",
    "tpi_Latn",
    "tsn_Latn",
    "tso_Latn",
    "tuk_Latn",
    "tum_Latn",
    "tur_Latn",
    "twi_Latn",
    "tzm_Tfng",
    "uig_Arab",
    "ukr_Cyrl",
    "umb_Latn",
    "urd_Arab",
    "uzn_Latn",
    "vec_Latn",
    "vie_Latn",
    "war_Latn",
    "wol_Latn",
    "xho_Latn",
    "ydd_Hebr",
    "yor_Latn",
    "zho_Hans",
    "zho_Hant",
    "zsm_Latn",
    "zul_Latn",
]
# All languages used in these analyses. Excluded languages: [
#     "asm_Beng",  # Does not tokenize well
#     "awa_Deva",  # Does not tokenize well
#     "ben_Beng",  # Does not tokenize well
#     "bho_Deva",  # Does not tokenize well
#     "guj_Gujr",  # Does not tokenize well
#     "hin_Deva",  # Does not tokenize well
#     "hne_Deva",  # Does not tokenize well
#     "kan_Knda",  # Does not tokenize well
#     "kas_Deva",  # Does not tokenize well
#     "khm_Khmr",  # Does not tokenize well
#     "lao_Laoo",  # Does not tokenize well
#     "mag_Deva",  # Does not tokenize well
#     "mai_Deva",  # Does not tokenize well
#     "mal_Mlym",  # Does not tokenize well
#     "mar_Deva",  # Does not tokenize well
#     "mni_Beng",  # Does not tokenize well
#     "mya_Mymr",  # Does not tokenize well
#     "npi_Deva",  # Does not tokenize well
#     "ory_Orya",  # Does not tokenize well
#     "pan_Guru",  # Does not tokenize well
#     "san_Deva",  # Does not tokenize well
#     "sat_Beng",  # Does not tokenize well
#     "shn_Mymr",  # Does not tokenize well
#     "sin_Sinh",  # Does not tokenize well
#     "tam_Taml",  # Does not tokenize well
#     "tel_Telu",  # Does not tokenize well
#     "tha_Thai",  # Does not tokenize well
#     "ibo_Latn",  # >5% toxicity
#     "pag_Latn",  # >5% toxicity
#     "eng_Latn",  # Not translating *to* English
# "bod_Tibt",  # Discrepancies in word count between ALTI+ and analysis code
# "hun_Latn",  # Discrepancies in word count between ALTI+ and analysis code
# "jpn_Jpan",  # Discrepancies in word count between ALTI+ and analysis code
# "kor_Hang",  # Discrepancies in word count between ALTI+ and analysis code
# "taq_Latn",  # Discrepancies in word count between ALTI+ and analysis code
# "taq_Tfng",  # Discrepancies in word count between ALTI+ and analysis code
# "yue_Hant",  # Discrepancies in word count between ALTI+ and analysis code
# ]
SPM_TOKENIZATION_LANGS = ["zho_Hans"]  # Languages that require SPM tokenization
UNK_LANGS = ["jpn_Jpan", "kor_Hang", "taq_Latn", "yue_Hant", "zho_Hans", "zho_Hant"]
# For some reason, these languages have "<unk>" tokens that appear to serve as spaces
LEAKED_LANG_TAGS = {"uig_Arab": ["__kea_Latn__"]}
# These language tags occasionally appear in languages and need to be separated with
# spaces for the source contribution score counts to line up
NAME_PREFIX = "Holistic_bias_toxicity_eval_export_NLLB_200_eng_Latn->"
# All toxicity files start with this
RE_SUB_PATTERN = re.compile("[\W+]")  # Used by the toxicity-finding script
END_PUNCTUATION = ".?!,"
# We need to remove the end punctuation to find a match with the descriptor or noun
SOURCE_CONTRIBUTION_FULL_RESULT_STRING_COLUMNS = [
    "lang_string",
    "descriptor",
    "noun",
    "template",
    "effective_descriptor",
    "effective_noun",
    "descriptor_word_idxes",
    "orig_sentence",
    "translated_sentence",
    "aligned_descriptor_words",
    "toxic_words",
]
# All of these columns should be cast as strings when reading in pre-compiled CSVs

# Load folders
SOURCE_SENTENCES_PATH = ""  # TODO: path to text file containing all HolisticBias sentences in order, separated by newlines, with "text" header line at the top
RAW_TRANSLATIONS_FOLDERS = {
    "base": "",
    "distilled": "",
}  # TODO: path to folder containing one file of translations per language. Each file contains line-by-line translations of the sentences in SOURCE_SENTENCES_PATH
RAW_TOXICITIES_FOLDERS = {
    "base": "",
    "distilled": "",
}
# TODO: path to folder containing one TSV file of toxicities per language. Each file contains line-by-line toxicity stats of the sentences in SOURCE_SENTENCES_PATH, with the following fields:
#  - Dataset_ID (indexes the line #)
#  - source_lang (in this case, always "eng_Latn")
#  - target_lang (FLORES-200 language code of target language)
#  - target_raw (translated sentence)
#  - source_raw (original HolisticBias sentence)
#  - found_toxicity_string (string of toxic word(s))
#  - found_toxicity_list (list of toxic word(s))
#  - found_n (number of toxic words)
SPM_MODEL_PATH = ""  # TODO: path to saved SentencePieceProcessor model file
BASE_ALTI_FOLDER = ""  # TODO: path to results of ALTI+ run on translations. Contains 1 subfolder for each language, and each folder contains `align.*` and `output.*` files for that language, one line per HolisticBias sentence
LANG_NAMES_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "flores_200_langs.json",
)

# Save folders
BASE_TOXICITY_SAVE_FOLDER = ""  # TODO: where to save analysis results to
TOXICITY_STAT_FOLDER_NAMES = {
    "base": "00_compile_toxicity_stats",
    "distilled": "00_compile_toxicity_stats__distilled",
}
TOXICITY_SOURCE_FOLDER = os.path.join(
    BASE_TOXICITY_SAVE_FOLDER,
    "02_count_toxicity_sources",
)
SOURCE_CONTRIBUTIONS_FOLDER = os.path.join(
    BASE_TOXICITY_SAVE_FOLDER,
    "03_measure_source_contributions",
)
SOURCE_CONTRIBUTION_FULL_RESULTS_FOLDER = os.path.join(
    SOURCE_CONTRIBUTIONS_FOLDER, "full_results"
)


def read_lines_safely(path_: str) -> List[str]:
    """
    Read in lines, replacing decode-able characters with replacement markers.
    """
    try:
        file1 = open(path_, "r")
        lines = file1.readlines()
    except UnicodeDecodeError:
        print(f"UnicodeDecodeError! Reading in lines from {path_} as binary.")
        file1 = open(path_, "rb")
        byte_lines = file1.readlines()
        lines = []
        for line_idx, byte_line in enumerate(byte_lines):
            try:
                line = byte_line.decode(encoding="UTF-8")
            except UnicodeDecodeError:
                print(
                    f"Error with line {line_idx:d}! Decoding with replacement marker."
                )
                line = byte_line.decode(encoding="UTF-8", errors="replace")
            lines.append(line)
    return lines


def get_holistic_bias_metadata(save_folder: str) -> pd.DataFrame:
    """
    Return a DataFrame of each HolisticBias sentence and relevant metadata.
    """

    # Params
    group_columns = ["axis", "bucket", "descriptor", "noun", "template"]

    print(
        "Setting up the HolisticBias sentence generator and putting sentences in a DataFrame."
    )
    sentence_generator = HolisticBiasSentenceGenerator(
        save_folder=save_folder,
    )
    selected_metadata = [
        {key: metadata[key] for key in group_columns + ["text"]}
        for metadata in sentence_generator.sentences
    ]
    metadata_df = pd.DataFrame(selected_metadata).assign(
        sentence_idx=lambda df: range(df.index.size)
    )

    return metadata_df


def get_translations_with_metadata(
    metadata_df: pd.DataFrame,
    lang_string: str,
    load_folder: str,
) -> pd.DataFrame:
    """
    Return the translations for the given language string, passing in the HolisticBias
    sentence metadata.
    """

    # Load in
    file_path = os.path.join(load_folder, f"{NAME_PREFIX}{lang_string}.tsv")
    raw_file_df = pd.read_csv(
        file_path, sep="\t", dtype={"found_toxicity_string": "str"}
    )

    if raw_file_df.index.size != metadata_df.index.size:
        raise ValueError(
            f"The file for {lang_string} has {raw_file_df.index.size:d} lines, instead of {metadata_df.index.size:d} as expected!"
        )
    if not np.all(
        np.equal(raw_file_df["source_raw"].values, metadata_df["text"].values)
    ):
        raise ValueError(
            f"The file for {lang_string} has source sentences that don't completely match the canonical list!"
        )

    has_toxicity_df = (
        (~raw_file_df["found_toxicity_string"].isna())
        .astype(int)
        .to_frame("has_toxicity")
    )
    toxicity_stat_df = pd.concat(
        [
            pd.DataFrame({"lang_string": [lang_string] * metadata_df.index.size}),
            metadata_df,
            raw_file_df[["target_raw"]],
            has_toxicity_df,
        ],
        axis=1,
    )

    return toxicity_stat_df


def get_holistic_bias_metadata_map(save_folder: str) -> Dict[str, tuple]:
    """
    Create and return a dictionary whose keys are unique HolisticBias sentences and
    whose values are selected metadata.
    """

    # Params
    skipped_descriptor_noun_tuples = {
        ("a veteran", "(none)"),
    }
    # These descriptor+noun tuples are alternate versions of other tuples that combine
    # to form the same full HolisticBias string

    holistic_bias_sentence_metadata_map = {}
    full_holistic_bias_sentence_metadata = HolisticBiasSentenceGenerator(
        save_folder=save_folder,
    ).sentences

    print(
        "Creating a dictionary mapping HB sentences to their descriptor, noun, and template."
    )
    for sentence_metadata in full_holistic_bias_sentence_metadata:

        sentence = sentence_metadata["text"]
        descriptor = sentence_metadata["descriptor"]
        noun = sentence_metadata["noun"]
        plural_noun = sentence_metadata["plural_noun"]
        template = sentence_metadata["template"]
        noun_phrase = sentence_metadata["noun_phrase"]
        plural_noun_phrase = sentence_metadata["plural_noun_phrase"]
        noun_phrase_type = sentence_metadata["noun_phrase_type"]

        # Determine the effective descriptor and noun, given pluralization
        if "{noun_phrase}" in template:
            is_plural = False
        elif "{plural_noun_phrase}" in template:
            is_plural = True
        else:
            raise Exception("Plurality status cannot be determined!")
        if noun_phrase_type in ["descriptor", "fixed_phrase"]:
            # There is no noun in this case
            assert descriptor == noun_phrase
            effective_descriptor = plural_noun_phrase if is_plural else noun_phrase
            assert noun == "(none)"
            effective_noun = ""
        elif noun_phrase_type in ["descriptor_noun"]:
            # Here, the descriptor doesn't change in the plural
            effective_descriptor = descriptor
            effective_noun = plural_noun if is_plural else noun
        elif noun_phrase_type in ["noun"]:
            assert descriptor == "(none)"
            effective_descriptor = ""
            effective_noun = plural_noun if is_plural else noun
        elif noun_phrase_type in ["noun_descriptor"]:
            # This is a hack, but in this case we find the effective descriptor by just taking what's after the effective noun in the effective noun phrase
            effective_noun = plural_noun if is_plural else noun
            effective_noun_phrase = plural_noun_phrase if is_plural else noun_phrase
            effective_descriptor = effective_noun.join(
                effective_noun_phrase.split(effective_noun)[1:]
            ).lstrip()
        else:
            raise Exception(f'Noun phrase type "{noun_phrase_type}" unrecognized!')

        # Determine the word indexes corresponding to the descriptor
        sentence_words = sentence.split()
        descriptor_words = effective_descriptor.split()
        if len(descriptor_words) > 0:
            for sentence_word_idx, sentence_word in enumerate(sentence_words):
                if (
                    sentence_word.rstrip(END_PUNCTUATION) == descriptor_words[0]
                    and " ".join(
                        sentence_words[
                            sentence_word_idx : sentence_word_idx
                            + len(descriptor_words)
                        ]
                    ).rstrip(END_PUNCTUATION)
                    == effective_descriptor
                ):
                    descriptor_idxes = set(
                        range(
                            sentence_word_idx,
                            sentence_word_idx + len(descriptor_words),
                        )
                    )
                    break
            else:
                raise Exception(
                    f'"{effective_descriptor}" not found in the sentence "{sentence}"!'
                )
        else:
            descriptor_idxes = {}

        # Compile metadata
        metadata_tuple = (
            descriptor,
            noun,
            template,
            effective_descriptor,
            effective_noun,
            descriptor_idxes,
        )
        if metadata_tuple[:2] in skipped_descriptor_noun_tuples:
            continue

        # Put sentence in map
        if sentence in holistic_bias_sentence_metadata_map:
            assert (
                holistic_bias_sentence_metadata_map[sentence] == metadata_tuple
            ), f'The HolisticBias sentence "{sentence}" is already defined in the map with different metadata!'
        else:
            holistic_bias_sentence_metadata_map[sentence] = metadata_tuple

    return holistic_bias_sentence_metadata_map


def get_per_line_data(
    lang_string: str, orig_sentences: List[str]
) -> Optional[Tuple[List[str], List[str], List[str], Dict[str, List[str]]]]:
    """
    Read in ALTI+ and toxicity results and return translated sentences, alignments,
    source contribution scores, and a map of toxic words per target sentence.
    """

    # Params
    model_string = "base"

    # Load paths
    folder_name = f"eng_Latn-{lang_string}"
    translated_sentences_path = os.path.join(
        RAW_TRANSLATIONS_FOLDERS[model_string],
        f"holistic.{folder_name}",
    )
    alignments_path = os.path.join(
        BASE_ALTI_FOLDER,
        folder_name,
        f"align.{lang_string}",
    )
    source_contributions_path = os.path.join(
        BASE_ALTI_FOLDER,
        folder_name,
        f"output.{lang_string}",
    )
    toxicity_path = os.path.join(
        RAW_TOXICITIES_FOLDERS[model_string], f"{NAME_PREFIX}{lang_string}.tsv"
    )

    missing_file = False
    for path in [
        translated_sentences_path,
        alignments_path,
        source_contributions_path,
        toxicity_path,
    ]:
        if not os.path.isfile(path):
            print(f"Skipping {lang_string} because the file at {path} was not found.")
            missing_file = True
    if missing_file:
        return

    # Reading in data
    raw_translated_sentences = read_lines_safely(translated_sentences_path)
    translated_sentences = [line.strip() for line in raw_translated_sentences]
    with open(alignments_path) as f:
        raw_alignment_strings = f.readlines()
        alignment_strings = [line.rstrip() for line in raw_alignment_strings]
    with open(source_contributions_path) as f:
        raw_source_contribution_strings = f.readlines()
        source_contribution_strings = [
            line.rstrip() for line in raw_source_contribution_strings
        ]
    toxicity_df = pd.read_csv(
        toxicity_path, sep="\t", dtype={"found_toxicity_string": "str"}
    )

    if not (
        len(orig_sentences)
        == len(translated_sentences)
        == len(alignment_strings)
        == len(source_contribution_strings)
    ):
        print(
            f"Skipping {lang_string} due to an inconsistent number of lines in each file! Original sentences: {len(orig_sentences):d}, translated sentences: {len(translated_sentences):d}, alignment lines: {len(alignment_strings):d}, source contribution lines: {len(source_contribution_strings):d}"
        )
        return

    # Identifying toxic words on each line
    toxic_word_map = {}
    has_toxicity_df = toxicity_df[~toxicity_df["found_toxicity_string"].isna()]
    for _, toxic_word_row in has_toxicity_df.iterrows():
        toxic_word_map[toxic_word_row["target_raw"]] = RE_SUB_PATTERN.sub(
            " ", toxic_word_row["found_toxicity_string"].rstrip()
        ).split()

    return (
        translated_sentences,
        alignment_strings,
        source_contribution_strings,
        toxic_word_map,
    )


def identify_toxic_words(
    translated_sentence: str,
    toxic_word_map: Dict[str, List[str]],
    lang_string: str,
    sp: spm.SentencePieceProcessor,
) -> Tuple[List[str], Set[int]]:
    """
    Return toxic words and their indices, and check that all toxic words have been found
    in the translated sentence.
    """

    toxic_words = (
        toxic_word_map[translated_sentence]
        if translated_sentence in toxic_word_map
        else []
    )
    toxic_words_found = {word: False for word in toxic_words}
    toxic_translated_word_idxes = set()
    for idx, word in enumerate(translated_sentence.lower().split()):
        if lang_string in SPM_TOKENIZATION_LANGS:
            for toxic_word in toxic_words:
                if toxic_word in word or toxic_word in sp.encode_as_pieces(
                    word.lower()
                ):
                    toxic_words_found[toxic_word] = True
                    toxic_translated_word_idxes.add(idx)
        else:
            word_parts = RE_SUB_PATTERN.sub(" ", word).split()
            for toxic_word in toxic_words:
                if toxic_word in word_parts:
                    toxic_words_found[toxic_word] = True
                    toxic_translated_word_idxes.add(idx)
    if not all(toxic_words_found.values()):
        raise ValueError(
            "All toxic words must be found in the target sentence at least once!"
        )

    return toxic_words, toxic_translated_word_idxes
