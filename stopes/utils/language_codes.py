# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
import typing as tp
from collections import defaultdict

import numpy as np
import pandas as pd
from langcodes import Language, LanguageTagError
from sklearn.neighbors import BallTree
from tqdm.auto import tqdm

from stopes.utils.web import cached_file_download

logger = logging.getLogger(__name__)


# This is a copy of the languoid file from Glottolog (https://glottolog.org/meta/downloads), release 4.8.
# It is distributed by Glottolog under the CC-BY-4.0 license: https://creativecommons.org/licenses/by/4.0/
GLOTTOLOG_DATA_URL = (
    "https://dl.fbaipublicfiles.com/nllb/languages/glottolog_languoid.csv"
)


def parse_language_code(lang: str) -> Language:
    """Convert a language code (in any format) to a Language object, bypassing some formatting errors."""
    try:
        lang_object = Language.get(lang)
        return lang_object
    except LanguageTagError as error:
        # Testing the hi_IN_rom case
        match = re.match("(?P<langtag>[a-z]{2})_(?P<geotag>[A-Z]{2})_rom", lang)
        if match:
            langtag = match.groupdict()["langtag"]
            geotag = match.groupdict()["geotag"]
            lang_object = Language.get(f"{langtag}_Latn_{geotag}")
            return lang_object
        raise error


def language_code_to_short_code(
    orig_code: str, try_replacing_with_macro: bool = False
) -> str:
    """
    Convert a language code (in any format) to its alpha-2 code, or, if it does not exist, to alpha-3.
    If `try_replacing_with_macro` is set and the language does not have an alpha-2 code, but its macrolanguage does,
    then the macrolanguage code is used (for example, Khalkha Mongolian `khk` may be "rounded" to just Mongolian `mn`).
    """
    language: Language = parse_language_code(orig_code)
    new_code = language.language

    if not isinstance(new_code, str):
        logger.warning(
            f"The code {orig_code} hasn't been matched, so language_code_to_short_code is returning it."
        )
        return orig_code

    # Special case: `langcodes` package insists on renaming Tagalog to Filipino, but we don't want that rename.
    # Filipino is a standardized version of Tagalog, so all Filipino is Tagalog, but not all Tagalog is Filipino.
    if new_code == "fil" and orig_code.split("_")[0] in {"tgl", "tl"}:
        new_code = "tl"

    if try_replacing_with_macro and len(new_code) == 3 and new_code in ISO_MICRO2MACRO:
        code_macro = parse_language_code(ISO_MICRO2MACRO[new_code]).language
        if isinstance(code_macro, str) and len(code_macro) == 2:
            logger.info(
                f"Replacing an alpha-3 code `{new_code}` (originally `{orig_code}`) with a macro-language alpha-2 code `{code_macro}`."
            )
            new_code = code_macro

    return new_code


class LanguageMatcher:
    """
    A class that matches the given language to the nearest neighbour from the given set of the target languages.
    The proximity is determined by a cascade of criteria; see the `match` method docstring.
    To work with it, please install the `[mono]` extra dependencies of Stopes.
    Usage example:
    ```
    from stopes.utils.language_codes import SONAR_LANGS, LanguageMatcher
    matcher = LanguageMatcher()
    matcher.set_target_langs(SONAR_LANGS)
    print(matcher.match("es"))  # spa_Latn
    print(matcher.match("en-US"))  # eng_Latn
    print(matcher.match("ar"))  # arb_Arab
    print(matcher.match("zh_TW"))  # zho_Hant
    print(matcher.match("no_XX"))  # nob_Latn
    print(matcher.match("ber"))  # tzm_Tfng
    print(matcher.match("foo"))  # None
    print(matcher.match("ns"))  # None
    print(matcher.match("ab"))  # kat_Geor
    ```
    """

    def __init__(self, glottolog_data_path: str = "auto"):
        if glottolog_data_path == "auto":
            glottolog_data_path = str(
                cached_file_download(GLOTTOLOG_DATA_URL, "glottolog_languoid.csv")
            )

        self.glottolog_data: pd.DataFrame = pd.read_csv(glottolog_data_path)
        # Checking that the table contains information about genetic and geographic relations
        expected_columns = {"id", "parent_id", "iso639P3code", "latitude", "longitude"}
        assert not expected_columns.difference(self.glottolog_data.columns)

        # Imputting codes for some weird languages, such as "Berber"
        for short_code, glottolog_code in MISSING_P3_CODES_IN_GLOTTOLOG.items():
            mask = (self.glottolog_data["id"] == glottolog_code) & (
                self.glottolog_data["iso639P3code"].isnull()
            )
            if sum(mask) > 0:
                self.glottolog_data.loc[mask, "iso639P3code"] = short_code

        self.code2row: tp.Dict[str, int] = (
            self.glottolog_data.iso639P3code.dropna()
            .reset_index()
            .set_index("iso639P3code")["index"]
            .to_dict()
        )
        self.fullid2row: tp.Dict[str, int] = (
            self.glottolog_data.id.dropna()
            .reset_index()
            .set_index("id")["index"]
            .to_dict()
        )
        self.row2children: tp.DefaultDict[int, tp.Set[int]] = defaultdict(set)
        for i, row in self.glottolog_data.iterrows():
            if not pd.isna(row["parent_id"]):
                self.row2children[self.fullid2row[row["parent_id"]]].add(row.name)

        # The attributes below will be set when setting the target languages
        self.target_langs_set: tp.Set[str] = set()
        self.target_langs_3_to_all: tp.Dict[str, tp.Set[str]] = {}
        self.target_langs_pop: tp.Dict[str, int] = {}
        self.matched_data: tp.Optional[pd.DataFrame] = None
        self.coordinate_tree: tp.Optional[BallTree] = None

    def strip_script(self, language: str) -> str:
        """Remove the script information from NLLB-styled language code: e.g. `eng_Latn` => `eng`."""
        message = f"Target languages should be formatted as `eng_Latn` or `eng`; got {language} instead."
        assert re.match("^[a-z]{3}(_[a-zA-Z]{4})?$", language), message
        return language[:3]

    def set_target_langs(self, target_langs_list: tp.List[str]):
        """
        Args:
            target_langs_list: a list of languages in the 'eng_Latn' form
        """
        self.target_langs_set = set(target_langs_list)
        target_langs_3_to_all = defaultdict(set)
        for lang in target_langs_list:
            target_langs_3_to_all[self.strip_script(lang)].add(lang)
        self.target_langs_3_to_all = dict(target_langs_3_to_all)

        self.target_langs_pop = {
            self.strip_script(lang): max(
                Language.get(self.strip_script(lang)).speaking_population(),
                POPULATIONS.get(lang, 1),
            )
            for lang in sorted(target_langs_list)
        }
        self.glottolog_data[MATCH_COLUMN] = self.glottolog_data["iso639P3code"].apply(
            lambda x: x if x in self.target_langs_3_to_all else None
        )
        self.glottolog_data[MATCH_POP_COLUMN] = self.glottolog_data.iso639P3code.apply(
            self.target_langs_pop.get
        )
        for lang_code, population in tqdm(
            sorted(self.target_langs_pop.items(), key=lambda x: -x[1])
        ):
            code3 = lang_code
            if code3 not in self.code2row:
                code3 = self.get_micro_language(code3) or code3
            if code3 not in self.code2row:
                logger.warning(f"Could not find the target language: {code3}")
            row_id = self.code2row[code3]
            row = self.glottolog_data.loc[row_id]
            while True:
                if pd.isna(row.parent_id):
                    break
                row = self.glottolog_data.loc[self.fullid2row[row.parent_id]]
                if (
                    not pd.isna(row[MATCH_POP_COLUMN])
                    and row[MATCH_POP_COLUMN] > population
                ):
                    break
                self.glottolog_data.loc[row.name, MATCH_POP_COLUMN] = population
                self.glottolog_data.loc[row.name, MATCH_COLUMN] = lang_code
        self.matched_data = self.glottolog_data.dropna(
            subset=["latitude", "longitude", MATCH_COLUMN]
        )
        self.coordinate_tree = BallTree(
            np.radians(self.matched_data[["latitude", "longitude"]]), metric="haversine"
        )

    def match(self, orig_code: str) -> tp.Optional[str]:
        """
        For the input language code, find the most similar language in the set of target codes.
        Try consecutively:
        - Exact matching
        - Exact matching after formatting the language with `langcodes` package
        - Matching using the mapping of individual and macro languages
        - Fuzzy matching to nearest genetic relative, using Gloggolog language genealogy tree
        - Fuzzy matching to nearest georaphic neigbour, using Gloggolog language coordinates
        """
        if orig_code in self.target_langs_set:
            return orig_code
        lang_obj_raw = parse_language_code(orig_code)
        try:
            code3 = lang_obj_raw.to_alpha3()
        except LookupError:
            logger.warning(
                f"Could not parse the language code '{orig_code}'; matching it to none."
            )
            return None
        # some languages are parsed weirdly, e.g. prs is mapped to fas-AF, but fas is mapped back to pes
        # TODO: take the territory into account, to fix this problem
        result = self.choose_script_if_matched(code3, orig_code=orig_code, strict=True)
        if result:
            return result
        result = self.find_gen_substitute(orig_code)
        logger.warning(f"Fuzzy lookup for language {orig_code} => {result}")
        return result

    def choose_script_if_matched(
        self, code3: str, orig_code: tp.Optional[str] = None, strict=False
    ) -> tp.Optional[str]:
        """
        If the language is in the set of target languoids,
        choose the languoid with the maching script, and return it.
        """
        candidates = self.target_langs_3_to_all.get(code3, set())

        # If there are several scripts, try choosing the one
        if len(candidates) > 1:
            scripted = parse_language_code(orig_code or code3)
            self.assume_script_(scripted)
            candidates_new = {c for c in candidates if c[4:] == scripted.script}
            if len(candidates_new) == 1:
                candidates = candidates_new
            elif not strict:
                # TODO: try choosing a script in a less arbitrary way
                logger.warning(
                    f"For {orig_code}, found several scripts: {candidates} ({len(candidates_new)} matching)"
                )
        if len(candidates) == 1 or len(candidates) > 1 and not strict:
            return list(candidates)[0]
        # if no scripts are matched, returning None; the later steps will do fuzzier search
        return None

    def find_geo_substitute(self, orig_code, fallback=True, not_the_same=False):
        """
        For the given language code, find its nearest geographic neighbour (as represented by Glottolog coordinates)
        that belongs to the set of the target languages.
        If nothing found, fall back to matching by genetic proximity.
        """
        code3 = self.standardize_code(orig_code)
        row_id = self.code2row[code3]
        row = self.glottolog_data.loc[row_id]
        if not pd.isnull(row[MATCH_COLUMN]):
            if row[MATCH_COLUMN] != orig_code or not not_the_same:
                return row[MATCH_COLUMN]
        if pd.isna(row.latitude) or pd.isna(row.longitude):
            if fallback:
                return self.find_gen_substitute(orig_code, fallback=False)
            return

        assert (
            self.coordinate_tree is not None and self.matched_data is not None
        ), "Please set the target language codes before the matching."
        distances, indices = self.coordinate_tree.query(
            np.radians([[row.latitude, row.longitude]]), k=50
        )
        neighbours = self.matched_data.iloc[indices[0]]
        if not_the_same:
            neighbours = neighbours[neighbours[MATCH_COLUMN] != orig_code]
        if neighbours.shape[0] > 0:
            matched_language = neighbours[MATCH_COLUMN].iloc[0]
            return self.choose_script_if_matched(matched_language, strict=False)

        if fallback:
            return self.find_gen_substitute(orig_code, fallback=False)

    def find_gen_substitute(
        self, orig_code, verbose=False, fallback=True, not_the_same=False
    ):
        """
        For the given language code, find its nearest neighbour in the genetic tree (as represented by Glottolog)
        that belongs to the set of the target languages.
        In case of ambiguity, return the neighbour with highest population.
        If nothing found, fall back to matching by geographic proximity.
        """
        code3 = self.standardize_code(orig_code)
        if code3 not in self.code2row:
            logger.warning(f"Code `{code3}` not found in Glottolog!")
            return
        row_id = self.code2row[code3]
        row = self.glottolog_data.loc[row_id]
        if not pd.isnull(row[MATCH_COLUMN]):
            if row[MATCH_COLUMN] != orig_code or not not_the_same:
                return self.choose_script_if_matched(row[MATCH_COLUMN], strict=False)
        while True:
            if verbose:
                print(f"{row['name']} : looking for genetic neighbours")
            children = self.glottolog_data.loc[
                sorted(self.row2children.get(row.name, set()))
            ]
            fltr = children[MATCH_COLUMN].notnull()
            if not_the_same:
                fltr = fltr & (children[MATCH_COLUMN] != orig_code)
            children = children[fltr]
            if children.shape[0] > 0:
                largest_child_lang = children[MATCH_COLUMN][
                    children[MATCH_POP_COLUMN].idxmax()
                ]
                return self.choose_script_if_matched(largest_child_lang, strict=False)
            if pd.isna(row.parent_id):
                if verbose:
                    print("found nothing genealogically, falling back to geography")
                if fallback:
                    return self.find_geo_substitute(
                        orig_code, fallback=False, not_the_same=not_the_same
                    )
            row = self.glottolog_data.loc[self.fullid2row[row.parent_id]]

    def standardize_code(self, lang: str) -> str:
        """
        Try to standardize a language code to match one in the glottolog data, by:
        1. Formatting it with langcode package
        2. Mapping a macrolanguage to its arbitrary individual language.
        """
        if lang in self.code2row:
            return lang
        lang_object = parse_language_code(lang)
        code3 = lang_object.to_alpha3()
        if code3 not in self.code2row:
            # Trying to match the macrolanguage to its variety
            child = self.get_micro_language(code3)
            if child:
                return child
        return code3

    def get_micro_language(self, code3: str) -> tp.Optional[str]:
        """If code is a macro language, return an individual language code of its first (usually arbitrary) child."""
        if code3 in ISO_MACRO2MICRO:
            for child in ISO_MACRO2MICRO[code3]:
                if child in self.code2row:
                    logger.warning(
                        f"Replacing the macrolanguage {code3} with its arbitrary sub-language: {child}"
                    )
                    return child
        # If no micro-language was found, we do nothing.
        return None

    def assume_script_(self, lang: Language):
        """Modify a Language object, by assiging a script to it, if possible"""
        lang.assume_script()
        if lang.script is not None:
            return

        # For Chinese, assuming Mandarin with simplified script in Mainland China, and with traditional one in Taiwan
        if lang.language in {"zh", "cmn"}:
            if lang.territory in {"CN", None, "XX"}:
                lang.script = "Hans"
            if lang.territory in {"TW"}:
                lang.script = "Hant"


# The list of ~200 languages supported by SONAR sentence encoder
SONAR_LANGS = [
    "ace_Arab",
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
    "asm_Beng",
    "ast_Latn",
    "awa_Deva",
    "ayr_Latn",
    "azb_Arab",
    "azj_Latn",
    "bak_Cyrl",
    "bam_Latn",
    "ban_Latn",
    "bel_Cyrl",
    "bem_Latn",
    "ben_Beng",
    "bho_Deva",
    "bjn_Arab",
    "bjn_Latn",
    "bod_Tibt",
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
    "eng_Latn",
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
    "guj_Gujr",
    "hat_Latn",
    "hau_Latn",
    "heb_Hebr",
    "hin_Deva",
    "hne_Deva",
    "hrv_Latn",
    "hun_Latn",
    "hye_Armn",
    "ibo_Latn",
    "ilo_Latn",
    "ind_Latn",
    "isl_Latn",
    "ita_Latn",
    "jav_Latn",
    "jpn_Jpan",
    "kab_Latn",
    "kac_Latn",
    "kam_Latn",
    "kan_Knda",
    "kas_Arab",
    "kas_Deva",
    "kat_Geor",
    "kaz_Cyrl",
    "kbp_Latn",
    "kea_Latn",
    "khk_Cyrl",
    "khm_Khmr",
    "kik_Latn",
    "kin_Latn",
    "kir_Cyrl",
    "kmb_Latn",
    "kmr_Latn",
    "knc_Arab",
    "knc_Latn",
    "kon_Latn",
    "kor_Hang",
    "lao_Laoo",
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
    "mag_Deva",
    "mai_Deva",
    "mal_Mlym",
    "mar_Deva",
    "min_Latn",
    "mkd_Cyrl",
    "mlt_Latn",
    "mni_Beng",
    "mos_Latn",
    "mri_Latn",
    "mya_Mymr",
    "nld_Latn",
    "nno_Latn",
    "nob_Latn",
    "npi_Deva",
    "nso_Latn",
    "nus_Latn",
    "nya_Latn",
    "oci_Latn",
    "ory_Orya",
    "pag_Latn",
    "pan_Guru",
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
    "san_Deva",
    "sat_Beng",
    "scn_Latn",
    "shn_Mymr",
    "sin_Sinh",
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
    "tam_Taml",
    "taq_Latn",
    "taq_Tfng",
    "tat_Cyrl",
    "tel_Telu",
    "tgk_Cyrl",
    "tgl_Latn",
    "tha_Thai",
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
    "yue_Hant",
    "zho_Hans",
    "zho_Hant",
    "zsm_Latn",
    "zul_Latn",
]

# This mapping is extracted from https://iso639-3.sil.org/code_tables/download_tables
# For some macrolanguages, I put at the first position its preferred individual language
# (usually, the most widely used one, or the one with an official status)
# The rest is ordered alphabetically.
ISO_MACRO2MICRO = {
    "aka": ["fat", "twi"],
    "ara": [
        "arb",  # Arabic => Modern Standard Arabic
        "aao",
        "abh",
        "abv",
        "acm",
        "acq",
        "acw",
        "acx",
        "acy",
        "adf",
        "aeb",
        "aec",
        "afb",
        "ajp",
        "apc",
        "apd",
        "arq",
        "ars",
        "ary",
        "arz",
        "auz",
        "avl",
        "ayh",
        "ayl",
        "ayn",
        "ayp",
        "bbz",
        "pga",
        "shu",
        "ssh",
    ],
    "aym": ["ayc", "ayr"],
    "aze": [
        "azj",
        "azb",
    ],  # putting North Azerbaijani first, because it has an official status
    "bal": ["bcc", "bgn", "bgp"],
    "bik": ["bcl", "bhk", "bln", "bto", "cts", "fbl", "lbl", "rbl", "ubl"],
    "bnc": ["ebk", "lbk", "obk", "rbk", "vbk"],
    "bua": ["bxm", "bxr", "bxu"],
    "chm": ["mhr", "mrj"],
    "cre": ["crj", "crk", "crl", "crm", "csw", "cwd"],
    "del": ["umu", "unm"],
    "den": ["scs", "xsl"],
    "din": ["dib", "dik", "dip", "diw", "dks"],
    "doi": ["dgo", "xnr"],
    "est": ["ekk", "vro"],
    "fas": ["pes", "prs"],
    "ful": ["ffm", "fub", "fuc", "fue", "fuf", "fuh", "fui", "fuq", "fuv"],
    "gba": ["bdt", "gbp", "gbq", "gmm", "gso", "gya", "mdo"],
    "gon": ["esg", "ggo", "gno", "wsg"],
    "grb": ["gbo", "gec", "grj", "grv", "gry"],
    "grn": ["gug", "gnw", "gui", "gun", "nhd"],
    "hai": ["hax", "hdn"],
    "hbs": ["bos", "cnr", "hrv", "srp"],
    "hmn": [
        "blu",
        "cqd",
        "hea",
        "hma",
        "hmc",
        "hmd",
        "hme",
        "hmg",
        "hmh",
        "hmi",
        "hmj",
        "hml",
        "hmm",
        "hmp",
        "hmq",
        "hms",
        "hmw",
        "hmy",
        "hmz",
        "hnj",
        "hrm",
        "huj",
        "mmr",
        "muq",
        "mww",
        "sfm",
    ],
    "iku": ["ike", "ikt"],
    "ipk": ["esi", "esk"],
    "jrb": ["ajt", "aju", "jye", "yhd", "yud"],
    "kau": ["kby", "knc", "krt"],
    "kln": ["enb", "eyo", "niq", "oki", "pko", "sgc", "spy", "tec", "tuy"],
    "kok": ["gom", "knn"],
    "kom": ["koi", "kpv"],
    "kon": ["kng", "kwy", "ldi"],
    "kpe": ["gkp", "xpe"],
    "kur": ["ckb", "kmr", "sdh"],
    "lah": ["hnd", "hno", "jat", "phr", "pmu", "pnb", "skr", "xhe"],
    "lav": [
        "lvs",  # Putting Standard Latvian first
        "ltg",
    ],
    "luy": [
        "bxk",
        "ida",
        "lkb",
        "lko",
        "lks",
        "lri",
        "lrm",
        "lsm",
        "lto",
        "lts",
        "lwg",
        "nle",
        "nyd",
        "rag",
    ],
    "man": ["emk", "mku", "mlq", "mnk", "msc", "mwk", "myq"],
    "mlg": [
        "bhr",
        "bjq",
        "bmm",
        "bzc",
        "msh",
        "plt",
        "skg",
        "tdx",
        "tkg",
        "txy",
        "xmv",
        "xmw",
    ],
    "mon": ["khk", "mvf"],
    "msa": [
        "zsm",  # Malay (macrolanguage) => Standard Malay (individual), a.k.a. Malaysian Malay
        "bjn",
        "btj",
        "bve",
        "bvu",
        "coa",
        "dup",
        "hji",
        "ind",
        "jak",
        "jax",
        "kvb",
        "kvr",
        "kxd",
        "lce",
        "lcf",
        "liw",
        "max",
        "meo",
        "mfa",
        "mfb",
        "min",
        "mly",
        "mqg",
        "msi",
        "mui",
        "orn",
        "ors",
        "pel",
        "pse",
        "tmw",
        "urk",
        "vkk",
        "vkt",
        "xmm",
        "zlm",
        "zmi",
    ],
    "mwr": ["dhd", "mtr", "mve", "rwr", "swv", "wry"],
    "nep": ["dty", "npi"],
    "nor": ["nno", "nob"],
    "oji": ["ciw", "ojb", "ojc", "ojg", "ojs", "ojw", "otw"],
    "ori": ["ory", "spv"],
    "orm": ["gax", "gaz", "hae", "orc"],
    "pus": ["pbt", "pbu", "pst"],
    "que": [
        "cqu",
        "qub",
        "qud",
        "quf",
        "qug",
        "quh",
        "quk",
        "qul",
        "qup",
        "qur",
        "qus",
        "quw",
        "qux",
        "quy",
        "quz",
        "qva",
        "qvc",
        "qve",
        "qvh",
        "qvi",
        "qvj",
        "qvl",
        "qvm",
        "qvn",
        "qvo",
        "qvp",
        "qvs",
        "qvw",
        "qvz",
        "qwa",
        "qwc",
        "qwh",
        "qws",
        "qxa",
        "qxc",
        "qxh",
        "qxl",
        "qxn",
        "qxo",
        "qxp",
        "qxr",
        "qxt",
        "qxu",
        "qxw",
    ],
    "raj": ["bgq", "gda", "gju", "hoj", "mup", "wbr"],
    "rom": ["rmc", "rmf", "rml", "rmn", "rmo", "rmw", "rmy"],
    "san": ["cls", "vsn"],
    "sqi": ["aae", "aat", "aln", "als"],
    "srd": ["sdc", "sdn", "src", "sro"],
    "swa": ["swc", "swh"],
    "syr": ["aii", "cld"],
    "tmh": ["taq", "thv", "thz", "ttq"],
    "uzb": ["uzn", "uzs"],
    "yid": ["ydd", "yih"],
    "zap": [
        "zaa",
        "zab",
        "zac",
        "zad",
        "zae",
        "zaf",
        "zai",
        "zam",
        "zao",
        "zaq",
        "zar",
        "zas",
        "zat",
        "zav",
        "zaw",
        "zax",
        "zca",
        "zcd",
        "zoo",
        "zpa",
        "zpb",
        "zpc",
        "zpd",
        "zpe",
        "zpf",
        "zpg",
        "zph",
        "zpi",
        "zpj",
        "zpk",
        "zpl",
        "zpm",
        "zpn",
        "zpo",
        "zpp",
        "zpq",
        "zpr",
        "zps",
        "zpt",
        "zpu",
        "zpv",
        "zpw",
        "zpx",
        "zpy",
        "zpz",
        "zsr",
        "ztc",
        "zte",
        "ztg",
        "ztl",
        "ztm",
        "ztn",
        "ztp",
        "ztq",
        "zts",
        "ztt",
        "ztu",
        "ztx",
        "zty",
    ],
    "zha": [
        "ccx",
        "ccy",
        "zch",
        "zeh",
        "zgb",
        "zgm",
        "zgn",
        "zhd",
        "zhn",
        "zlj",
        "zln",
        "zlq",
        "zqe",
        "zyb",
        "zyg",
        "zyj",
        "zyn",
        "zzj",
    ],
    "zho": [
        "cmn",  # Putting Mandarin as "Standard Chinese" first
        "cdo",
        "cjy",
        "cnp",
        "cpx",
        "csp",
        "czh",
        "czo",
        "gan",
        "hak",
        "hsn",
        "lzh",
        "mnp",
        "nan",
        "wuu",
        "yue",
    ],
    "zza": ["diq", "kiu"],
}


ISO_MICRO2MACRO = {
    micro: macro for macro, micros in ISO_MACRO2MICRO.items() for micro in micros
}

# This mapping augments the one produced by Langcodes.
# It affects the choice between children languages when finding genetic neighbours
POPULATIONS = dict(
    acm_Arab=28_000_000,  # https://en.wikipedia.org/wiki/Mesopotamian_Arabic
    acq_Arab=12_000_000,  # https://en.wikipedia.org/wiki/Ta%CA%BDizzi-Adeni_Arabic divided by 2
    ajp_Arab=27_000_000,  # https://en.wikipedia.org/wiki/South_Levantine_Arabic
    als_Latn=1_800_000,  # https://en.wikipedia.org/wiki/Tosk_Albanian
    apc_Arab=27_000_000,  # https://en.wikipedia.org/wiki/North_Levantine_Arabic divided by 2
    arb_Arab=270_000_000,  # https://en.wikipedia.org/wiki/Modern_Standard_Arabic
    ayr_Latn=1_700_000,  # https://en.wikipedia.org/wiki/Aymara_language [choose central-southern?]
    azb_Arab=13_000_000,  # https://en.wikipedia.org/wiki/Azerbaijani_language#South_Azerbaijani
    azj_Latn=9_000_000,  # https://en.wikipedia.org/wiki/Azerbaijani_language#North_Azerbaijani
    cjk_Latn=2_500_000,  # https://en.wikipedia.org/wiki/Chokwe_language
    dik_Latn=4_200_000,  # https://en.wikipedia.org/wiki/Dinka_language [choose southwestern?]
    gaz_Latn=45_500_000,  # https://en.wikipedia.org/wiki/Oromo_language [choose western central?]
    kbp_Latn=1_000_000,  # https://en.wikipedia.org/wiki/Kabiye_language
    khk_Cyrl=3_000_000,  # https://en.wikipedia.org/wiki/Khalkha_Mongolian
    kmr_Latn=16_000_000,  # https://en.wikipedia.org/wiki/Kurmanji
    knc_Arab=8_450_000,  # https://en.wikipedia.org/wiki/Central_Kanuri
    knc_Latn=8_450_000,  # https://en.wikipedia.org/wiki/Central_Kanuri
    lus_Latn=1_000_000,  # https://en.wikipedia.org/wiki/Mizo_language
    lvs_Latn=1_300_000,  # https://en.wikipedia.org/wiki/Latvian_language minus Latgalian
    npi_Deva=19_000_000,  # https://en.wikipedia.org/wiki/Nepali_language
    ory_Orya=35_000_000,  # https://en.wikipedia.org/wiki/Odia_language
    pbt_Arab=16_000_000,  # https://en.wikipedia.org/wiki/Southern_Pashto
    pes_Arab=57_000_000,  # https://en.wikipedia.org/wiki/Iranian_Persian
    plt_Latn=10_893_000,  # https://en.wikipedia.org/wiki/Malagasy_language, chose Plateau
    quy_Latn=918_000,  # https://en.wikipedia.org/wiki/Ayacucho_Quechua
    swh_Latn=18_000_000,  # https://en.wikipedia.org/wiki/Swahili_language
    taq_Latn=900_000,  # https://en.wikipedia.org/wiki/Tamasheq_language
    taq_Tfng=900_000,  # https://en.wikipedia.org/wiki/Tamasheq_language
    twi_Latn=16_000_000,  # https://en.wikipedia.org/wiki/Twi
    uzn_Latn=28_000_000,  # https://en.wikipedia.org/wiki/Uzbek_language
    ydd_Hebr=600_000,  # https://en.wikipedia.org/wiki/Yiddish_dialects
    zsm_Latn=33_000_000,  # https://en.wikipedia.org/wiki/Malaysian_Malay
)

MATCH_COLUMN = "matched_lang_code3"
MATCH_POP_COLUMN = "matched_lang_population"

# Some language groups, like "Berber", are present in Glottolog data, but don't have an iso639-3 code.
MISSING_P3_CODES_IN_GLOTTOLOG = {
    "ber": "berb1260",
}
