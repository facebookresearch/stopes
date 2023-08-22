# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import json
import logging
import typing as tp
from pathlib import Path

import fairseq
import torch

from stopes.eval.alti.wrappers.multilingual_transformer_wrapper import (
    FairseqMultilingualTransformerHub,
)
from stopes.eval.alti.wrappers.transformer_wrapper import FairseqTransformerHub

from .alti_metrics_utils import compute_alti_scores_for_batch
from .file_utils import join_lists_of_dicts, read_tsv, select_columns, write_tsv

logger = logging.getLogger(__name__)

NLLB200_LANGS = "ace_Arab,ace_Latn,acm_Arab,acq_Arab,aeb_Arab,afr_Latn,ajp_Arab,aka_Latn,amh_Ethi,apc_Arab,arb_Arab,ars_Arab,ary_Arab,arz_Arab,asm_Beng,ast_Latn,awa_Deva,ayr_Latn,azb_Arab,azj_Latn,bak_Cyrl,bam_Latn,ban_Latn,bel_Cyrl,bem_Latn,ben_Beng,bho_Deva,bjn_Arab,bjn_Latn,bod_Tibt,bos_Latn,bug_Latn,bul_Cyrl,cat_Latn,ceb_Latn,ces_Latn,cjk_Latn,ckb_Arab,crh_Latn,cym_Latn,dan_Latn,deu_Latn,dik_Latn,dyu_Latn,dzo_Tibt,ell_Grek,eng_Latn,epo_Latn,est_Latn,eus_Latn,ewe_Latn,fao_Latn,pes_Arab,fij_Latn,fin_Latn,fon_Latn,fra_Latn,fur_Latn,fuv_Latn,gla_Latn,gle_Latn,glg_Latn,grn_Latn,guj_Gujr,hat_Latn,hau_Latn,heb_Hebr,hin_Deva,hne_Deva,hrv_Latn,hun_Latn,hye_Armn,ibo_Latn,ilo_Latn,ind_Latn,isl_Latn,ita_Latn,jav_Latn,jpn_Jpan,kab_Latn,kac_Latn,kam_Latn,kan_Knda,kas_Arab,kas_Deva,kat_Geor,knc_Arab,knc_Latn,kaz_Cyrl,kbp_Latn,kea_Latn,khm_Khmr,kik_Latn,kin_Latn,kir_Cyrl,kmb_Latn,kon_Latn,kor_Hang,kmr_Latn,lao_Laoo,lvs_Latn,lij_Latn,lim_Latn,lin_Latn,lit_Latn,lmo_Latn,ltg_Latn,ltz_Latn,lua_Latn,lug_Latn,luo_Latn,lus_Latn,mag_Deva,mai_Deva,mal_Mlym,mar_Deva,min_Latn,mkd_Cyrl,plt_Latn,mlt_Latn,mni_Beng,khk_Cyrl,mos_Latn,mri_Latn,zsm_Latn,mya_Mymr,nld_Latn,nno_Latn,nob_Latn,npi_Deva,nso_Latn,nus_Latn,nya_Latn,oci_Latn,gaz_Latn,ory_Orya,pag_Latn,pan_Guru,pap_Latn,pol_Latn,por_Latn,prs_Arab,pbt_Arab,quy_Latn,ron_Latn,run_Latn,rus_Cyrl,sag_Latn,san_Deva,sat_Olck,scn_Latn,shn_Mymr,sin_Sinh,slk_Latn,slv_Latn,smo_Latn,sna_Latn,snd_Arab,som_Latn,sot_Latn,spa_Latn,als_Latn,srd_Latn,srp_Cyrl,ssw_Latn,sun_Latn,swe_Latn,swh_Latn,szl_Latn,tam_Taml,tat_Cyrl,tel_Telu,tgk_Cyrl,tgl_Latn,tha_Thai,tir_Ethi,taq_Latn,taq_Tfng,tpi_Latn,tsn_Latn,tso_Latn,tuk_Latn,tum_Latn,tur_Latn,twi_Latn,tzm_Tfng,uig_Arab,ukr_Cyrl,umb_Latn,urd_Arab,uzn_Latn,vec_Latn,vie_Latn,war_Latn,wol_Latn,xho_Latn,ydd_Hebr,yor_Latn,yue_Hant,zho_Hans,zho_Hant,zul_Latn"  # noqa


def load_nllb_model(
    checkpoint: Path,
    data_dir: Path,
    spm: Path,
    src_lang: str,
    tgt_lang: str,
    add_ssl_task_tokens: bool = False,
    add_data_source_prefix_tags: bool = True,
    langs: str = NLLB200_LANGS,
    fp16: bool = True,
    langs_as_single_string=False,
    **kwargs,
) -> FairseqMultilingualTransformerHub:
    """Loading an ALTI hub for an NLLB model in the format of the NLLB branch of Fairseq."""
    checkpoint_dir = str(checkpoint.parent)
    checkpoint_filename = str(checkpoint.name)

    x = fairseq.hub_utils.from_pretrained(
        model_name_or_path=checkpoint_dir,
        checkpoint_file=checkpoint_filename,
        task="translation_multi_simple_epoch",
        data_name_or_path=str(data_dir),
        langs=langs.split(","),
        lang_pairs=f"{src_lang}-{tgt_lang}",
        source_lang=src_lang,
        target_lang=tgt_lang,
        bpe="sentencepiece",
        sentencepiece_model=str(spm),
        decoder_langtok=True,
        add_data_source_prefix_tags=add_data_source_prefix_tags,
        encoder_langtok="src",
        beam=4,
        fp16=fp16,
        # extra arguments
        add_ssl_task_tokens=add_ssl_task_tokens,
        finetune_dict_specs=None,
        eval_bleu=False,
        eval_bleu_args=None,
        eval_bleu_all_same_batch=None,
        eval_bleu_remove_bpe=None,
        eval_tokenized_bleu=None,
        eval_bleu_detok=None,
        eval_bleu_print_samples=None,
        keep_inference_langtok=False,
        eval_bleu_tokenizer=None,
        **kwargs,
    )
    if langs_as_single_string:  # it may be done to please fairseq.hub_utils
        x["args"].task.langs = ",".join(x["args"].task.langs)
    alti_hub = FairseqMultilingualTransformerHub(
        models=x["models"], cfg=x["args"], task=x["task"]
    )
    return alti_hub


def load_bilingual_model(
    checkpoint: Path,
    data_dir: Path,
    spm: Path,
    **kwargs,
) -> FairseqTransformerHub:
    """Loading an ALTI hub for a bilingual Fairseq translation model."""
    checkpoint_dir = str(checkpoint.parent)
    checkpoint_filename = str(checkpoint.name)
    alti_hub = FairseqTransformerHub.from_pretrained(
        checkpoint_dir=checkpoint_dir,
        checkpoint_file=checkpoint_filename,
        data_name_or_path=str(data_dir),
        bpe="sentencepiece",
        sentencepiece_model=str(spm),
        **kwargs,
    )
    return alti_hub


@dataclasses.dataclass
class ALTIMetricsConfig:
    """The config indicating how to load sentence pairs, load the model,
    compute the ALTI metrics with it, and save results. - to use with the `compute_nllb_alti` function."""

    # the model used to compute ALTI
    is_multilingual: bool
    checkpoint: Path
    data_dir: Path
    spm: Path
    use_gpu: bool
    # location of the results
    metrics_filename: Path  # a .tsv file with sentence-level metrics
    alignment_filename: tp.Optional[
        Path
    ]  # a .jsonl file with token-level contributions
    # format and location of the source data
    input_filename: Path  # the source file with sources and translations; assumed to be .tsv
    src_lang: str
    tgt_lang: str
    src_col: tp.Union[str, int] = "src"
    tgt_col: tp.Union[str, int] = "mt"


def compute_nllb_alti(config: ALTIMetricsConfig) -> None:
    """Compute ALTI+ based attributions and metrics with an NLLB-like models and store the results in files."""
    columns_are_named = any(
        isinstance(c, str) for c in [config.src_col, config.tgt_col]
    )
    input_data = read_tsv(config.input_filename, named_columns=columns_are_named)
    src_texts, tgt_texts = select_columns(input_data, [config.src_col, config.tgt_col])
    if config.is_multilingual:
        alti_hub = load_nllb_model(
            checkpoint=Path(config.checkpoint),
            data_dir=config.data_dir,
            spm=config.spm,
            src_lang=config.src_lang,
            tgt_lang=config.tgt_lang,
        )
    else:
        alti_hub = load_bilingual_model(
            checkpoint=Path(config.checkpoint),
            data_dir=config.data_dir,
            spm=config.spm,
        )
    if config.use_gpu:
        if torch.cuda.is_available():
            alti_hub.cuda()
        else:
            logger.warning(
                "You requested use_gpu=True, but there is no GPU available. Falling back to CPU."
            )
    metrics, alignments = compute_alti_scores_for_batch(
        alti_hub=alti_hub,
        src_texts=src_texts,
        tgt_texts=tgt_texts,
        src_langs=config.src_lang,
        tgt_langs=config.tgt_lang,
    )
    write_tsv(config.metrics_filename, join_lists_of_dicts(input_data, metrics))

    if config.alignment_filename:
        with open(config.alignment_filename, "w", encoding="utf-8") as f:
            for item in alignments:
                json.dump(item, f, ensure_ascii=False)
                print(file=f)
