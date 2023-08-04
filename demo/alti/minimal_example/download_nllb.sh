# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

mkdir nllb
cd nllb
# downloading the vocabulary; 
wget --trust-server-names https://tinyurl.com/flores200sacrebleuspm
wget --trust-server-names https://tinyurl.com/nllb200dictionary

# downloading the smallest NLLB200 model; it may take about 5 minutes
wget --trust-server-names https://tinyurl.com/nllb200densedst600mcheckpoint 

for lang in ace_Latn acm_Arab acq_Arab aeb_Arab afr_Latn ajp_Arab aka_Latn als_Latn amh_Ethi apc_Arab arb_Arab ars_Arab ary_Arab arz_Arab asm_Beng ast_Latn awa_Deva ayr_Latn azb_Arab azj_Latn bak_Cyrl bam_Latn ban_Latn bel_Cyrl bem_Latn ben_Beng bho_Deva bjn_Latn bod_Tibt bos_Latn bul_Cyrl cat_Latn ceb_Latn ces_Latn cjk_Latn ckb_Arab crh_Latn cym_Latn dan_Latn deu_Latn dik_Latn dyu_Latn dzo_Tibt ell_Grek eng_Latn epo_Latn est_Latn eus_Latn ewe_Latn fao_Latn fij_Latn fin_Latn fon_Latn fra_Latn fur_Latn fuv_Latn gaz_Latn gla_Latn gle_Latn glg_Latn grn_Latn guj_Gujr hat_Latn hau_Latn heb_Hebr hin_Deva hne_Deva hrv_Latn hun_Latn hye_Armn ibo_Latn ilo_Latn ind_Latn isl_Latn ita_Latn jav_Latn jpn_Jpan kab_Latn kac_Latn kam_Latn kan_Knda kas_Arab kas_Deva kat_Geor kaz_Cyrl kbp_Latn kea_Latn khk_Cyrl khm_Khmr kik_Latn kin_Latn kir_Cyrl kmb_Latn kmr_Latn knc_Arab knc_Latn kon_Latn kor_Hang lao_Laoo lij_Latn lim_Latn lin_Latn lit_Latn lmo_Latn ltg_Latn ltz_Latn lua_Latn lug_Latn luo_Latn lus_Latn lvs_Latn mag_Deva mai_Deva mal_Mlym mar_Deva min_Latn mkd_Cyrl mlt_Latn mni_Beng mos_Latn mri_Latn mya_Mymr nld_Latn nno_Latn nob_Latn npi_Deva nso_Latn nus_Latn nya_Latn oci_Latn ory_Orya pag_Latn pan_Guru pap_Latn pbt_Arab pes_Arab plt_Latn pol_Latn por_Latn prs_Arab quy_Latn ron_Latn run_Latn rus_Cyrl sag_Latn san_Deva scn_Latn shn_Mymr sin_Sinh slk_Latn slv_Latn smo_Latn sna_Latn snd_Arab som_Latn sot_Latn spa_Latn srd_Latn srp_Cyrl ssw_Latn sun_Latn swe_Latn swh_Latn szl_Latn tam_Taml taq_Latn tat_Cyrl tel_Telu tgk_Cyrl tgl_Latn tha_Thai tir_Ethi tpi_Latn tsn_Latn tso_Latn tuk_Latn tum_Latn tur_Latn twi_Latn tzm_Tfng uig_Arab ukr_Cyrl umb_Latn urd_Arab uzn_Latn vec_Latn vie_Latn war_Latn wol_Latn xho_Latn ydd_Hebr yor_Latn yue_Hant zho_Hans zho_Hant zsm_Latn zul_Latn; do
    cp dictionary.txt dict.${lang}.txt
done

cd ..
