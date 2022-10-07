# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import argparse
import os
import random

import pandas as pd
import sentencepiece as spm
from tqdm import tqdm
from util import (
    END_PUNCTUATION,
    LANG_ALLOWLIST,
    LEAKED_LANG_TAGS,
    SOURCE_SENTENCES_PATH,
    SPM_MODEL_PATH,
    SPM_TOKENIZATION_LANGS,
    TOXICITY_SOURCE_FOLDER,
    UNK_LANGS,
    get_holistic_bias_metadata_map,
    get_per_line_data,
    identify_toxic_words,
)


def count_toxicity_sources(skip_compilation: bool):

    # Save paths
    save_folder = TOXICITY_SOURCE_FOLDER
    log_folder = os.path.join(save_folder, "logs")
    full_result_folder = os.path.join(save_folder, "full_results")
    for folder in [save_folder, log_folder, full_result_folder]:
        os.makedirs(folder, exist_ok=True)
    log_path_template = os.path.join(log_folder, "{lang_string}.txt")
    full_result_path_template = os.path.join(full_result_folder, "{lang_string}.csv")
    num_toxic_words_path = os.path.join(save_folder, "num_toxic_words.csv")
    aligned_word_category_frac_path = os.path.join(
        save_folder, "aligned_word_category_fractions.csv"
    )
    aligned_word_category_frac_by_language_path = os.path.join(
        save_folder, "aligned_word_category_fractions_by_language.csv"
    )
    grouped_category_and_word_count_path = os.path.join(
        save_folder, "counts_of_aligned_words_and_categories.csv"
    )
    pivoted_aligned_words_by_lang = os.path.join(
        save_folder, "counts_of_aligned_words_and_categories_by_language.csv"
    )

    if not skip_compilation:

        sp = spm.SentencePieceProcessor(model_file=SPM_MODEL_PATH)

        holistic_bias_sentence_metadata_map = get_holistic_bias_metadata_map(
            save_folder
        )

        print("Loading source sentences.")
        with open(SOURCE_SENTENCES_PATH) as f:
            raw_orig_sentences = f.readlines()
            orig_sentences = [line.strip() for line in raw_orig_sentences]

        print("Finding the English word that each toxic word is most aligned to:")
        alignment_info_per_lang_dfs = []
        for lang_string in tqdm(LANG_ALLOWLIST):

            # Save paths
            log_path = log_path_template.format(lang_string=lang_string)
            full_result_path = full_result_path_template.format(lang_string=lang_string)

            f_log = open(log_path, "w")

            outputs = get_per_line_data(
                lang_string=lang_string, orig_sentences=orig_sentences
            )
            if outputs is None:
                continue
            else:
                (
                    translated_sentences,
                    alignment_strings,
                    _,
                    toxic_word_map,
                ) = outputs

            all_alignment_info = []
            num_wrong_num_alignment_strings = 0
            num_invalid_toxic_orig_word_idxes = 0
            for (
                line_idx,
                orig_sentence,
                translated_sentence,
                alignment_string,
            ) in zip(
                range(len(orig_sentences)),
                orig_sentences,
                translated_sentences,
                alignment_strings,
            ):

                if orig_sentence == "text":
                    # This is a header line
                    continue

                # Extract per-word info and check word counts
                orig_sentence_words = orig_sentence.lower().split()
                if lang_string in UNK_LANGS:
                    translated_sentence = translated_sentence.replace("<unk>", "<unk> ")
                if lang_string in LEAKED_LANG_TAGS:
                    # Separate the anomalous language tags with spaces in order for the
                    # alignment pair counts to line up
                    for lang_tag in LEAKED_LANG_TAGS[lang_string]:
                        translated_sentence = translated_sentence.replace(
                            lang_tag, " " + lang_tag
                        )
                translated_sentence_words = translated_sentence.lower().split()
                alignment_pairs = [
                    tuple(
                        [
                            int(word_idx_str)
                            for word_idx_str in alignment_pair.split("-")
                        ]
                    )
                    for alignment_pair in alignment_string.split()
                ]
                if len(translated_sentence_words) != len(alignment_pairs):
                    f_log.write(
                        f"\nLanguage {lang_string}, line {line_idx:d}: implied word count does not match among files.\n\t{len(translated_sentence_words):d} translated sentence words: {translated_sentence}\n\t{len(alignment_pairs):d} alignment pairs: {alignment_string}\n"
                    )
                    if lang_string in SPM_TOKENIZATION_LANGS:
                        f_log.write(
                            f"\tTranslated sentence, divided into SPM-tokenized words: "
                            + ", ".join(
                                f'"{word}"' for word in translated_sentence_words
                            )
                            + "\n"
                        )
                    num_wrong_num_alignment_strings += 1
                    continue
                assert [
                    alignment_pair[0] for alignment_pair in alignment_pairs
                ] == list(range(len(translated_sentence_words)))

                (
                    descriptor,
                    noun,
                    template,
                    effective_descriptor,
                    effective_noun,
                ) = holistic_bias_sentence_metadata_map[orig_sentence][:5]

                # Identify toxic words
                toxic_words, toxic_translated_word_idxes = identify_toxic_words(
                    translated_sentence=translated_sentence,
                    toxic_word_map=toxic_word_map,
                    lang_string=lang_string,
                    sp=sp,
                )

                # Pick a random instance of the toxic word (in the case where there is
                # more than one toxic word in the target sentence)
                if len(toxic_translated_word_idxes) == 0:
                    # No toxic words, so nothing to analyze for this translation
                    continue
                else:
                    toxic_translated_word_idx = random.sample(
                        toxic_translated_word_idxes, k=1
                    )[0]

                # Identify the category of the aligned word in the source sentence
                toxic_orig_word_idx = alignment_pairs[toxic_translated_word_idx][1]
                if toxic_orig_word_idx < len(orig_sentence_words):
                    aligned_word = orig_sentence_words[toxic_orig_word_idx]
                    effective_descriptor_words = effective_descriptor.lower().split()
                    template_words = template.lower().split()
                    in_descriptor = (
                        aligned_word.rstrip(END_PUNCTUATION)
                        in effective_descriptor_words
                    )
                    is_noun = aligned_word.rstrip(END_PUNCTUATION) == effective_noun
                    in_template = aligned_word in template_words
                    if in_descriptor and is_noun and not in_template:
                        # For instance, "I am a kid with one kid."
                        aligned_word_category = "descriptor_and_noun"
                    elif in_descriptor and not is_noun and not in_template:
                        aligned_word_category = "descriptor"
                    elif not in_descriptor and is_noun and not in_template:
                        aligned_word_category = "noun"
                    elif not in_descriptor and not is_noun and in_template:
                        aligned_word_category = "template"
                    else:
                        raise Exception(
                            f"{lang_string}, line {line_idx:d}: The aligned word cannot be mapped to a valid part of the sentence!"
                        )
                elif toxic_orig_word_idx == len(orig_sentence_words):
                    # The toxic word is aligned to the EOS of the original sentence
                    aligned_word_category = "eos"
                    aligned_word = ""
                else:
                    f_log.write(
                        f"\nLanguage {lang_string}, line {line_idx:d}: the index of the aligned English word, {toxic_orig_word_idx:d}, exceeds the number of words of the original sentence.\n\t{len(orig_sentence_words):d} original sentence words: {orig_sentence}\n\t{len(translated_sentence_words):d} translated sentence words: {translated_sentence}\n\t{len(alignment_pairs):d} alignment pairs: {alignment_string}\n"
                    )
                    num_invalid_toxic_orig_word_idxes += 1
                    continue

                alignment_info = {
                    "lang_string": lang_string,
                    "descriptor": descriptor,
                    "noun": noun,
                    "template": template,
                    "effective_descriptor": effective_descriptor,
                    "effective_noun": effective_noun,
                    "orig_sentence": orig_sentence,
                    "translated_sentence": translated_sentence,
                    "toxic_words": ",".join(toxic_words),
                    "num_toxic_word_matches": len(toxic_translated_word_idxes),
                    "aligned_word": aligned_word,
                    "aligned_word_category": aligned_word_category,
                }

                all_alignment_info.append(alignment_info)

            alignment_info_this_lang_df = pd.DataFrame(all_alignment_info)

            alignment_info_per_lang_dfs.append(alignment_info_this_lang_df)

            alignment_info_this_lang_df.to_csv(full_result_path, index=False)

            f_log.write(
                f"\nNumber of {lang_string} lines with an unexpected number of alignment pairs: {num_wrong_num_alignment_strings:d}\n"
            )
            f_log.write(
                f"\nNumber of {lang_string} lines with an invalid index for the source word aligned to the toxic word: {num_invalid_toxic_orig_word_idxes:d}\n"
            )

            f_log.close()

    else:

        alignment_info_per_lang_dfs = []
        for lang_string in tqdm(LANG_ALLOWLIST):
            full_result_path = full_result_path_template.format(lang_string=lang_string)
            try:
                alignment_info_this_lang_df = pd.read_csv(full_result_path)
            except pd.errors.EmptyDataError:
                # No toxicity for this language
                continue
            alignment_info_per_lang_dfs.append(alignment_info_this_lang_df)

    print("Concatenating results from all languages.")
    alignment_info_df = pd.concat(alignment_info_per_lang_dfs, axis=0)

    print(
        f"Saving the distribution of numbers of toxic words to {num_toxic_words_path}."
    )
    num_toxic_words_dist_df = (
        alignment_info_df.assign(count=1)
        .groupby("num_toxic_word_matches")
        .agg({"count": "sum"})
        .assign(frac=lambda df: df["count"] / df["count"].sum())
    )
    num_toxic_words_dist_df.to_csv(num_toxic_words_path)

    print(
        f"Saving the breakdown of aligned word category to {aligned_word_category_frac_path}."
    )
    word_category_dist_df = (
        alignment_info_df.assign(count=1)
        .groupby("aligned_word_category")
        .agg({"count": "sum"})
        .assign(frac=lambda df: df["count"] / df["count"].sum())
    )
    word_category_dist_df.to_csv(aligned_word_category_frac_path)

    print(
        f"Saving the aligned word category distribution as a function of language to {aligned_word_category_frac_by_language_path}."
    )
    grouped_lang_count_df = (
        alignment_info_df.assign(lang_count=1)
        .groupby("lang_string")
        .agg({"lang_count": "sum"})
        .reset_index()
    )
    grouped_lang_and_category_count_df = (
        alignment_info_df.assign(lang_and_category_count=1)
        .groupby(["lang_string", "aligned_word_category"])
        .agg({"lang_and_category_count": "sum"})
        .reset_index()
    )
    pivoted_category_frac_df = (
        pd.merge(
            left=grouped_lang_count_df,
            right=grouped_lang_and_category_count_df,
            how="outer",
            on="lang_string",
        )
        .assign(
            category_frac=lambda df: df["lang_and_category_count"] / df["lang_count"]
        )
        .pivot(
            index="lang_string", columns="aligned_word_category", values="category_frac"
        )
        .fillna(0)
        .join(grouped_lang_count_df.set_index("lang_string"))
        .sort_values("lang_count", ascending=False)
    )
    pivoted_category_frac_df.to_csv(aligned_word_category_frac_by_language_path)

    print(
        f"Saving the counts of toxic words by aligned word category and aligned word to {grouped_category_and_word_count_path}."
    )
    alignment_info_with_rstripped_alignment_word_df = alignment_info_df.assign(
        rstripped_aligned_word=lambda df: df["aligned_word"].str.rstrip(
            END_PUNCTUATION
        ),
        count=1,
    )
    grouped_category_and_word_count_df = (
        alignment_info_with_rstripped_alignment_word_df.groupby(
            ["aligned_word_category", "rstripped_aligned_word"]
        )
        .agg({"count": "sum"})
        .sort_values(["count"], ascending=False)
    )
    grouped_category_and_word_count_df.to_csv(grouped_category_and_word_count_path)

    print(
        f"Saving the counts of toxic words by aligned word category, aligned word, and language to {pivoted_aligned_words_by_lang}."
    )
    grouped_lang_category_and_word_count_df = (
        alignment_info_with_rstripped_alignment_word_df.groupby(
            ["lang_string", "aligned_word_category", "rstripped_aligned_word"]
        ).agg({"count": "sum"})
    )
    pivoted_lang_cat_word_count_df = (
        pd.pivot_table(
            data=grouped_lang_category_and_word_count_df,
            values="count",
            index=["aligned_word_category", "rstripped_aligned_word"],
            columns="lang_string",
        )
        .fillna(0)
        .assign(lang_count=lambda df: df.sum(axis=1))
        .sort_values(["lang_count"], ascending=False)
    )
    pivoted_lang_cat_word_count_df.to_csv(pivoted_aligned_words_by_lang)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--skip-compilation",
        action="store_true",
        help="Read in pre-compiled raw results, for further analysis.",
    )
    args = parser.parse_args()
    count_toxicity_sources(
        skip_compilation=args.skip_compilation,
    )
