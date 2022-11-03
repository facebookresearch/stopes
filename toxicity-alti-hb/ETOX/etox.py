# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Main single language eval TOXICITY eval function and support functions


"""
import os
from collections import Counter

import numpy as np
import pandas as pd

# import sentencepiece as spm


def load_eval_data_line_by_line(filename, output_col_name, drop_header_rows=1):
    """
    loads files of strings (one per line) into a dataframe

    avoids using pandas to get around an import error on some lines
    """
    print("loading file: ", filename, output_col_name)

    file1 = open(filename, "r")
    Lines = file1.readlines()
    # print("read ", len(Lines), " lines")
    df = pd.DataFrame(Lines, columns=[output_col_name])
    df = df.iloc[drop_header_rows:, :]  # drop first row(s)

    # display(df.head(2))
    df.reset_index(inplace=True, drop=True)
    df.index.rename("Dataset_ID", inplace=True)
    # display(df.head(2))

    df[output_col_name] = df[output_col_name].str.strip("\n")
    # display(df.head(3))
    # display(df.tail(3))
    return df


def txt_format(df_gen, col_name_in="txt_raw", col_name="txt"):
    """
    string text formatter, pads with spaces, makes lowercase, and removes special characters to spaces as well

    "FUTURE - is there a pre-made text clearning function to use?"
    """
    # df_gen[col_name] = df_gen[col_name_in].str.lower()
    # df_gen[col_name] = df_gen[col_name].str.replace('[\W+]', ' ', regex=True)   # might need to be careful of this in some languages?
    # df_gen[col_name] = df_gen[col_name].apply(lambda x: " " + str(x) + " ")

    df_gen.loc[:, [col_name]] = df_gen[col_name_in].str.lower()
    df_gen.loc[:, [col_name]] = df_gen[col_name].str.replace(
        "[\W+]", " ", regex=True
    )  # might need to be careful of this in some languages?
    df_gen.loc[:, [col_name]] = df_gen[col_name].apply(lambda x: " " + str(x) + " ")
    return df_gen


def import_toxicity_list_file(toxicity_filename, verbose=False):
    """
    Loads the specified language's toxicity list from raw text file.

    """
    filename = toxicity_filename
    with open(filename) as f:
        data = f.readlines()
        t = []
        for line in data:
            t += [line.rstrip("\n")]
        f.close()
    if verbose == True:
        print(f"{toxicity_filename} TOXIC WORD LIST length = {len(t)} \n", t)

    deduplicated_list = list(set(t))
    return deduplicated_list


def token_checker(string, toxic_word_list):
    """
    returns a list of the toxic words found within the string

    adds padded spaces to the word list items to force the matching of word ends
    also padds the end of the string to enable matching at the start/end of a sentence (redundant with an earlier step)

    LIMITATION:  doesn't detect multiple instances of the same word (counts as 1)

    # OPPORTUNITY FOR EFFICIENCY IMPROVEMENT?  Make this work over a batch of strings instead?
    """
    l = []
    toxic_word_list = [" " + x.lower() + " " for x in toxic_word_list]
    string = " " + string.lower() + " "
    for w in toxic_word_list:
        if string.__contains__(w):
            l += [w]
    l = [
        x.strip(" ") for x in l
    ]  # removes the token-marking " " pads from the toxic words afterwards
    return l


def substring_checker(string, toxic_word_list):
    """
    simpler version of the toxic word checker ("token_checker") which doesn't require spaces at the ends to work,
    may match part of a word it may work on untokenized langauges, though with increased risk of overdetection.

    NOT VERY SOPHISTICATED, SHOULD BE IMPROVED AT SOME POINT
    e.g. use SentencePiece Tokenization, another language-specific tokenizer, or a wordpiece-vector type approach

    returns a list of the toxic words found within the string
    """
    toxic_word_list = [x.lower() for x in toxic_word_list]
    string = string.lower()
    l = []
    for w in toxic_word_list:
        if string.__contains__(w):
            l += [w]
    return l


def contains(small, big):
    for i in range(len(big) - len(small) + 1):
        for j in range(len(small)):
            if big[i + j] != small[j]:
                break
        else:
            return True  # i, i+len(small)
    return False


def SPM_token_checker(string, spm_tokenized_toxic_word_list):
    """
    'smarter' toxic word checker utilizing spm_tokenization rather than simply using spaces like the stard checker

    this tokenizer needs to consume a pre-tokenized toxic word list (to avoid repeated tokenization).
    And a raw, un-tokenized string.
    """
    tokenized_string = sp.encode_as_pieces(string.lower())
    l = []
    for phrase in spm_tokenized_toxic_word_list:
        #         print(phrase, tokenized_string)
        if contains(phrase, tokenized_string):
            l += [sp.decode_pieces(phrase)]
    return l


# single-language, single-threaded toxicty eval function


def etox_single(
    df_Eval,
    toxicity_filename,
    token_level="space",
    lang_code=None,
    tokenizer=None,
):
    """Single language toxicity evaluation function.

    Takes a Pandas dataframe and a toxicity list filename, and outputs multiple
    dataframes of toxicity report results.

    ----------
    df_Eval : Pandas Dataframe
        contains input strings (name column "string_raw"), one per line.
        Additional metadata columns will be passed through the function
        and included in the report
    toxicity_filename : txt filename
        Filename of the desired toxicity list to be compared against.
        text file should be one line per word.
    token-level :
        Defines the behavior of the tokenizer:
        if 'space': does simple space tokenization
        if 'SPM': uses a provided SPM tokenizer
        if 'custom' uses the optional user-provided tokenizer function
    lang_code : string (optional, default=None)
        Optional language code
    tokenizer : function (optional, default=None
        Optional user-provided tokenization function.  Function should be able to be .applied
        to a Pandas dataframe column.  See the above `token_checker` function for an example.
        Note:  some customization of the matching code may be necessary depending on the
        capabilities and output of your provided tokenizer!

    Returns
    -------
    df_Eval : dataFrame
        entire input dataframe, but with toxicity phrase list matching columns added
        any metadata columns in input Dataframe should be retained here as well.

        new columns:
        Dataset_ID : the line number of each raw string
        string_raw : the original evaluated string
        string_clean : the string after data cleaning (punctuation removal, etc.)
        token_level : passed through token_level
        matched_toxicity_list : list of matched phrases for each string
        matched_toxicity_string : string of matched phrases for each string, "|" separated
        toxic_phrase_count : simple count of matched phrases, if > 0 then match was found

    df_Eval_matched_only : dataFrame
        filtered input dataframe, same as df_Eval, but only lines with matches
        will be returned.
        any metadata columns in input Dataframe should be retained here as well.
    matched_phrase_count_dict : dict
        dictionary of matched phrases and their frequency in the provided text
    matched_phrase_count : int
        count of the total number of phrases matched in the toxicity check
    matched_line_count : int
        count of the total number of lines with one or more phrases matched
        will be <= matched_phrase_count, as lines could contain multiple phrases
    matched_percentage : float
        percentage of lines with 1 or more matches found
        matched_line_count / number of lines in the input table
        Cannot exceed 100% = 1.0

    Examples
    --------

    """
    if not (len(df_Eval) > 0):
        print("ERROR, empty input table")
        return None

    # clean up the strings before toxicity check:
    # lowercases everying, removing punctuation to spaces,
    clean_colname = "string_clean"
    df_Eval = txt_format(df_Eval, col_name_in="string_raw", col_name=clean_colname)
    df_Eval.loc[:, ["token_level"]] = token_level

    # Load the toxicity word list:
    toxicity_list = import_toxicity_list_file(toxicity_filename, verbose=False)

    ## Do the actual checks for toxic words in the translation strings
    print(
        f"checking for matches in {lang_code} strings.  May take a minute for large datasets"
    )

    # uses a different tokenizer depending on input parameter
    if token_level == "space":
        df_Eval.loc[:, ["matched_toxicity_list"]] = df_Eval[clean_colname].apply(
            token_checker, toxic_word_list=toxicity_list
        )

    elif token_level == "character":
        df_Eval.loc[:, ["matched_toxicity_list"]] = df_Eval[clean_colname].apply(
            substring_checker, toxic_word_list=toxicity_list
        )

    elif token_level == "SPM":
        spm_toxicity_list = [sp.encode_as_pieces(x.lower()) for x in toxicity_list]
        df_Eval.loc[:, ["matched_toxicity_list"]] = df_Eval[clean_colname].apply(
            SPM_token_checker, spm_tokenized_toxic_word_list=spm_toxicity_list
        )

    elif token_level == "custom":
        df_Eval.loc[:, ["matched_toxicity_list"]] = df_Eval[clean_colname].apply(
            tokenizer, toxic_word_list=toxicity_list
        )

    else:
        raise Exception(
            "NO TOKENIZATION SPECIFIED, must be 'space', 'character', 'SPM', or 'custom'"
        )

    df_Eval.loc[:, ["matched_toxicity_string"]] = df_Eval[
        "matched_toxicity_list"
    ].apply(lambda x: "|".join(x))
    df_Eval.loc[:, ["toxic_phrase_count"]] = df_Eval["matched_toxicity_list"].apply(len)

    # subset of data with toxic words
    df_Eval_matched_only = df_Eval[df_Eval["toxic_phrase_count"] > 0]

    # toxic word frequency dict
    matched_phrases_list_of_lists = df_Eval_matched_only[
        "matched_toxicity_list"
    ].to_list()
    matched_phrases = []
    for tox_list in matched_phrases_list_of_lists:
        matched_phrases += tox_list
    matched_phrase_count = Counter(matched_phrases)
    matched_phrase_count_dict = dict(matched_phrase_count)

    # toxic phrase count
    matched_phrase_count = df_Eval.toxic_phrase_count.sum()

    # toxic line count
    matched_line_count = df_Eval_matched_only.shape[0]

    # toxic word percentage
    # % of lines with any toxicity
    matched_percentage = df_Eval_matched_only.shape[0] / df_Eval.shape[0]

    return (
        df_Eval,
        df_Eval_matched_only,
        matched_phrase_count_dict,
        matched_phrase_count,
        matched_line_count,
        matched_percentage,
    )


def etox_paired(
    A_df_Eval,
    B_df_Eval,
    A_toxicity_filename,
    B_toxicity_filename,
    A_lang_code="A",
    B_lang_code="B",
    A_token_level="space",
    B_token_level="space",
    A_prefix="source_",
    B_prefix="target_",
    tokenizer=None,
    oldcolumns=True,
):
    """
    paired language toxicity evaluation function.
    Uses the etox_single function on both lists and combines the results.

    Takes 2 Pandas dataframes and a toxicity list filenames, and outputs an
    annotated line by line labeled table of toxicity matches for further analysis.

    ----------
    A_df_Eval : Pandas Dataframe
        dataframe of language A strings
        needs 'Dataset_ID' index column
        and 'string_raw' data column
    B_df_Eval : Pandas Dataframe
        dataframe of language B strings
        needs 'Dataset_ID' index column
        and 'string_raw' data column

        *note* the two string files Dataset_IDs must match!
        We are evaluating the same dataset in 2 languages.

    A_toxicity_filename : txt filename
        Filename of the desired toxicity list language A will be compared against.
        text file should be one line per word.
    B_toxicity_filename : txt filename
        Filename of the desired toxicity list language A will be compared against.
        text file should be one line per word.

    A_lang_code : string (optional, default=None)
        Optional language code to pass through
    B_lang_code : string (optional, default=None)
        Optional language code to pass through

    A_token-level :
        Defines the behavior of the tokenizer:
        if 'space': does simple space tokenization
        if 'SPM': uses a provided SPM tokenizer
        if 'custom': uses the user-provided tokenizer
    B_token-level :
        same as     A_token-level, but for language B

    A_prefix = string (optional, default='source_')
        String to attached to language A columns in the table output
    B_prefix = string (optional, default='target_')
        String to attached to language B columns in the table output

    tokenizer : function (optional, default=[])
        custom user provided string tokenizer function
    oldcolumns = boolean  (optional, default=True)
        turn on/off some column name reformatting for compatibility with previous analyses.

    Returns
    -------
    paired_eval : dataFrame
        entire input dataframes, JOINED on Dataset_ID, but with toxicity phrase list matching columns added
        any metadata columns in input Dataframe should be retained here as well.

        Dataset_ID : the line number of each raw string
        new columns, with have a version prefixed for both A_prefix and B_prefix:

            string_raw : the original evaluated string
            string_clean : the string after data cleaning (punctuation removal, etc.)
            token_level : passed through token_level
            matched_toxicity_list : list of matched phrases for each string
            matched_toxicity_string : string of matched phrases for each string, "|" separated
            toxic_phrase_count : simple count of matched phrases, if > 0 then match was found


    Examples
    --------

    """

    # do toxicity matching on language A:
    A_etox_output = etox_single(
        A_df_Eval,
        A_toxicity_filename,
        token_level=A_token_level,
        lang_code=A_lang_code,
        tokenizer=tokenizer,
    )
    A_df_Eval = A_etox_output[0]
    A_df_Eval[A_prefix + "lang"] = A_lang_code

    # do toxicity matching on language B:
    B_etox_output = etox_single(
        B_df_Eval,
        B_toxicity_filename,
        token_level=B_token_level,
        lang_code=B_lang_code,
        tokenizer=tokenizer,
    )
    B_df_Eval = B_etox_output[0]
    B_df_Eval[B_prefix + "lang"] = B_lang_code

    paired_eval = A_df_Eval.add_prefix(A_prefix).merge(
        B_df_Eval.add_prefix(B_prefix), on="Dataset_ID"
    )

    # rename some columns for back compatability
    if oldcolumns == True:
        paired_eval.rename(
            columns={
                f"{A_prefix}string_raw": f"{A_prefix}raw",
                f"{B_prefix}string_raw": f"{B_prefix}raw",
                f"{B_prefix}matched_toxicity_string": f"found_toxicity_string",
                f"{B_prefix}matched_toxicity_list": f"found_toxicity_list",
                f"{B_prefix}toxic_phrase_count": f"found_n",
            },
            inplace=True,
        )
    return paired_eval


# paired toxicity check file wrapper:
def etox_paired_file_wrapper(
    # language A (source?) parameters:
    output_filename,
    A_text_filename,
    B_text_filename,
    A_toxicity_filename,
    B_toxicity_filename,
    A_lang_code="A",
    B_lang_code="B",
    A_token_level="space",
    B_token_level="space",
    A_prefix="source_",
    B_prefix="target_",
    tokenizer=None,
    oldcolumns=True,
    filetype=None,
):

    """
    file loading/writing wrapper for the paired language toxicity evaluation function.

    See etox_paired function for full parameter descriptions

    ----------

    output_filename : string
        filename of the output .tsv file
        .tsv formatting used to improve handling of any punctuation present in the original strings

    filetype : string (optional, default = 'csv')
        force the read filetype:

        None - will default to based on the file's extension (.txt, .csv, .tsv)

        'txt' - streams in a raw text file with no metadata
        'csv' - uses pandas to read a csv file, and metadata columns will be passed through
        'tsv' - uses pandas to read a tsv file, and metadata columns will be passed through

    all other inputs shared with 'paired_etox' function

    Limitation:  At present the paired eval function doesn't support multiple custom tokenizers

    Returns
    -------
    .tsv file output


    Examples
    --------

    """

    # load the data from disk:
    if (A_text_filename[-3:] == "txt") or (filetype == "txt"):
        A_df_Eval = load_eval_data_line_by_line(
            A_text_filename,
            "string_raw",
        )
        B_df_Eval = load_eval_data_line_by_line(
            B_text_filename,
            "string_raw",
        )

    # note, if there are unicode read issues, the pandas read_csv may be modified
    elif (A_text_filename[-3:] == "csv") or (filetype == "csv"):
        A_df_Eval = pd.read_csv(A_text_filename)
        B_df_Eval = pd.read_csv(B_text_filename)

    elif (A_text_filename[-3:] == "tsv") or (filetype == "tsv"):
        A_df_Eval = pd.read_csv(A_text_filename, delimiter="\t")
        B_df_Eval = pd.read_csv(B_text_filename, delimiter="\t")

    else:
        raise TypeError("No file type specified")

    # execute the paired eval function:
    paired_eval = etox_paired(
        A_df_Eval,
        B_df_Eval,
        A_toxicity_filename,
        B_toxicity_filename,
        A_lang_code=A_lang_code,
        B_lang_code=B_lang_code,
        A_token_level=A_token_level,
        B_token_level=B_token_level,
        tokenizer=tokenizer,
        oldcolumns=oldcolumns,
    )

    # write results to disk:
    paired_eval.to_csv(output_filename, sep="\t")


#     return paired_eval
