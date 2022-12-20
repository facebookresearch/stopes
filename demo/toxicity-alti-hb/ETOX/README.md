# ETOX tool

Contains scripts for calculating toxicity results, given files of input strings and toxicity lists.

Prerequisites:
- Install the HolisticBias module ([setup instructions](https://github.com/facebookresearch/ResponsibleNLP/tree/main/holistic_bias))
- Define paths to include etox.py

Files:
- `ETOX example calls.ipynb`: Example Usage of the main ETOX toxicity tool functions.
- `etox.py`: contains all the python functions for the ETOX tool
- `README.md`: this file

Functions:
Main Functions:
- `etox_single`:  Takes a Pandas dataframe and a toxicity list filename, and outputs multiple dataframes of toxicity report results.
- `etox_paired`:  Paired language toxicity evaluation function.  Takes 2 Pandas dataframes and a toxicity list filenames, and outputs an annotated line by line labeled table of toxicity matches for further analysis.
- `etox_paired_file_wrapper`:  File reading/writing wrapper for the paired language toxicity evaluation function.

Support Functions
- `load_eval_data_line_by_line` Loads a text file of strings, returns a Pandas Dataframe
- `txt_format`: simple data cleaning function.  Lowercases and uses regex to remove punctuation, etc.
- `import_toxicity_list_file`: reads a toxicity list file into memory given a filename.  Returns a List.
- `token_checker`:  Checks for matches between a string and a toxic word list used if 'space' tokenization selected
- `substring_checker`: checks for character level matches ignoring spaces.  Will find subwords.  Used if 'character' tokenization selected
- `SPM_token_checker`: Toxic phrase checker utilizing sub-word spm_tokenization rather than simply using spaces like the stard checker.  Useful for a few languages where space tokenization is unreliable, or when matching subtokens may be important.  Requires the Sentencepiece library to function.
