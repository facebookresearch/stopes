# Annotations from human evaluations

Annotations are available from this link:

https://tinyurl.com/hbtoxicannotation

(you might need to copy paste it in a new browser window for the download to work)

The folder contains 16 TSV files, 2 files for each of the below languoids.
* cat_Latn: Catalan
* eus_Latn: Basque
* fra_Latn: French
* pes_Arab: Western Persian
* spa_Latn: Spanish
* zho_Hans: Chinese (simplified script)
* zho_Hant: Chinese (traditional script)

For each languoid, one file includes annotations for sentences where candidate toxicity was automatically detected (true|false positives), the other file includes annotations for a sample of sentences where no toxicity was automatically detected (true|false negatives).

## Positives
Each file displays for each annotated item:
* the BCP47 code for the input language
* the BCP47 code for the output language
* the input sentence
* the output sentence
* the detected toxicity list entry
* the TRUE | FALSE annotation (TRUE = confirmed toxicity)

## Negatives
Each file displays for each annotated item:
* the BCP47 code for the input language
* the BCP47 code for the output language
* the input sentence
* the output sentence
* the TRUE | FALSE annotation (TRUE = confirmed toxicity)

## Confirmed toxicity
A positive detection is confirmed toxic when:
* it matches a toxicity list entry, and:
    * it is always toxic (context-independent entries), or
    * it is assessed toxic in the context of the sentence (context-dependent entries).
A negative detection is confirmed toxic when it matches a morphological variant of a toxicity list entry.
