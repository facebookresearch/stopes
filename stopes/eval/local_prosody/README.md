# Local prosody toolkit

This folder contains a set of tools to evaluate different prospects of local prosody (speech rate and pauses)
and their similarities across parallel spoken sentences in different languages.

## Requrements

To use the tools, please first install the `local_prosody` subset of requirements of `stopes`.

```
pip install -e '.[local_prosody]'
```

If you plan to use the UnitY2 forced aligner (a recommended option), please also install the [seamless_communication](https://github.com/facebookresearch/seamless_communication) package.

If you plan to work with Chinese, please install [pkuseg](https://github.com/lancopku/pkuseg-python) for word segmentation.

## Usage

The tool works in two steps:

1. First, it aligns a spoken utterance with it transcription for each language individually,
   annotates paused and duration of each word, and computes the speech rate.
2. Then it aligns the words across two utterances it different languages and compares durations and locations of pauses between them.

In the example below, we will suppose that you have a .tsv file at `$INPUT_PATH`, with pairs of English and Spanish utterances
(by utterance, we mean a spoken equivalent of a sentence, or maybe a sequence of a few uninterrupted sentences by the same speaker).
Their transcriptions and paths to the corresponding .wav files are represented as columns `text_spa`, `text_eng`, `path_spa` and `path_eng`.

### Step 1: Utterance annotation

To produce annotations for the Spanish side (for English, it is equivalent), please run the following command:

```
python stopes/eval/local_prosody/annotate_utterances.py \
    +data_path=$INPUT_PATH \
    +result_path=$RESULT_PATH_SPA \
    +audio_column=spa_path \
    +text_column=text_spa \
    +speech_units=[word,syllable,char,phoneme,vowel] \
    +vad=true \
    +net=true \
    +lang=spa \
    +forced_aligner=fairseq2_nar_t2u_aligner
```

Some of the important arguments of this script are:

- `text_column`, `audio_column`: columns with the input texts and paths to audios
- `forced_aligner`: an [algorithm to align audio with text](#forced-alignment-of-words-with-speech); either `fairseq2_nar_t2u_aligner` or `ctc_wav2vec2-xlsr-multilingual-56`.
- `lang`: 3-letter language code, used for phonemization and word tokenization
- `speech_units`: list of units to compute speech rate with; a subset of `[word,syllable,char,phoneme,vowel]`
- `net`: whether to exclude pauses from speech duration (recommended value is `true`)
- `vad`: whether to use a VAD (voice activity detection) model to adjust the durations of words and pauses (recommended value is `true`)

The .tsv file at `$RESULT_PATH_SPA` will contain the following columns:

- `utterance`: a json string with word-level timestamps (see an example below)
- `text_with_markup`: a text with words of the utterance and pauses between them, like `de acuerdo [pause x 1.66] rellene esto y tráigamelo`
- `duration`: duration of the utterance.
  If you used the parameter `net=true`, the duration of the pauses will be excluded from it, i.e. we will add up durations of the words only.
- `speech_rate_word`: number of words per second (i.e. divided by `duration`, as defined above)
- `speech_rate_syllable`: number of syllables per second
  (identified with the [syllables](https://pypi.org/project/syllables/) package; reliable only for English and similar languages)
- `speech_rate_char`: number of characters per second
- `speech_rate_phoneme`: number of phonemes per second (computed with the [phonemizer](https://pypi.org/project/phonemizer/) package)
- `speech_rate_vowel`: number of vowels (as an more language-independent proxy of syllables) per second

The `utterance` column will contain JSON-dumped instances of `stopes.eval.local_prosody.Utterance` objects
which contain word segmentation of utterances and timestamps of these words:

```
{
    "text": "no|sabe|qu|le|pasa|te|necesita",
    "words": ["no", "sabe", "qu", "le", "pasa", "te", "necesita"],
    "starts": [0.1, 0.24, 0.52, 0.68, 0.8, 1.88, 2.04],
    "ends": [0.18, 0.48, 0.64, 0.76, 1.08, 2.0, 2.46]
}
```

### Step 2: Utterance comparison

To compare parallel utteranced in two different languages, you can run the following script, providing the paths
to the utterance annotation results in both languages (`$RESULT_PATH_SPA` and `$RESULT_PATH_ENG` in our example):

```
python stopes/eval/local_prosody/compare_utterances.py \
    +src_path=$RESULT_PATH_SPA \
    +tgt_path=$RESULT_PATH_ENG \
    +result_path=$RESULT_PATH_COMPARE \
    +pause_min_duration=0.1
```

The script will print aggregated results of pause and speech rate comparison, like below:

```
Micro-averaged (by pause) and macro-averaged (by utterance -> avg) results of pause alignment:
                        micro_avg  macro_avg
mean_duration_score      0.633677   0.668434
mean_alignment_score     0.747985   0.735321
mean_joint_score         0.603856   0.648817
wmean_duration_score     0.609355   0.687016
wmean_alignment_score    0.730457   0.753747
wmean_joint_score        0.575004   0.668704
total_weight           133.660000   1.336600
n_items                192.000000   1.920000
n_src_pauses            74.000000   0.740000
n_tgt_pauses            80.000000   0.800000

Speech rate source-target correlations:
                       pearson  spearman
speech_rate_word      0.227924  0.331777
speech_rate_syllable  0.073217  0.315232
speech_rate_char      0.117409  0.287887
speech_rate_phoneme   0.166429  0.302138
```

The .tsv file `$RESULT_PATH_COMPARE` will contain the utterance-level results of
word alignment across languages and pause alignment, with the following columns:

- `src_utterance`, `tgt_utterance`: the `utterance` objects copied from the input files (including timestamps of words and pauses)
- `word_alignment`: the computed mapping between source and target words
- `n_src_pauses`: number of pauses in the source utterance
- `n_tgt_pauses`: number of pauses in the target utterance
- `total_weight`: sum of source and targer pause durations
- `mean_duration_score`: mean of duration scores of all pauses in the source and target (or 1, if there are none)
- `mean_alignment_score`: mean of alignment scores of all pauses in the source and target (or 1, if there are none)
- `mean_joint_score`: mean of products of duration and alignment scores of all pauses in the source and target (or 1, if there are none)
- `wmean_duration_score`, `wmean_alignment_score`, `wmean_joint_score`: their counterparts, averaged with weights equalt to the pause durations

Due to averaging over pooled source and target pauses, the `*_score` columns listed above do not depend on which translation sides are considered source and target.

### The recommended metrics to report

To evaluate **speech rate similarity**, we recommend reporting Spearman correlation of `speech_rate_syllable` for languages
with Latin script, and `speech_rate_char` for other languages (due to our lack of reliable syllabization tools for them).

To evaluate **pause similarity**, we recommend reporting micro-averaged `wmean_joint_score`.
When aggregating these scores on the corpus level, we recommend computing the average `wmean_joint_score`
weighted by the `total weight` column (which is the total duration of pauses in the pair).

## Implementation details

Below, we describe in more details how each step of the pipeline is implemented.

### Forced alignment of words with speech

For evaluation of speech rate and pauses, we rely on forced alignment of audio and its transcription on the word level.

We support 3 different implementations of forced alignment:

| File with implementation      | Description                                                                                                                                                                             | Example config                      | Dependencies                         |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- | ------------------------------------ |
| `ctc_forced_aligner.py`       | Forced decoding with a pretrained ASR model (Wav2Vec2ForCTC)                                                                                                                            | `ctc_wav2vec2-xlsr-multilingual-56` | `transformers`                       |
| `unity2_forced_aligner_f2.py` | [Aligner](https://github.com/facebookresearch/seamless_communication/blob/main/docs/m4t/unity2_aligner_README.md) for the non-autoregressive text-to-unit decoder (UnitY2 architecture) | `fairseq2_nar_t2u_aligner`          | `fairseq2`, `seamless_communication` |

We recommend the `fairseq2_nar_t2u_aligner` algorithm as a more accurate one, and use it by default in the pipeline.
It was used in for evaluation of the `Seamless Expressive` models.

To choose an alignment method when annotating utterances, pass its config name as the `forced_aligner` argument
(e.g. `forced_aligner=fairseq2_nar_t2u_aligner`).

To initialize a forced aligner separately in Python code, you can call `stopes.hub.forced_aligner(config_name)`;
for example, `stopes.hub.forced_aligner("fairseq2_nar_t2u_aligner")`.

### Pause detection

Pause prediction is based on forced alignment: and all "spaces" between words longer than the threshold
(0.1 seconds by default) are considered as pauses.

Optionally, we trim the durations of the detected pauses with a VAD model,
so that only the "real silence" within these intervals counts as the pause duration.
This option is recommended, because forced aligners sometimes detect non-existing pauses.

These detected pauses can be extracted from the simplified `text_with_markup` column
of the output file or from the deserialized `utternace` column.

### Pause alignment

For expressivity evaluation, we want to compare how well pauses are translated.
We can just compare the total duration of pauses in the source and translation,
but this can mislead us if there are several pauses in the source and/or target,
or if they are placed between the wrong words.

This tool tries to take this into account:

0. First, we align the source and target words, using an [awesome-align](https://github.com/neulab/awesome-align) weekly supervised model.
1. We find a correspondence between the pauses on the source and target side; all unmatched pauses get a score of 0.
2. All matched pauses get a `duration_score` between 0 and 1 (shortest-to-longest ratio).
3. All matched pauses get an `alignment_score` between 0 and 1 (proportion of word alignment edges that **do not cross** the pause alignment edge).
4. The matching on step (1) can be chosen as the as the one that maximizes the sum of products of scores in (2) and (3) (potentially with some weights).
5. The aggregate score may be computed as the average of products of scores in (2) and (3), potentially weighted by the duration of the corresponding pauses.

When aggregating these metrics to the utterance level, we set the scores to 1 if there are no pauses in the source and target.
However, such sentence pairs receive a `total_weight` of zero, so they do not participate in weighted micro-averaging of the scores.

# Citation

If you use these tools in your work, please cite the `Seamless` paper:

```
@misc{seamless,
  title={Seamless: Multilingual Expressive and Streaming Speech Translation},
  author={Seamless Communication et al},
  year={2023},
  publisher = {arXiv},
}
```
