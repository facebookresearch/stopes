# Stopes Evaluation

This directory contains a collection of tools used for evaluation in NLLB, Seamless, and related projects.

Most of the tools have a detailed readme file in their own directory; here we provide their general outline.

## Speech expressivity evaluation tools

These tools were introduced in the `Seamless` paper to evaluate the `SeamlessExpressive` models
(read [here](https://github.com/facebookresearch/seamless_communication/tree/main/docs/expressive) about reproducing the evaluation).

- [auto_pcp](auto_pcp): AutoPCP, neural models to predict human judgements of prosodic similarity of two utterances.
- [local_prosody](local_prosody): A toolkit for interpretable comparison of individual aspects of prosody (speech rate and pauses).
- [vocal_style_similarity](vocal_style_similarity): evaluation of Vocal style similarity (VSim).

To use these tools, please install Stopes with their corresponding sets of requirements, by running (from the repository root):

```
pip install -e '.[auto_pcp,local_prosody,vocal_style_sim]'
```

To run `local_prosody` with the recommended UnitY2 forced aligner, you will also need to install the [seamless_communication](https://github.com/facebookresearch/seamless_communication) package and its dependency, [fairseq2](https://github.com/facebookresearch/fairseq2).

## Semantic evaluation tools

- [alti](alti): Our implementation of ALTI+, an algorithm for computing token attributions in machine translation.
- [blaser](blaser): BLASER and BLASER 2.0, evaluation metrics of translation accuracy for speech and text.
- [toxicity](toxicity): A re-implementation of [ETOX](../../demo/toxicity-alti-hb/ETOX), wordlist-based multilingual detector of toxicity in text.
- [word_alignment](word_alignment): Tools for aligning words in a translation sentence pair.
