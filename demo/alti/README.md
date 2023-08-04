# ALTI+

ALTI+ is a tool for inspecting token contributions in a transformer encoder-decoder model.
It might be useful for detecting hallucinated translations or undertranslations.
Our implementation is based on the code from the paper [Ferrando et al., 2022](https://arxiv.org/abs/2205.11631).
The code and readme for it are located at `stopes/eval/alti`.

# Installation
To use the core ALTI+ code, you need to install Stopes with the `alti` dependencies:
```
git clone https://github.com/facebookresearch/stopes.git
cd stopes && pip install -e '.[alti]' && cd ..
```

You will also need Fairseq. To work with [NLLB models](https://github.com/facebookresearch/fairseq/tree/nllb), 
you have to checkout the corresponding branch:
```
git clone https://github.com/pytorch/fairseq
cd fairseq && git checkout nllb && pip install -e . && python setup.py build_ext --inplace && cd ..
```

# The minimal example (CLI with an NLLB model)
The code, configs and toy data for this example is in the `minimal_example` directory.

To download the official 600M checkpoint of the [NLLB-200 model](https://github.com/facebookresearch/fairseq/tree/nllb),
run the script `download_nllb.sh` from the `minimal_example` directory; 
it will create the `nllb` directory there and download the model and the dictiory into it.

The following command will read sentence pairs `test_input.tsv` and compute ALTI and ALTI-based metrics for them.

It will output various text-level scores into `test_output.tsv`, 
and token-level contributions and alignments will be stored in `test_output_alignments.jsonl`.

The file `test_input.tsv` contains columns `src` and `mt` with a few French-English translation pairs with different pathologies:
```
src     mt
Traduction normale.     A normal translation.
Traduction incomplète, dans laquelle une partie de l'entrée est ignorée.        An incomplete translation.
Traduction avec une hallucination.      A translation with a hallucination that actually references itself.
Cette entrée sera ignorée.      Here is an example of a full hallucination.
Traduction dans laquelle je ne me suis pas trompé.      A translation in which I have not made an error.
Traduction dans laquelle j'ai fait une erreur.  A transfer in which I have accessed an error.
Traduction avec hallucination cyclique. A translation with with with with with with a cyclical hallucination.
```

To analyze these sentence pairs, run the following command:

```bash
python compute_nllb_alti.py \
    input_filename=$(pwd)/test_input.tsv \
    metrics_filename=$(pwd)/test_output.tsv \
    alignment_filename=$(pwd)/test_output_alignments.jsonl \
    src_lang=fra_Latn \
    tgt_lang=eng_Latn \
    +preset=nllb_demo +demo_dir=$(pwd)/nllb
```
Some arguments to the command are stored in the configuration file: `preset/nllb_demo.yaml`.
You can edit this file or create another preset, if you want.

The file `test_output.tsv` contains multiple columns with various ALTI-based metrics 
(here we show only a few columns for better readability):

```
                                                                      mt  avg_sc  min_sc  top_sc_mean  src_sum_contr_below_01
0                                                  A normal translation.    0.69    0.56         0.24                    0.00
1                                             An incomplete translation.    0.73    0.40         0.17                    0.25
2    A translation with a hallucination that actually references itself.    0.52    0.22         0.13                    0.00
3                            Here is an example of a full hallucination.    0.41    0.08         0.12                    0.00
4                       A translation in which I have not made an error.    0.65    0.45         0.15                    0.00
5                          A transfer in which I have accessed an error.    0.61    0.40         0.14                    0.00
6  A translation with with with with with with a cyclical hallucination.    0.61    0.28         0.13                    0.00
```
One can see that the metrics `avg_sc`, `min_sc` and `top_sc_mean` may help to detect hallucinations, 
whereas `src_sum_contr_below_01` indicates incomplete translations.


The file `test_output_alignments.jsonl` contains individual subword tokens of the source and target sentences, 
the raw ALTI+ contribution matrices produced for these tokens, and the alignments computed from these matrices:
```
{"contributions": [[0.52, 0.06, ...]], "alignment": [[0, 0], [0, 4], [0, 6], [1, 0], ...], "src_toks": ["__fra_Latn__", "▁Trad", "uction", "▁normale", ".", "</s>"], "tgt_toks": ["</s>", "__eng_Latn__", "▁A", "▁normal", "▁translation", "."], "pred_toks": ["__eng_Latn__", "▁A", "▁normal", "▁translation", ".", "</s>"]}
{"contributions": ...
...
```
`src_toks` are the encoder inputs, `tgt_toks` are the inputs to the decoder, whereas `pred_toks` are its outputs.
In fact, `tgt_toks` are `pred_toks` shifted by one position.


# Reproducing the hallucination detection experiments
The folder `detecting_hallucinations` contains the code for reproducing the experiments on hallucination detection
from the paper [Detecting and Mitigating Hallucinations in Machine Translation: Model Internal Workings Alone Do Well, Sentence Similarity Even Better](https://arxiv.org/abs/2212.08597).

The detailed instructions for reproduction are in that folder.

To refer to these results, please cite:


```bibtex
@article{dale2022detecting,
    title={Detecting and Mitigating Hallucinations in Machine Translation: Model Internal Workings Alone Do Well, Sentence Similarity Even Better},
    author={Dale, David and Voita, Elena and Barrault, Lo{\"\i}c and Costa-juss{\`a}, Marta R},
    journal={arXiv preprint arXiv:2212.08597},
    url={https://arxiv.org/abs/2212.08597},
    year={2022}
}
```

