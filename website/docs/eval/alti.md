# ALTI+

ALTI+ is a tool for inspecting token contributions in a transformer encoder-decoder model.
It might be useful for detecting hallucinated translations or undertranslations.

This repository is based on the code from the paper [Ferrando et al., 2022](https://arxiv.org/abs/2205.11631).
The original code is located at https://github.com/mt-upc/transformer-contributions-nmt.
It is licensed under the Apache 2.0 license included in the current directory.

We have made a few adaptation to the code so that it can run with the dense NLLB-200 models.
The code in this directory is licensed both under the Apache 2.0 license of the original code (in the current directory),
and under the MIT license of the whole project (in the parent directory).

# Usage
An instruction for setting up the environment and computing ALTI+ token contributions from an NLLB model
with a command line interface is present in the folder `demo/alti`.

Below is another example, that uses a bilingual model and the Python interface.
Here is how you can run it:

1. Prepare the environment by installing Fairseq and Stopes:
```
pip install fairseq==0.12.1
git clone https://github.com/facebookresearch/stopes.git
cd stopes
pip install -e '.[alti]'
```

2. Download the model and dictionary from https://github.com/deep-spin/hallucinations-in-nmt:
    - model: https://www.mediafire.com/file/mp5oim9hqgcy8fb/checkpoint_best.tar.xz/file
    - data: https://www.mediafire.com/file/jfl7y6yu7jqwwhv/wmt18_de-en.tar.xz/file
3. Run the following commands to unpack the data:
```tar -xvf checkpoint_best.tar.xz && tar -xvf wmt18_de-en.tar.xz```
4. Run the following command to download the tokenizers:
```
wget https://github.com/deep-spin/hallucinations-in-nmt/raw/main/sentencepiece_models/sentencepiece.joint.bpe.model
wget https://github.com/deep-spin/hallucinations-in-nmt/raw/main/sentencepiece_models/sentencepiece.joint.bpe.vocab
```
Now you can run the following Python code to look at the ALTI analysis:


```Python
from stopes.eval.alti.wrappers.transformer_wrapper import FairseqTransformerHub
from stopes.eval.alti.alti_metrics.alti_metrics_utils import compute_alti_nllb, compute_alti_metrics

# load the model, vocabulary and the sentencepiece tokenizer
hub = FairseqTransformerHub.from_pretrained(
    checkpoint_dir='.',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='wmt18_de-en',
    bpe='sentencepiece',
    sentencepiece_model='sentencepiece.joint.bpe.model',
)

# translate an example of a German sentence to English.
# the source sentence means "The breakfast buffet is very good and varied.", so the translation is wrong.
src = 'Frühstückbüffet ist sehr gut und vielseitig.'
tgt = hub.translate(src)
print(tgt)  # The staff were very friendly and helpful.

# compute the token contributions for this translation pair
attributions, src_tok, tgt_tok, pred_tok = compute_alti_nllb(hub, src, tgt)
# attributions is a 2d numpy array, and src/tgt/pred_tok are lists of subword strings
print(attributions.shape, len(src_tok), len(tgt_tok), len(pred_tok))  # (9, 21) 12 9 9
print(src_tok)  # ['▁Frühstück', 'bü', 'ff', 'et', '▁ist', '▁sehr', '▁gut', '▁und', '▁vielseit', 'ig', '.', '</s>']
print(pred_tok)  # ['▁The', '▁staff', '▁were', '▁very', '▁friendly', '▁and', '▁helpful', '.', '</s>']

# compute 18 different metrics based on the ALTI+ matrix.
# 'avg_sc' is average source contribution, and the value of 0.4 is not very high (we expect about 0.5 or more).
metrics = compute_alti_metrics(attributions, src_tok, tgt_tok, pred_tok)
print(len(metrics))  # 18
print(metrics['avg_sc'])  # 0.40330514

# for a correct translation, average source contribution is slightly higher
tgt2 = "The breakfast buffet is very good and diverse."
print(compute_alti_metrics(*compute_alti_nllb(hub, src, tgt2))['avg_sc'])  # 0.47343665
```

# Citation
If you use ALTI+ in your work, please consider citing:
```bibtex
@inproceedings{alti_plus,
    title = {Towards Opening the Black Box of Neural Machine Translation: Source and Target Interpretations of the Transformer},
    author = {Ferrando, Javier and Gállego, Gerard I. and Alastruey, Belen and Escolano, Carlos and Costa-jussà, Marta R.},
    booktitle = {Proc of the EMNLP},
    url = {https://preview.aclanthology.org/emnlp-22-ingestion/2022.emnlp-main.599.pdf},
    year = {2022}
}
```
