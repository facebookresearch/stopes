# AutoPCP

Here is the code to train and apply models that compare two audios.
We apply them primarily to compare prosody of two utterances in different languages.
The model uses audio embeddings from a pretrained audio encoder
and a trainable MLP regressor on top of them.

## Usage

To prepare the environment, please first install the `auto_pcp` subset of requirements of `stopes`.

```
pip install -e '.[auto_pcp]'
```

The recommended entrypoint is a Stopes module: [stopes.modules.evaluation.compare_audio_module](../../modules/evaluation/compare_audio_module.py)
that reads a tsv file with paths to source and target audios, loads the audio encoder and comparator model,
produces the similarity scores, and saves them to another `tsv` file.

For example, if you have a file `input.tsv` with columns `src_audio` and `tgt_audio` that contain paths to the audio pair files,
you can compute the similarity scores for them using the Stopes launcher:

```
python -m stopes.modules +compare_audios=AutoPCP_multilingual_v2 \
    +compare_audios.input_file=input.tsv \
    +compare_audios.src_audio_column=src_audio \
    +compare_audios.tgt_audio_column=tgt_audio \
    +compare_audios.named_columns=true \
    +compare_audios.output_file=output.txt
```

The file `output.txt` will contain a single column with predicted similarity scores as float numbers.
If the model was trained with prosodic consistency protocol (which is the main expected use),
the scores around 4 will represent a high similarity, while very dissimilar pairs will have a score close to 1.

This call uses the `AutoPCP_multilingual_v2` config that currently resides in [stopes/pipelines/bitext/conf/compare_audios](../../pipelines/bitext/conf/compare_audios).

## Parameters

Here are a few of them which you could override (and likely need to):

- `input_file` and `output_file`: paths to the input and output .tsv files.
- `src_audio_column`, `tgt_audio_column`: can be ints (if the source .tsv file has no headers).
  or strings (only if it has column names, which is indicated by the `named_columns` flag).
  By default, they equal 1 and 2, assuming that source and target audio paths are in the 2nd and 3rd columns.
- `num_process`: you can increase it if you believe that for your data, loading audios from disk is the bottleneck.
- `batch_size`: you can increase it if you believe that your GPU can process more audios at once (or set to 1 if you use CPU).
- `symmetrize`: if this is set to `True` (which is the default in the `AutoPCP_multilingual_v2` config),
  the predicted scores will be forced to be symmetric w.r.t. the source and target audios.
- `comparator_path`, `encoder_path` and `pick_layer`: normally, you don't need to modify these parameters,
  as they are tied to the particular AutoPCP model;
  please use the the whole bundle from one of the configs in `stopes/pipelines/bitext/conf/compare_audios`.

To understand the parameters in more details, please inspect the source code.

## Notes

- The default config, `AutoPCP_multilingual_v2`, downloads and unpacks the `AutoPCP-multilingual-v2` comparator model from [here](https://dl.fbaipublicfiles.com/speech_expressivity_evaluation/AutoPCP-multilingual-v2.zip).
This comparator was used to evaluate the expressive speech translation models in the `Seamless` paper.
-  The `v1` version of this model was used in creation of the `SeamlessAlignExpressive` dataset. It is not publicly released.

# Citation

If you use AutoPCP in your work, please cite the `Seamless` paper:

```
@misc{seamless,
  title={Seamless: Multilingual Expressive and Streaming Speech Translation},
  author={Seamless Communication et al},
  year={2023},
  publisher = {arXiv},
}
```
