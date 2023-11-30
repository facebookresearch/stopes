# Vocal style similarity (VSim) evaluation

This is a Stopes wrapper for various models of vocal style similarity.

It is measured as cosine similarity of source and translation embeddings extracted with pretrained speech encoders.
Currently, ECAPA and WavLM models are supported.

## Pre-requisites

This module requires additional dependencies of Stopes, which can be installed by running

```
pip install -e '.[vocal_style_sim]'
```

from the root directory of this repository.

To run it, you need to download a speech encoder model.
We recommend using the last model from the
[UniSpeeh repository](https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification):
WavLM large without fixed pre-train, aka
[wavlm_large_finetune.pth](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view?usp=sharing).

Below, we assume that the model is saved to `$MODEL_PATH`.
We refer to the type of this model as `valle` (as it was used to evaluate the eponymous model);
the other supported model type is `ecapa`.

## Usage

### Command line

You can run vocal style similarity evaluation as a Stopes module.
If you have an input .tsv file `$INPUT_PATH` with paths to the source and target .wav files
in the `src_audio` and `hypo_audio` columns, you can run the module as follows:

```bash
python -m stopes.modules +vocal_style_similarity=base \
    vocal_style_similarity.model_type=valle \
    +vocal_style_similarity.model_path=$MODEL_PATH \
    +vocal_style_similarity.input_file=$INPUT_PATH \
    +vocal_style_similarity.output_file=$RESULT_PATH \
    vocal_style_similarity.named_columns=true \
    vocal_style_similarity.src_audio_column=src_audio \
    vocal_style_similarity.tgt_audio_column=hypo_audio
```

This command will create the `$RESULT_PATH` text file, with one similarity score (a value between -1 and 1) per line.

### Python

Alternatively, you can call the vocal style similarity tools from your Pyton code.
Assuming that you have lists of paths to the source and target audiofiles in `src_audio_paths` and `tgt_audio_paths` respectively,
you can run:

```Python
from stopes.eval.vocal_style_similarity.vocal_style_sim_tool import get_embedder, compute_cosine_similarity
embedder = get_embedder(model_name="valle", model_path=MODEL_PATH)
src_embs = wavlm_embedder(src_audio_paths)
tgt_embs = wavlm_embedder(tgt_audio_paths)
similarities = compute_cosine_similarity(src_embs, tgt_embs)
```

The resulting list `similarities` will contain the similarity scores between each source-target audio pair.
