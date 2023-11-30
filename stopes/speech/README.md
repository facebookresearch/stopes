# Stopes Speech

The speech package consists of Python modules (not to confuse with the term "module" in `StopesModule` objects, which are under `stopes.modules`) that are dedicated to speech processing, modelling and evaluation, and can be run as standalone APIs. Currently the following speech modules are available:

## Speech segments data format

In Seamless, current speech data has TSV format, where the main audio columns (in one language) consists of path to the audio
files, as well as the position of the segments. The general format of the column is:

```
<audio file path> [SEP <start> SEP <number> SEP [<sampling factor or optional parameters>]]
```

Where `SEP` is the inner separator in one column. Depending on the value of `SEP`, `start` and `end` have different meanings:

- `SEP` = "|" : `start` refers to the start time and `number` referes to the end time of the segment in the audio file, in miliseconds. In this case `sampling factor` corresponds to the sampling rate of the audio in the factor of 1000, e.g. "16" means sampling rate = 16000, etc.
- `SEP` = ":" : `start` refers to the offset byte and `number` refers to the number of bytes of the segment in the audio file. In this case the sample rate will be derived from the metadata of the audio file, _rounding to 1000_ (e.g. if the raw file sample rate is 16001, the segment will be read with the sample rate 16000).
- `SEP` = " " : `start` refers to the start frame and `number` referes to the end frame of the segment in the audio file. In this case the sample rate is always set to 16000.

Note that all `SEPS`, `start` and `number` are optional. The audio file can contain one path per line - in this case we assume the audios are already aligned and each line corresponds to an audio segment / sentence.

## Speech Tokenizer:

Speech tokenizer is a set of Python modules that take speech audios in waveforms and translate them into discrete units ("tokens") trained by some speech encoders (such as the encoders in Encodec, Hubert). These units can also be translated back ("decoded") to the original waveforms. In particular, the interface `SpeechTokenizer`provides major functions:

- `SpeechTokenizer.encode()` : Convert audio input (raw wave form features) to discrete units
- `SpeechTokenizer.decode()`: Convert units to raw wave forms<>

In this perspective, each unit is equivalent to a "vocal" token, that can be decoded back to audio, in a similar fashion to how textual tokens are decoded back to text. Under the hood, the `encode()` is a chained execution of two customizable functions (with some sanity checks):

- `SpeechTokenizer.extract_features()` : This uses e.g. a speech encoder such as wave2vec to convert audio raw wave forms to some embedding vectors.
- `SpeechTokenizer.to_units()` : This converts the embedding vectors into discrete units from a vocabulary (kmeans centroids).

Each of the above functions accept a tensor (Torch or numpy arrays) and return another tensor.

### Pretrained tokenizers:

The Speech tokenizer module comes with a set of prepackaged encoders, unit conversions and vocoders that are compatibled with each other (i.e. in terms of dimension, data used to encode/decode, quatization level, etc.), and are bundled together. These packaged combinations are available under some name (called a _pretrain tokenizer_) and accessible via the `speech_tokenizer(NAME)` API:

```python
# Alternatively: stopes.AutoSpeech.speech_tokenizer()
tokenizer = stopes.hub.speech_tokenizer(NAME)

units = tokenizer.encode(original_data)
....
resynth_data = tokenizer.decode(units)
```

For example, we can use the [encodec model](https://github.com/facebookresearch/encodec) to provide the units for an audio:

```python
tokenizer = speech_tokenizer("encodec_24khz")
```

Other pretrained models (HuBERT, X-LSR) are to be released soon under particular licenses.
