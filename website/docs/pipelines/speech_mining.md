---
sidebar_position: 1
---

# Speech Mining Pipeline

With the [Seamless Communication project](https://github.com/facebookresearch/seamless_communication), FAIR has introduced a new mechanism for speech mining. In the `stopesV1`, you could mine large text datasets to create aligned text accross languages. This was useful to train machine translation algorithms. From `stopesV2`, we introduce a mecanism that lets you mine speech and text together accross languages to create aligned multimodal datasets for training and evaluating speech tasks. This mining is based on the [SONAR multimodal/multilingual embedding space](https://github.com/facebookresearch/SONAR).

## Installation

Speech mining requires the installation of [fairseq2](https://github.com/facebookresearch/fairseq2) and [SONAR](https://github.com/facebookresearch/SONAR). You can install these with:

```
pip install 'stopes[speech,mining,speech_mining]'
```

or if you are installing a local checkout of this repository:

```
pip install '.[speech,mining,speech_mining]'
```

If fairseq2 does not provide a build for your hardware, see the installation instructions in the fairseq2 documentation to build the package for your own machine.

## Configuration

The speech mining pipeline is an extension of the [global mining](https://facebookresearch.github.io/stopes/docs/pipelines/global_mining) pipeline. We recommend reading that page first to understand the base configuration for text to text mining.

You can decide to mine text-text, speech-text, text-speech or speech-speech with the SONAR space encoders, you can do this withing language (e.g. to mine aligned text-speech in English), or accross languages (e.g. to mine aligned speech-speech between Catalan and Korean). To do this, you need to configure the type of encoder that you want to use for the data you are feeding the global mining pipeline. You will want to set different language configurations in your mining `yaml` preset. While the configurations are called language, you might end up creating multiple entries for the same language, but with different modalities.

For example, if you wanted to mine text-speech within French, you will add something like:

```
lang_configs:
  frA:
    data:
      data_version: 23H1RCLSP
      iteration: 1
      data_shard_dir: /path/tothe/rawaudio/dataset
      shard_type: speech
      bname: speech
      shard_list: null
      shard_glob: ${.data_shard_dir}/speech/*.ogg
      nl_file_template: "{lang}.nl"

    embed_text:
      preprocess: null
      encoder:
        _target_: stopes.modules.preprocess.mining_speech_encoder.Sonar2MiningSpeechEncoder
        encoder_model: NAME_OF_SONAR_ENCODER_MODEL
        _name: sonar2_speech_encoder
        spm_model: null # unused
        spm_vocab: null # unused
        mini_batch_size: null
        fp16: true
        gpu: true
        num_processes: 4
  fr:
    data:
      data_version: 23H1RCL
      iteration: 1
      data_shard_dir: /path/tothe/rawtext/dataset
      shard_type: text
      bname: text
      shard_list: null
      shard_glob: ${.data_shard_dir}/text/text.txt
      meta_glob: ${.data_shard_dir}/text/meta.txt
      nl_file_template: "{lang}.nl"

    embed_text:
      encoder:
        _target_: stopes.modules.preprocess.sonar_sentence_encoder.SonarTextEncoder
        _name: NAME_OF_SONAR_ENCODER_MODEL
        spm_model: null # unused
        spm_vocab: null # unused
```

Note that the audio dataset shards are text files containing segmentation information about your raw audio. This follows the format:

```
audio_file_name start_timestamp end_timestamp batch_no
```

where `start/end` timestamps are timestamps of a speech segment within the audio file. `batch_no` is a batching number for this segment. You can use it to batch segments of similar length together for faster embedding, or just leave it set to 0. 

This means that you need to run the speech segmentation separately. There are many ways to segment audio, so we do not discuss this here.

In this sample config, we have set a text data source `fr` and an audio data source `frA`. The example assumes they are the same language, but they could be different languages, or they could both be audio or text. The audio data source uses a SONAR speech encoder while the text source uses the text encoder. Make sure to refer to the SONAR model cards to choose the appropriate encoder model to specify in each config entry. You will need to set `encoder_model` or `_name` to the name of the SONAR model card. fairseq2 will then download the model for you if needed.

The rest of the mining pipeline is similar to the one described on the global mining page, please see the description there for run examples.