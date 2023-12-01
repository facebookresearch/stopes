---
sidebar_position: 1
---

# Speech Mining Pipeline

With the [Seamless Communication project](https://github.com/facebookresearch/seamless_communication), FAIR has introduced a new mechanism for speech mining. In the `stopesV1`, you could mine large text datasets to create aligned text accross languages. This was useful to train machine translation algorithms. From `stopesV2` onwards, we introduce a mechanism that lets you mine speech and text together accross languages to create aligned multimodal datasets for training and evaluating speech tasks. This mining is based on the [SONAR multimodal/multilingual embedding space](https://github.com/facebookresearch/SONAR).

## Installation

Speech mining requires the installation of [fairseq2](https://github.com/facebookresearch/fairseq2) and [SONAR](https://github.com/facebookresearch/SONAR). You can install these with:

```
pip install 'stopes[speech,mining,sonar_mining]'
```

or if you are installing a local checkout of this repository:

```
pip install '.[speech,mining,sonar_mining]'
```

The above message should install SONAR and fairseq2 with their default settings. If you want to install fairseq2 with custom support for your hardware (i.e. GPU suppport with different CUDA versions), see the installation instructions in the [fairseq2](https://github.com/facebookresearch/fairseq2) documentation to build the package for your own machine.

## Preset configuration

The speech mining pipeline is an extension of the [global mining](https://facebookresearch.github.io/stopes/docs/pipelines/global_mining) pipeline. We recommend reading that page first to understand the base configuration for text to text mining.

While in the global mining, you run one encoder for all laguages, in speech mining, you can configure different encoders of different modalities (text or speech), depending on whether you want to mine text-text, speech-text, text-speech or speech-speech with the SONAR space encoders. You can do this within a language (e.g. to mine aligned text-speech in English), or accross languages (e.g. to mine aligned speech-speech between Catalan and Korean).

More specifically, in speech mining, we use a preset configuration to set up the embedding modules for each language under the `lang_configs`. Note that in your configuration, `lang_configs` might end up having multiple entries for the same language, but with different modalities.

Below is an example if you want to mine text-speech within French (i.e. aligning between French audios and text). You can just replace the `lang_configs` section in `stopes/pipelines/bitext/conf/preset/demo.yaml` with the following setting:

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

    embed_speech:
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

In this sample config, we have set a text data source `fr` and an audio data source `frA`. The example assumes they are the same language, but they could be different languages, or they could both be audio or text. The audio data source uses a SONAR speech encoder while the text source uses the text encoder. Make sure to refer to the SONAR model cards to choose the appropriate encoder model to specify in each config entry. Replace "NAME_OF_SONAR_ENCODER_MODEL" in the config above with the name of the SONAR model card (You can find the list of available model card in the [SONAR Github repository](https://github.com/facebookresearch/SONAR/tree/main/sonar/cards)).

## Run the mining pipeline

The above setting is analog to the `embed_text` step in the [global mining](https://facebookresearch.github.io/stopes/docs/pipelines/global_mining). The different here is that we can use either `embed_speech` or `embed_text` to encode the data of different modalities. The rest of the mining pipeline is similar to the one described on the global mining page.

We provide the example preset in `stopes/pipelines/bitext/conf/preset/demo_speechmine.yaml`, so you can simply run:

```bash
python -m stopes.pipelines.bitext.global_mining_pipeline src_lang=frA tgt_lang=fr demo_text_dir=.../stopes-repo/demo demo_audio_dir=[YOUR AUDIO DIR] +preset=demo_speechmine output_dir=.
```

## Audio data format and segmentation

In order to run the above example, you would need to provide your own audio data. Note that the audio dataset shards are text files containing segmentation information about your raw audio. This follows the format:

```
audio_file_name start_timestamp end_timestamp batch_no
```

where `start/end` timestamps are timestamps of a speech segment within the audio file. `batch_no` is a batching number for this segment. You can use it to batch segments of similar length together for faster embedding, or just leave it set to 0.

This means that you need to run the speech segmentation separately. There are many ways to segment audio, one of them is to use our internal VADSegmentation, found in `stopes.modules.speech.vad_segment_audio.VADSegmentAudioModule`. Below is the example of the updated `lang_configs` for the audio data with segmentation option:

```
lang_configs:
  frA:
    data:
      data_version: 23H1RCLSP
      iteration: 1
      data_shard_dir: /path/tothe/segmented_dataset
      shard_type: speech
      bname: speech
      shard_list: null
      shard_glob: ${.data_shard_dir}/speech/*.ogg
      nl_file_template: "{lang}.nl"

    segment_audio:
      _target_: stopes.modules.speech.vad_segment_audio.VADSegmentAudioModule
      lang: fr
      shards: /path/tothe/rawaudio/dataset
      max_duration_in_seconds: null
      output_dir: /path/tothe/segmented_dataset
      model: [MODEL_TO_SILERO_VAD]
      hard_limit_min_length: 1.0

    embed_speech:
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
```

Where the `MODEL_TO_SILERO_VAD` point to the silero VAD checkpoint found in e.g. [Silero models hub](https://github.com/snakers4/silero-models).
