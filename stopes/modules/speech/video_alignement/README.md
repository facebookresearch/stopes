# Extracting parallel segments from videos
The pipeline presented here allows to extract parallel audio (disjoint) segments
from publicly available videos that are translated in different languages. For each extracted segment, it also provides several scores that can be used to estimate the final alignment quality (speech/text similarity, [Blaser](https://huggingface.co/facebook/blaser-2.0-qe) scores).

The overall pipeline consists of two sequential steps (the output of the first step serves as input to the second):
* [Audio Segmentation Module](./video_segmentor.py)
* [Segment Alignment Module](./video_segment_aligner.py)


The fist step relies mainly on [Whisper](https://github.com/openai/whisper) models for audios segmentation and transcription.
It also uses
* [SONAR](https://github.com/facebookresearch/SONAR) for speech embeddings
* [LaBSE](https://huggingface.co/sentence-transformers/LaBSE) for text embeddings

The second alignment step additionally uses [Blaser](https://github.com/facebookresearch/SONAR#predicting-sentence-similarity-with-blaser-20-models) models.


# Usage example
To launch video segments alignment pipeline one need to install some extra dependencies :
`pip install -e ".[video_alignment]"`

As a pipeline input, we expect here a source `.csv` file with the following schema :

* `sibling_id` (string) - a video_id that is the same for all languages of the same video
* `lang` (string) - video language in 3 chars code (as for SONAR)
* `audio_path` (string) - path to readable video audio (various formats accepted here)

This source file is supposed be multilingual (for each value of `sibling_id` we expect to have audios in several languages).

The precise methodology for building a such input file is very dependent on the raw sources and is out of this scope.

To run the pipeline inside python code (sequentially) :
```python
from stopes.modules.speech.video_alignement.video_segment_aligner import (
    LocalAlignmentConfig,
    LocalAlignmentModule,
)
from stopes.modules.speech.video_alignement.video_segmentor import (
    WhisperSegmentorConfig,
    WhisperSegmentorModule,
)
output_dir = "path/to/your/output/dir"
test_config = WhisperSegmentorConfig(
    shards=".../video_source.csv",
    output_dir=output_dir,
    parquet_file_name="segments",
    whisper_model="large-v2",
    text_model="LaBSE",
    batch_size_=5,
    segment_padding=0.03,
    min_segment_length=0.05,
    max_segment_length=20,
)
wsm = WhisperSegmentorModule(test_config)
for ch in wsm.array():
    wsm.run(ch)

alignment_config = LocalAlignmentConfig(
    output_dir=output_dir + "/aligned_segments",
    dataset_root=output_dir + "/segments",
    min_duration_ratio=0.2,
    min_text_sim=0.4,
    min_speech_sim=0.5,
    min_blaser_score=3.0,
    speech_sim_weight=1,
    text_sim_weight=2,
    sentence_force=0.1,
    small_segment_boost=-10,
)
lam = LocalAlignmentModule(alignment_config)
for ch in lam.array():
    lam.run(ch)
```
Please refer to the documentation within the classes to get full details about the configuration parameters and output formats.

Example of launch via the command line (as StopesModule):
* first 
```bash
python -m stopes.modules \
        +speech_preproc=video_segmentation \
        speech_preproc.shards=path/to/my/video_source.csv \
        speech_preproc.output_dir=/my/output/dir/ \
        launcher=submitit
```
* next (once all first step segmentation is done)
```bash
python -m stopes.modules \
    +speech_preproc=video_segment_alignement \
    speech_preproc.dataset_root=/my/output/dir/segments/ \
    speech_preproc.output_dir=/my/output/dir/aligned/ \
    launcher=submitit
```

# Citation

If you use this pipeline in your work, please cite the `Seamless` paper
(specifically described in Section 4.1.4):

```
@misc{seamless,
  title={Seamless: Multilingual Expressive and Streaming Speech Translation},
  author={Seamless Communication et al},
  year={2023},
  publisher = {arXiv},
}