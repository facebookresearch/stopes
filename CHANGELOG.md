# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 1.1.0

### Added

- speech mining with [SpeechMatrix](https://github.com/facebookresearch/fairseq/blob/ust/examples/speech_matrix/speech_laser_encoders.md)
- [ALTI+](https://facebookresearch.github.io/stopes/docs/eval/alti)
- [BLASER](https://facebookresearch.github.io/stopes/docs/eval/blaser)
- many tests for the mining pipeline and different modules of `stopes`
- the `Launcher` can now retry jobs when running on a flaky slurm cluster
- different margin implementations in mining
- possibility to take the best neighbour when running the margin instead of the first one (fast)
- mine large datasets by splitting them in sub-languages
- when mining, keep metadata about what pairs come from the forward and backward pass
- when mining, choose if you want to do only forward, backward or both passes



### Changed

- embeddings for mining are now stored in real npy files with headers
- `StopesModule` is not `async` anymore, just the APIs of `Launcher`. You should write your `run` function as
a normal non-async function
- mining neighbours is now optimized to have a smaller memory load
- progress bar of pipelines is simplified to avoid overly busy logs
- do not rely on existing line count files and compute them as part of the pipeline in the mining


### Fixed

- many improvements in the mining code
- many fixes in the NMT eval pipeline


## 1.0.0

Initial release
