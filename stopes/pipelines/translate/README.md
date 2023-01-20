# Large-scale translation pipeline

This pipeline is intended for translation jobs involving large amounts of text,
typically with sentence counts in the hundreds of thousands. Given a model and all the
necessary information for preparing its inputs, the pipeline takes care of sharding the
data and, where applicable, scheduling parallel jobs for binarizing and translating it.

The prototypical use case for this pipeline is backtranslation, which often involves
translating tens of millions of sentences. It can also be useful for evaluating the
performance of an MT system on large evaluation sets, as running this in parallel can
often significantly speed up the process.

An example configuration is provided in `conf/example.yaml` showing how two corpora,
`monolingual.arb_Arab.txt` and `monolingual.ajp_Arab.txt`, can be each translated into
multiple other languages. Given the above configuration, the pipeline will take care of
sharding corpora in chunks smaller than 60Kb (as specified by the `max_size_per_shard`
option).

For the generation options, please refer to the following config dataclass:
`stopes.modules.translation.fairseq_generate.FairseqGenerateConfig`.
