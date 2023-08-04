---
sidebar_position: 5
---

# Caching/Memoization

An important part of the launcher is its caching system. When you call the
schedule method with a configured module, the launcher will check if this
configuration was already run in the past and reuse the results when possible.
The cache is indexed on the configuration of the module, so if you change
anything in the configuration input, the module will be executed from scratch
and the new result will be cached with a different key. It's also important to
remember that all inputs to the module that could change its results (and thus
the caching) should be specified in the config input.

If you change the code of your module to a point that would change its output,
you can implement the `version() `method to return a new value so that the cache
knows that it needs to recompute from scratch even from known configs.

You can also implement the` validate() `method to check the outputs from your
module and from the cache if you want to actively invalidate the cache. For
example, if it’s known how many lines are to be embedded into a particular
dimension (say 1024), you can validate that the output file size is e.g.
`num_lines * 1024 * float32.`

Here is an example of rerunning the global mining pipeline that was interrupted
in the middle. The caching layer recovers what was already executed
successfully. This was started with the same command that would require a full
run:
```bash
python yourpipeline.py src_lang=bn tgt_lang=hi +data=ccg
```

Here are the logs:
```
[global_mining][INFO] - output: .../global_mining/outputs/2021-11-02/08-56-40
[global_mining][INFO] - working dir: .../global_mining/outputs/2021-11-02/08-56-40
[mining_utils][WARNING] - No mapping for lang bn
[embed_text][INFO] - Number of shards: 55
[embed_text][INFO] - Embed bn (hi), 55 files
[stopes_launcher][INFO] - for encode.bn.55 found 55 already cached array results,0 left to compute out of 55
[train_faiss_index][INFO] - lang=bn, sents=135573728, required=40000000, index type=OPQ64,IVF65536,PQ64
[stopes_launcher][INFO] - index-train.bn.iteration_2 done from cache
[stopes_launcher][INFO] - for populate_index.OPQ64,IVF65536,PQ64.bn found 44 already cached array results,11 left to compute out of 55
[stopes_launcher][INFO] - submitted job array for populate_index.OPQ64,IVF65536,PQ64.bn: ['48535900_0', '48535900_1', '48535900_2', '48535900_3', '48535900_4', '48535900_5', '48535900_6', '48535900_7', '48535900_8', '48535900_9', '48535900_10']
[mining_utils][WARNING] - No mapping for lang hi
[embed_text][INFO] - Number of shards: 55
[embed_text][INFO] - Embed hi (hi), 55 files
[stopes_launcher][INFO] - for encode.hi.55 found 55 already cached array results,0 left to compute out of 55
[train_faiss_index][INFO] - lang=hi, sents=162844151, required=40000000, index type=OPQ64,IVF65536,PQ64
```

We can see that the launcher has found out that it doesn't need to run the
encode and train index steps for the `bn` lang (src language) and can skip
straight to populating the index with embeddings, but it also already processed
44 shards for that step, so will only re-schedule jobs for 11 shards. In
parallel, it is also processing the tgt language (`hi`) and found that it still
needs to run the index training step as it also recoverred all the encoded
shards.

All this was done automatically. The person launching the pipeline doesn't have
to micromanage what has already succeeded and what needs to be started when.
