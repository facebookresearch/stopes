---
sidebar_position: 3
---

# Composition (aka pipelining)

The StopesModule framework provides a "launcher" abstraction that takes care of
scheduling your module "somewhere". Currently, and in most Stopes use cases, this
somewhere is SLURM, but you can also choose to launch it locally and more
launcher implementations might come when other execution environments are needed.

The global_mining
pipeline is a good example of how all of this works together and you should
check it out when reading this doc to have a good idea of how things fit
together.

You can initialize a launcher from code with its python init, but ideally, your
pipeline will initialize it from a config with hydra:


```python
self.launcher = hydra.utils.instantiate(config.launcher)
```


We provide pre-made configs for the main SLURM launcher and instantiating the
launcher from config will allow you to override it from the CLI for debugging.

Once you have a launcher, you can launch a module in code with:


```python
embedded_files = await self.launcher.schedule(embed_module)
```


The launcher will take care of submitting a job to the execution engine (e.g.
SLURM) and wait for it to be done. The launcher will also take care of raising
any exception happening in the execution engine and if using the submitit
launcher, it will also take care of checkpointing (see above).

## Asyncio

Because` launcher.schedule `will potentially schedule your module run method on
a separate host, wait for it to find a slot and to eventually finish. The result
that this `schedule` method returns is not available immediately. We use python
asyncio to deal with waiting for the results to be available. This means that
you need to `await `the result of schedule before being able to use it.

This also means that you can use asyncio helpers to organize your code and tell
the launcher when things can be scheduled in parallel. For instance you can
await for two results in "parallel" with:


```python
src_embeddings, tgt_embeddings = await asyncio.gather(
     launcher.schedule(
          StopesModule.build(self.config.embed_text, lang="bn")
     ),
     launcher.schedule(
          StopesModule.build(self.config.embed_text, lang="hi")
     ),
 )
```
