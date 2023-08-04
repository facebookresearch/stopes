---
sidebar_position: 1
---

# stopes Module Framework

The `stopes` library was built for easily managing complex pipelines without
worrying about scaling and reliability code.

## Key features:

- **Reproducibility.** `stopes` is built with a research mindset first. The
underlying Hydra framework gives you full control over the configuration of your
pipelines. All the important parameters of your experiments can be defined and
tracked.
- **Easier scaling.** The `stopes` framework provides clean separation between
your pipeline step logic and the scaling code. If you use slurm, run locally or
want to deploy on another cluster, your pipeline code and steps shouldn't
change.
- **Caching/memoization.** With `stopes`, you can iterate faster and more reliably
via transparent memoization. We've built the library so your code doesn't need
to know what's happening with the cache
- **Composition.** The `stopes` API surface is minimum, so you can build a
pipeline by simply writing idiomatic python (using asyncio) and have a quick
understanding of what's going on without needing to understand complex job APIs.

Checkout the [quickstart](quickstart) guide and the
[pipelines](category/prebuilt-pipelines) we've provided as well as the docs in
the sidebar.

## Concepts

The idea of the `stopes` framework is to make it easy to build reproducible
pipelines. This is done though "modules", a module is just a class with a `run`
function that executes something. A module can then be scheduled with the `stopes`
"launcher", this will decide where the code gets executed (locally or on a
cluster) and then wait for the results to be ready.

A **module** in `stopes` encapsulate a single step of a pipeline and its
requirements. This step is supposed to be able to execute on its own given its
input and generate an output. It will most often be executed as an isolated
job, so shouldn't depend on anything else than its config (e.g. global
variables, etc.). This ensures that each module can be run separately and in
parallel if possible.
A module also defines a clear API of the step via its configuration.

A **pipeline** in `stopes` it not much more than a python function that connects a
few modules together, but it could contain other python logic in the middle.
While you can run a `stopes` module as a normal python callable, the power of
`stopes` comes from the `launcher` that will manage the execution of the modules,
find the correct machines with matching requirements (if executing on a cluster)
and deal with memoization.

A **launcher** is the orchestrator of your pipeline, but is exposed to you
through a simple `async` API, so it looks like any
[asyncio](https://docs.python.org/3/library/asyncio.html) function and you do not have
to deal with where your code is being executed, if [memoization](stopes/cache)
is happening, etc. If you have never dealt with `async` in python, I do
recommend checking [this tutorial](https://realpython.com/async-io-python/), it
looks scarier than it is.

## Example

Here is an example of a basic pipeline that will take some file inputs, train a
[FAISS](https://faiss.ai/) index on it and then populate the index with the
files.

This example shows the usage of the launcher and how we reuse existing modules.

Here we assume
that the files have already been encoded with something that LASER to keep the
example simple, but you  want to have a first step doing
the encoding (see the [global mining pipeline](pipelines/global_mining) for a real example).

```python title="mypipeline.py"
import asyncio

import hydra
from omegaconf import DictConfig
from stopes.core.utils import clone_config
from stopes.modules.bitext.indexing.populate_faiss_index import PopulateFAISSIndexModule
from stopes.modules.bitext.indexing.train_faiss_index_module import TrainFAISSIndexModule

# the pipeline
async def pipeline(config):
    # setup a launcher to connect jobs together
    launcher = hydra.utils.instantiate(config.launcher)

    # train the faiss index
    trained_index = await launcher.schedule(TrainFAISSIndexModule(
        config=config.train_index
    ))

    # pass in the trained index to the next step through config
    with clone_config(config.populate_index) as config_with_index:
        config_with_index.index=trained_index

    # fill the index with content
    populated_index = await launcher.schedule(PopulateFAISSIndexModule(
        config=config_with_index
    ))
    print(f"Indexes are populated in: {populated_index}")

# setup main with Hydra
@hydra.main(config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    asyncio.run(pipeline(config))
```

Let's start with the `main`, this is a very basic boilerplate that:

1. sets up [hydra](https://www.hydra.cc) to get configuration when running the
   script. We recommend checking the [hydra tutorial](https://hydra.cc/docs/tutorials/intro/) on their site to understand
   how to build configurations and organize them. See below also for an example
   config.
2. starts `asyncio` and runs our async `pipeline` function.

The `pipeline` function is `async` as it will run some asynchronous code inside
it, so we need to tell python that this will be the case. The first thing it
does, is to initialize the `launcher` from the config, this is a trick to be
able to swap launchers on the CLI using config overrides. After that, we setup
the `TrainFAISSIndexModule` and `schedule` it with the launcher. This will check
if this step was already executed in the past, and if not, will schedule the
module on the cluster (or just locally if you want).

The `await` keyword tells python to "wait" for the job to finish and once that
is done, move to the next step. As we need to pass the generated `index` to the
populate step, we take the config read from hydra, and fill up the `index` with
the output of the training. We schedule and await that step, and finally just
log the location of the output file.

Let's look at the config:

```yaml title="conf/config"

embedding_files: ???
embedding_dimensions: 1024
index_type: ???

launcher:
    _target_: stopes.core.Launcher
    log_folder: executor_logs
    cluster: local
    partition:

train_index:
  lang: demo
  embedding_file: ${embedding_files}
  index_type: ${index_type}
  data:
    bname: demo
    iteration: 0
  output_dir: ts.index.iteration_0
  num_cpu: 40
  use_gpu: True
  embedding_dimensions: ${embedding_dimensions}
  fp16: True


populate_index:
  lang: demo
  index: ???
  index_type: ${index_type}
  embedding_files: ${embedding_files}
  output_dir: index.0
  num_cpu: 40
  embedding_dimensions: ${embedding_dimensions}
```

Hydra will take a yaml file and structure it for our usage in python. You can
see that we define at the top level:
```
embedding_files: ???
index_type: ???
```
This tells hydra that these two entries are empty and required, so it will
enforce that we specify them on the CLI. We pass them down to the sub-configs
for train/populate by using the `${}` placeholders.

The `launcher` entry is setup to initialize the
[submitit](https://github.com/facebookincubator/submitit) that currently
provides the main job management system. If you wanted to use a different
job/cluster system, you could implement your own launcher.

We can now call our script with:
```bash
python mypipeline.py embedding_files='[pathtomyfile.bin]' index_type="OPQ64,IVF1024,PQ64"
```

We could also override some of the defaults:

```bash
python mypipeline.py embedding_files='[pathtomyfile.bin]' index_type="OPQ64,IVF1024,PQ64" train_index.use_gpu=false
```

:::note

We use [hydra](https://www.hydra.cc) as the configuration system, but note that most modules
take a dataclass as config, so you could build that manually from a different
system (like argparse) if you did not want to use hydra.

:::
