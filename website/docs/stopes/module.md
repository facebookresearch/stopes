---
sidebar_position: 2
---

# Module Overview

A module is a python class that extends `StopesModule`:


```python
from stopes.code.stopes_module import StopesModule

class MyModule(StopesModule):
	def __init__(self, config: MyModuleConfig):
		...

	def run(
       self,
       iteration_value: tp.Optional[tp.Any] = None,
       iteration_index: int = 0,
   	):
		...
```


You should implement at least the `run `method, this is what will get executed
when your module is launched. By default, you don't need to worry about the
iteration parameters, see below for details of what these do.

If you want to initialize things before the module is run, you can use
`__init__.`

You can also implement the following methods to give more information about your
module:



* `requirements` - if you have specific requirements (gpus, memory, …) for your
  module, return a Requirements specification from this method. This will be
  called after `__init__` but before `run`.
* `name/comment` - some launchers (see below) might use this to identify/log
  your module runs. Feel free to implement them if you want, but you don't have
  to and they might not always be used.


## Arrays

We've observed that in many cases, pipeline steps are repeated on a number of
shards of data. This is common with large datasets and allows to chunk the data
processing on different machines for faster processing.

In this execution case, the goal is to execute the same code with the same
requirements on a number of shards, in order to avoid implementing this logic
for every module that needs to work on shards in the pipeline driving the
module. The StopesModule system can take care of this for you.

If your module implements the `array` method and returns an array of N values to
process, the module will be executed N times separately and the `run` method
will be called multiple times, independently. Every time the `run `method is
called for a module with an array, it will be passed two extra parameters:



* `iteration_value` that will contain a single value from the array
* `iteration_index` that corresponds the the index of that value in the array

The array method will be called after the module is initialized and in the same
process as the initialization. You can therefore compute the array based on the
config of the module or anything you compute in the `__init__` method.


## Gotchas



* In most cases, the `run` method will be executed in a distributed fashion.
  That means that:
    * `run` and `__init__ `might not be called with the same machine/process.
      E.g. when launching modules, `__init__` will be called where your pipeline
      driving script is executed, but `run` will be called in a separate
      process/job.
    * When using `array`, each separate call to `run` will potentially be called
      on a separate machine/process and on a separate copy of your module. That
      means that you can share value from `__init__` down to `run`, but you
      cannot share anything in your object between calls of `run, `you should
      not modify self inside of` run`.
    * When using `array`, there is no guarantee that `run` will be called in the
      same order as the values in your array. Only rely on the index passed to
      you and not on an execution order.
    * Your `run` method will probably apply side effects (e.g. write files). If
      this is the case, make sure to return the file path/handle from the run
      method so we can keep track of these.
