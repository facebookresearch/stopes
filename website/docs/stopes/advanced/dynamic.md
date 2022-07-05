---
sidebar_position: 2
---

# Dynamic Initializing Modules (advanced)

It is easy to initialize a module like a normal python class with`
MyModule(config)`. However this would make your pipeline static as the module
couldn't be swapped.


**Problem:** For instance, imagine that your pipeline has an _embed_ step that
takes raw text as input and outputs an embedding of that text.

You might want to test different embedding methods, let's say, compare the LASER
implementation with a HuggingFace encoder.

If you wrote your code with `encoder = LaserEncoderModule(config)` you will not
be able to swap this step to use the `HFEncoderModule` without changing the code
of your pipeline.

**Solution**: Because we are using Hydra, we have an easy way to specify modules
in config and override them when calling the pipeline. All you have to do is to
use:


```python
embed_module = StopesModule.build(self.config.embed_text, lang=lang)
```


The `build` helper will find the `_target_` entry in the embed_text config and
initialize the module that it points to. The `kwargs` of build can be used to
specify in code a specific value of the config.

Thanks to `build`, we can now have two config files in the embed_text group that
will point to the different modules:




```yaml title="laser_module.yaml"
# @package module

_target_: modules.LaserEncoderModule
config:
    lang: ???
```



```yaml title="hf_module.yaml"
# @package module

_target_: modules.HFEncoderModule
config:
    lang: ???
```


And you can override this module from the cli:

```bash
python yourpipeline.py embed_text=hf_module src_lang=bn tgt_lang=hi
+data=ccg
```

This does look a bit odd at first, but look at the implementation of
global_mining to see how it flows and how modules are used and config/data is
passed around.
