---
sidebar_position: 4
---

# Configuration

We use hydra for configuration. You should probably check out the hydra
tutorial:
[https://hydra.cc/docs/tutorials/intro](https://hydra.cc/docs/tutorials/intro)
but it's not a requirement.

Modules `__init__` HAVE to take either a structured configuration as parameter
or an `omegaconf.DictConfig`. A structured configuration is a [python
dataclass](https://docs.python.org/3/library/dataclasses.html), e.g.


```python
from dataclasses import dataclass
from omegaconf import MISSING

@dataclass
class MyModuleConfig:
    lang: str = MISSING
    spm_model: str = "/path/to/my/model.spm"
```


Structured configs make it easier to track what is expected as a config for a
module and makes it self documenting. But you can also just use a DictConfig if you prefer.


If you implement the init method of the module, make sure to call
`super().__init__(config) `so that the module system knows about your module
setup. You can then access `self.config `anywhere in your module after
initialization

Actual configs live in YAML files in the config/module/ folder and should look
like this:


```yaml
# @package module
_target_: stopes.modules.MyModule
config:
    lang: null
    spm_model: /path/to/my/model.spm
```


The `_target_` field should point to the full python module path of your module

`config` should contain the config of your module.

You should save this in a file with your model name. You could have multiple
versions of your config, save them with the same `_target_` but different file
names (e.g. `my_module_large_spm.yaml`, `my_module_small_spm.yaml`, etc.).

The yaml config file should contain the baseline configuration for your module
and things that you do not expect to change often. In hydra terms, you are
adding a possible option for a config group (the module group: see `@package
module`)

You can use hydra/[omegaconf
"resolvers"](https://omegaconf.readthedocs.io/en/2.1_branch/custom_resolvers.html#built-in-resolvers)
to depend on other bits of configs or environment variables:


```yaml
# @package module
_target_: stopes.modules.MyModule
config:
    lang: null
    laser_path: /laser/is/here
    laser_model: ${module.my_module.laser_path}/model1.mdl
    spm_model: ${oc.env:SPM_MODEL}
```


Note: try not to rely too much on environment variables as we want these files
to be the base for reproducibility and shareability of the module configurations
you experiment with. Relying on special environment variables will make this
hard.

You can use hydra config composition if you want your config to inherit or
configure a subpart of your config, see
https://hydra.cc/docs/patterns/extending_configs
