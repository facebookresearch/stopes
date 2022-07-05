---
sidebar_position: 3
---

# Debugging

You can launch an individual module with:


```bash
python launch.py +module=my_module
```


Where `my_module` is the name of the config file you want to use. This is useful
for debugging, usually.

The launcher is configured in
`global_mining/conf/main_conf.yaml`

You should not have to change this config to run your module.

The `+module= `argument is the way to tell hydra to pick up your module config
file. The launcher will use the `_target_` directive in the module to initialize
the correct module and then pass the right config.

You can override any part of the configuration with [normal hydra
overrides](https://hydra.cc/docs/1.0/advanced/override_grammar/basic/#basic-override-syntax).
For example, if you wanted to specify the lang parameter for your module, you
can do:


```bash
python launch.py +module=my_config module.config.lang=luo
```


The `module/my_config.yaml` file will be loaded and then the lang will be
overridden. This will create a new config for you.

The launcher will then run your module and dump the full config (with overrides)
in the outputs folder.

To do more advanced debugging, remember that a module is just a normal python
object that you can run as any python callable.  You can therefore just call
your module from the REPL or a notebook with:


```python
module = MyModule(config)
module()
```
