# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import asyncio
import sys

import hydra
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf

from stopes.core.stopes_module import StopesModule


@hydra.main(config_path="conf", config_name="launch_conf")
def main(config: DictConfig) -> None:
    """
    Launches a module from CLI.

    The module need to have a sample .yaml file in conf/ folder.
    The .yaml file will need a _target_ key containing the name of the module
    to instantiate.

    Examples:
        * python -m stopes.pipelines.bitext.launch_module +spm=standard_conf spm.config.output_dir=/tmp spm.config.train_data_file=wiki.txt
        will use conf/spm/standard_conf.yaml to start a TrainSpmModule
        * adding 'launcher.cluster=debug' will run the module in the same process
    """
    config_keys = [k for k in config.keys() if k != "launcher"]
    if len(config_keys) == 0:
        print(main.__doc__, file=sys.stderr)
        sys.exit(1)

    config_keys = [k for k in config.keys() if k != "launcher"]
    assert len(config_keys) == 1, "should only specify one module config"
    launcher = hydra.utils.instantiate(config.launcher)
    module_conf = config[config_keys[0]]
    module = StopesModule.build(module_conf)

    loop = asyncio.get_event_loop()
    if config.launcher.cluster == "debug":
        loop.set_debug(True)
    loop.run_until_complete(launcher.schedule(module))


if __name__ == "__main__":
    main()
