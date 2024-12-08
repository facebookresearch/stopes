# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import logging
import typing as tp
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import hydra
import submitit
from omegaconf import DictConfig, OmegaConf

import stopes.core
from stopes.core import utils

# Set up a default logging handler.
logger = logging.getLogger("stopes.module")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(process)d:%(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M",
)


@dataclasses.dataclass
class Requirements:
    nodes: int = 1
    mem_gb: tp.Optional[int] = None
    tasks_per_node: int = 1
    gpus_per_node: int = 0
    cpus_per_task: int = 5
    timeout_min: int = 720
    constraint: tp.Optional[str] = None
    max_num_timeout: int = 10


class StopesModule(ABC):
    # list of retries per index
    retry_counts: tp.List[int]

    @staticmethod
    def build(config: tp.Any, **kwargs) -> "StopesModule":
        """Builds the correct module given a loaded config with a _target_ entry."""
        assert hasattr(
            config, "_target_"
        ), "You need to specify the module to create in the yaml file with _target_"
        target = config._target_

        if hasattr(config, "config"):
            # nested config
            warnings.warn(
                f"Nested configs are deprecated. Received nested config for {target}",
                DeprecationWarning,
            )
            merged_conf = OmegaConf.merge(config, {"config": kwargs})
            return hydra.utils.instantiate(merged_conf, _recursive_=False)

        if kwargs:
            config = OmegaConf.merge(config, kwargs)
        # Hydra will detach `config` from its parent in `instantiate`.
        # We need to resolve before that.
        OmegaConf.resolve(config)
        return hydra.utils.instantiate({"_target_": target}, config, _recursive_=False)  # type: ignore[no-any-return]

    def __init__(
        self, config: tp.Any, config_class: tp.Optional[tp.Type[tp.Any]] = None
    ):
        if dataclasses.is_dataclass(config):
            config = OmegaConf.structured(config)
        if config_class is not None:
            self.config = utils.promote_config(config, config_class)
        else:
            assert isinstance(config, DictConfig), (
                "stopes module configs must be either a dataclass or a omega.DictConfig."
                f" Received a {type(config)}"
            )
            self.config = config
        OmegaConf.resolve(self.config)
        OmegaConf.set_readonly(self.config, True)
        self.retry_counts = [0]
        self.transient_configs = self._get_transient_configs(config_class)

    def __call__(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
        cache: tp.Optional["stopes.core.Cache"] = None,
    ) -> tp.Any:
        """
        called when the job starts running.
        please implement `run` instead as we might need to add generic stuff in here
        """
        res = self.run(iteration_value=iteration_value, iteration_index=iteration_index)
        if cache is not None:
            cache.save_cache(self, res, iteration_value, iteration_index)
        return res

    @classmethod
    def _get_transient_configs(cls, config_class: tp.Optional[tp.Type[tp.Any]] = None):
        """Parse the config definition and get the transient config attributes"""

        def inspect_transiency(field: dataclasses.Field, states: dict):
            # get transient attributes from the field definition
            if field.metadata.get("transient", False):
                states[field.name] = True

            # Requirements are always transient
            elif field.name == "requirements":
                states[field.name] = True
            else:
                states[field.name] = {}
                if dataclasses.is_dataclass(field.type):
                    for sub_field in dataclasses.fields(field.type):
                        inspect_transiency(sub_field, states[field.name])
                # If a normal Python class, check if `transient_attributes` exists
                elif hasattr(field.type, "transient_attributes"):
                    for transient_attr in getattr(field.type, "transient_attributes"):
                        states[field.name][transient_attr] = True
                if len(states[field.name]) == 0:
                    del states[field.name]

        transient_configs: tp.Dict[str, tp.Any] = {}
        if config_class is None:
            # Attempt to infer the config class from the stopes module class itself
            config_class = tp.get_type_hints(cls).get("config")
        if dataclasses.is_dataclass(config_class):
            for field in dataclasses.fields(config_class):
                inspect_transiency(field, transient_configs)

        return transient_configs

    @abstractmethod
    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> tp.Any:
        """
        the core of your module, implement your logic here.

        If `array` returned an array, this is an array job and this `run` be
        called for each iteration, the value of the specific iteration in the array
        will be passed down to you. If it's not an array job, this will be None.
        """
        ...

    def array(self) -> tp.Optional[tp.List[tp.Any]]:
        """
        if you want to submit your job as an array job, you can compute
        the array of values you want to process here. This will be processed
        before the job is submitted to the particular cluster chosen.

        By default, we return None to indicate this is not an array job.
        """
        return None

    @abstractmethod
    def requirements(self) -> Requirements:
        """
        return a set of Requirements for your module, like num of gpus etc.
        """
        ...

    def name(self):
        """
        implement this if you want to give a fancy name to your job
        """
        # TODO ideally use hydra override_dirname here
        return "_".join([self.__class__.__name__, self.sha_key()])

    def cache_key(self):
        return (
            self.__class__.__module__,
            self.__class__.__qualname__,
            self.version(),
            self.get_config_for_cache(),
        )

    def get_config_for_cache(self):
        """
        Return a dictionary corresponding to the config. Here transient attributes,
        attributes of stopes.core.stopes_modules.Requirements will be excluded
        """
        config_for_cache = OmegaConf.to_container(self.config, resolve=False)
        assert isinstance(
            config_for_cache, dict
        ), "StopesModule.config need to be a dict config."
        OVERWRITE_VALUE_FOR_CACHE = -1

        def deep_overwrite(dct: dict, transient_cfg: dict):
            for k, v in dct.items():
                if k in transient_cfg:
                    if transient_cfg[k] is True:
                        dct[k] = OVERWRITE_VALUE_FOR_CACHE
                        continue
                    elif isinstance(transient_cfg[k], dict):
                        transient_cfg = transient_cfg[k]
                if isinstance(v, dict):
                    # Some configs are defined dynamically within the root module's compose config
                    if "_target_" in v:
                        sub_cls = hydra.utils.get_class(v["_target_"])

                        # If sub_cls is a config / dataclass
                        if dataclasses.is_dataclass(sub_cls):
                            sub_transient_cfg = self._get_transient_configs(sub_cls)
                        # sub_cls is a normal Python class, merge from the class property
                        # `transient_attributes` if exists
                        elif hasattr(sub_cls, "transient_attributes"):
                            sub_transient_cfg = {
                                k: True
                                for k in getattr(sub_cls, "transient_attributes")
                            }
                        else:
                            sub_transient_cfg = {}

                        transient_cfg = {**transient_cfg, **sub_transient_cfg}
                    deep_overwrite(v, transient_cfg)

        deep_overwrite(config_for_cache, self.transient_configs)
        return config_for_cache

    # TODO: @functools.cached_property()
    # This is only safe if cache_key is not allowed to change, in particular if config is frozen.
    # Can we guarantee that ?
    def sha_key(self):
        return utils.sha_key(repr(self.cache_key()))

    def comment(self):
        """
        same as `name` but for the job comment
        """
        return None

    @classmethod
    def version(cls):
        """
        the version of the module. If you want to invalidate
        some cache, you can change that
        """
        return "0.0"

    def validate(
        self,
        output: tp.Any,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> bool:
        """
        Validate the output of this module (for a single step of the array if it's an array module).
        This is called to invalidate the cache when needed but also at the end of the module run for
        sanity check.
        You can either return False or Raise/throw an exception if the results are not valid.

        The default implementation checks if the content of pickle is a "Path",
        and invalidate cache if the corresponding file is gone.
        """
        if isinstance(output, Path) and not output.exists():
            logger.warning(
                f"{self.name()} iteration {iteration_index}"
                f" points to missing file {output}, will invalidate it."
            )
            return False
        return True

    def should_retry(
        self,
        ex: Exception,
        attempt: int,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> bool:
        """
        decide if an exception is worth retrying. This will also depend on the retry
        settings of the launcher.
        """
        # by default, retry on missing output errors
        # they usually mean that there was a problem in slurm
        if type(ex) == submitit.core.utils.UncompletedJobError:
            return "has not produced any output" in str(ex)
        return False
