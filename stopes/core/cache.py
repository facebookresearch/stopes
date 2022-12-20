# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import pickle
import typing as tp
from abc import ABC, abstractmethod
from pathlib import Path

from omegaconf.omegaconf import OmegaConf

from stopes.core.utils import sha_key

if tp.TYPE_CHECKING:
    from stopes.core import StopesModule

logger = logging.getLogger("stopes.cache")

################################################################################
#  Caching definition
################################################################################


class MissingCache(Exception):
    """Raised when we do not find the cache"""

    pass


class Cache(ABC):
    def get_cache_key(
        self,
        module: "StopesModule",
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> str:
        return sha_key(
            self._raw_key(
                module,
                iteration_value,
                iteration_index,
            )
        )

    def _raw_key(
        self,
        module: "StopesModule",
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> str:
        local_key = ".".join(
            [self.__class__.__module__, self.__class__.__qualname__, self.version()]
        )
        cache_key = module.cache_key() + (iteration_value, iteration_index, local_key)
        k = repr(cache_key)
        return k

    @abstractmethod
    def get_cache(
        self,
        module: "StopesModule",
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
        validate: bool = True,
    ) -> tp.Optional[tp.Any]:
        """
        gets the cached value for this module. If no cache is found
        raise MissingCache.
        """
        ...

    @abstractmethod
    def save_cache(
        self,
        module: "StopesModule",
        value: tp.Any,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        ...

    @abstractmethod
    def invalidate_cache(
        self,
        module: "StopesModule",
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        ...

    def invalidate_module_cache(
        self,
        module: "StopesModule",
    ) -> None:
        array = module.array()
        if array is None:
            array = [None]
        for idx, val in enumerate(array):
            self.invalidate_cache(module, val, idx)

    def version(self):
        return "0.0"


class NoCache(Cache):
    """
    cache that doesn't cache. Useful to keep the same logic even
    if we don't use caching
    """

    def get_cache(
        self,
        module: "StopesModule",
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
        validate: bool = True,
    ) -> tp.Optional[tp.Any]:
        raise MissingCache()

    def save_cache(
        self,
        module: "StopesModule",
        value: tp.Any,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        pass

    def invalidate_cache(
        self,
        module: "StopesModule",
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        pass


# TODO add LRU
class FileCache(Cache):
    def __init__(self, caching_dir: Path):
        self.caching_dir = Path(caching_dir)
        self.caching_dir.mkdir(exist_ok=True)

    def get_cache_file_path(
        self,
        module: "StopesModule",
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> Path:
        cache_key = self.get_cache_key(
            module, iteration_value=iteration_value, iteration_index=iteration_index
        )
        return self.caching_dir / f"{cache_key}.pickle"

    def get_config_file_path(
        self,
        module: "StopesModule",
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> Path:
        cache_key = self.get_cache_key(
            module, iteration_value=iteration_value, iteration_index=iteration_index
        )
        return (
            self.caching_dir / f"{module.name()}.{cache_key}.{iteration_index:03d}.yaml"
        )

    def get_cache(
        self,
        module: "StopesModule",
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
        validate: bool = True,
    ) -> tp.Optional[tp.Any]:
        cache_file_path = self.get_cache_file_path(
            module, iteration_value=iteration_value, iteration_index=iteration_index
        )

        if not cache_file_path.is_file():
            raise MissingCache()

        try:
            with cache_file_path.open("rb") as cache_file:
                cached = pickle.load(cache_file)
        except Exception as e:
            logger.warning(
                f"couldn't load cache from {cache_file_path} for: "
                f"{module.name()} iteration {iteration_index}",
                exc_info=e,
            )
            cache_file_path.unlink()
            raise MissingCache()

        if validate:
            try:
                valid_result = module.validate(cached, iteration_value, iteration_index)
            except Exception as e:
                logger.warning(f"cache is invalid for {module.name()}", exc_info=e)
                valid_result = False
            if not valid_result:
                self.invalidate_cache(module, iteration_value, iteration_index)
                raise MissingCache()

        return cached

    def save_cache(
        self,
        module: "StopesModule",
        result: tp.Any,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        try:
            cache_file_path = self.get_cache_file_path(
                module, iteration_value=iteration_value, iteration_index=iteration_index
            )
            with cache_file_path.open("wb") as cache_file:
                pickle.dump(result, cache_file)
            cache_cfg_path = self.get_config_file_path(
                module, iteration_value=iteration_value, iteration_index=iteration_index
            )
            OmegaConf.save(config=module.config, f=cache_cfg_path)
            logger.info(f"written cache for {module.name()}:{iteration_index}")
        except Exception as e:
            logger.warning(
                f"couldn't write cache for {cache_file_path} for: {module.name()}:{iteration_index}",
                exc_info=e,
            )

    def invalidate_cache(
        self,
        module: "StopesModule",
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> None:
        try:
            logger.info(f"removing cache for {module.name()}:{iteration_index}")
            cache_file_path = self.get_cache_file_path(
                module, iteration_value=iteration_value, iteration_index=iteration_index
            )
            cache_file_path.unlink()
        except Exception as e:
            logger.warning(
                f"couldn't invalidate cache for {cache_file_path} for: {module.name()}:{iteration_index}",
                exc_info=e,
            )
