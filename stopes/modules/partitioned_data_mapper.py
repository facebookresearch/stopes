# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import abc
import copy
import gc
import getpass
import inspect
import logging
import math
import typing as tp
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import submitit

from stopes.core.launcher import SkipValue
from stopes.core.stopes_module import Requirements, StopesModule
from stopes.utils.sharding.abstract_shards import (
    BatchType,
    InputShardingConfig,
    OutputDatasetConfig,
    PartitionedDataMapperState,
    Shard,
    batch_length,
    batch_tail,
    batch_to_table,
    concat_batches,
)
from stopes.utils.sharding.text_shards import TextShard


def get_class_hierarchy(cls) -> tp.List[tp.Any]:
    classes = inspect.getmro(cls)
    return [cls for cls in classes if cls not in [object, abc.ABC, BatchMapper]]


def source_code(cls) -> str:
    try:
        return inspect.getsource(cls)
    except Exception:
        return ""


def get_class_hierarchy_code(cls) -> tp.Dict[str, str]:
    classes = get_class_hierarchy(cls)
    return {repr(cls): source_code(cls) for cls in classes}


@dataclass
class PartitionedDataMapperConfig:
    input_dataset_config: InputShardingConfig
    output_dataset_config: OutputDatasetConfig

    def __post_init__(self):
        # to propagate parquet partitions
        if getattr(self.output_dataset_config, "keep_same_partitioning", False):
            assert hasattr(self.input_dataset_config, "partition_columns")
            assert hasattr(self.output_dataset_config, "partition_columns")
            self.output_dataset_config.partition_columns = getattr(
                self.input_dataset_config, "partition_columns"
            )


class PartitionedDataMapper(StopesModule):
    """
    The main goal of the `PartitionedDataMapper` (and other classes around it)
    is to create an efficient abstraction layer for Batch Processing in Stopes.
    In essence, we want to disentangle the batch based transformation logic from data IO.
    Thus,
    - `PartitionedDataMapper` takes care of all partitioning logic, IO operations, ... and when it's used for sub-classing,
    - the developer needs only to implement the logic that define a function that transforms a batch to a batch
    - `def get_batch_mapper(self) -> tp.Callable[[tp.Optional[BatchType]], tp.Optional[BatchType]]:`
    - Here, we need to return a callable that could be applied several times in mini-batching loop without reinitazing it
    - One could subclass `class BatchMapper(ABC):` to structure such callable objects typically for model inference case.

    This pattern looks similar to [`Dask.DataFrame.map_partitions`](https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.map_partitions.html),
      [`pyspark.RDD.mapPartitions`](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.RDD.mapPartitions.html) or
      [`ray.data.Dataset.map_batches`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html).

    * In particular, this disentangle would allow to reuse Stopes batch transformer code interchangeably for any such higher level framework.
    * Chaining several transformation over on batch inside a single Stopes Model should be also easier with this approach.

    """

    state: tp.Optional[PartitionedDataMapperState]
    iteration_index: int
    written_batches_index: int

    def __init__(
        self,
        config: PartitionedDataMapperConfig,
        state: tp.Optional[PartitionedDataMapperState] = None,
        shards_to_skip: tp.Optional[tp.Dict[int, tp.List[Path]]] = None,
    ) -> None:
        super().__init__(config, PartitionedDataMapperConfig)

        self.input_dataset_config = config.input_dataset_config
        self.output_dataset_config = config.output_dataset_config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.state = state
        self.shards_to_skip = shards_to_skip

        if (
            self.output_dataset_config.expected_schema is None
            and self.output_dataset_config.validate_schema
        ):
            self.output_dataset_config.expected_schema = self.guessed_expected_schema()

        if (
            self.output_dataset_config.expected_schema is None
            or not self.output_dataset_config.validate_schema
        ):
            self.logger.warning("Output schema will NOT be validated")

    def array(self) -> tp.List[Shard]:
        if self.state is not None:
            return [self.state.iteration_value]

        full_array = self.input_dataset_config.make_shards()
        if self.shards_to_skip is not None:
            self.logger.warn(f"Adding shards to skip: {len(self.shards_to_skip)}")
            for idx, shards_to_skip in self.shards_to_skip.items():
                full_array[idx] = SkipValue(shards_to_skip)  # type: ignore
        return full_array

    @abstractmethod
    def get_batch_mapper(
        self,
    ) -> tp.Callable[[tp.Optional[BatchType]], tp.Optional[BatchType]]:
        # for stateful mapping one can follow the this pattern
        # `return CallableBatchMapperClass(self.my_batch_mapper_config)`
        # with `CallableBatchMapperClass.__call__(tp.Optional[BatchType]]) -> BatchType`
        return lambda batch: batch

    def get_metadata(self, *args, **kwargs) -> tp.Dict[str, tp.Any]:
        return {
            **self.get_common_metadata(*args, **kwargs),
            **self.get_custom_metadata(*args, **kwargs),
        }

    def get_common_metadata(self, *args, **kwargs) -> tp.Dict[str, tp.Any]:
        meta = {}
        batch_mapper = kwargs.get("batch_mapper", None)

        meta["config"] = self.config
        if batch_mapper:
            meta["batch_mapper_class"] = get_class_hierarchy_code(
                batch_mapper.__class__
            )
            try:
                meta["batch_mapper_code"] = inspect.getsource(batch_mapper)
            except Exception:
                meta["batch_mapper_code"] = ""

        meta["username"] = getpass.getuser()
        meta["save_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # TODO: add pip/git/conda info
        return meta

    @abstractmethod
    def get_custom_metadata(self, *args, **kwargs) -> tp.Dict[str, tp.Any]:
        return {}

    def guessed_expected_schema(
        self, num_rows=10, nb_tries=3
    ) -> tp.Optional[pa.Schema]:

        arrays = self.array()
        if len(arrays) == 0:
            self.logger.warning("Empty ARRAYs detected")
            return None
        batch_mapper_fn = self.get_batch_mapper()
        for i in range(nb_tries):
            batch = next(
                arrays[i].to_batches(
                    batch_size=min(
                        num_rows, self.input_dataset_config.batch_size or num_rows
                    ),
                    columns=self.input_dataset_config.columns,
                    batch_format=self.input_dataset_config.batch_format,
                )
            )
            if batch is not None:
                output_batch = batch_mapper_fn(batch)
                return batch_to_table(output_batch).schema

        self.logger.warning("Resulted batch is empty")
        return None

    def run(
        self, iteration_value: tp.Optional[tp.Any] = None, iteration_index: int = 0
    ) -> tp.List[Path]:

        if self.state is None:
            # try to get state from checkpoint
            self.state = self.output_dataset_config.reload_state(iteration_value)  # type: ignore

        if self.state is None:
            self.state = PartitionedDataMapperState(
                iteration_index=iteration_index,
                iteration_value=iteration_value,  # type: ignore
                written_batches_index=-1,
                intermediate_index=0,
                input_rows_written=0,
            )
        iteration_value = self.state.iteration_value
        iteration_index = self.state.iteration_index

        if iteration_value is None:
            self.logger.warning(
                f"Input `None` iteration_value, skipping iteration {iteration_index}"
            )
            return []

        if not isinstance(iteration_value, Shard):
            raise ValueError("Partitioned dataset should be defined")

        shard: Shard = iteration_value
        # we need to initialize the batch_mapper_fn each to to avoid sharing it state between workers
        batch_mapper_fn = self.get_batch_mapper()
        metadata = self.get_metadata(**{"batch_mapper": batch_mapper_fn})
        output_batch_size = self.output_dataset_config.batch_size
        # transforming batch split by mini-batches (typical for GPU inference)
        mini_batch_results: tp.List[BatchType] = []

        nb_current_samples = 0
        output_written_paths = []

        # FIXME: text_shard has been entered implicitly in to_batches(), change
        # this to have a consistent code among all types of shards
        if isinstance(shard, TextShard):
            _context = nullcontext()
        else:
            _context = shard  # type: ignore

        current_input_rows_processed = 0

        if hasattr(shard, "skip_n_rows") and shard.skip_n_rows and output_batch_size:
            # we need to advance intermediate_index to make sure to not overwrite existing files
            self.state.intermediate_index = (
                math.ceil(shard.skip_n_rows / output_batch_size) + 1
            )

        with _context:
            for batch_idx, batch in enumerate(
                shard.to_batches(
                    batch_size=self.input_dataset_config.batch_size,
                    columns=self.input_dataset_config.columns,
                    batch_format=self.input_dataset_config.batch_format,
                )
            ):
                if batch_idx <= self.state.written_batches_index:
                    continue

                input_batch_len = batch_length(batch)

                left_to_skip = 0
                # skip from the state
                if (
                    current_input_rows_processed + input_batch_len
                    <= self.state.input_rows_written
                ):
                    current_input_rows_processed += input_batch_len
                    continue
                else:
                    left_to_skip = (
                        self.state.input_rows_written - current_input_rows_processed
                    )

                # skip from shard config
                if hasattr(shard, "skip_n_rows") and shard.skip_n_rows > 0:
                    if (
                        current_input_rows_processed + input_batch_len
                        <= shard.skip_n_rows
                    ):
                        current_input_rows_processed += input_batch_len
                        continue
                    else:
                        # shard config overrides state
                        left_to_skip = shard.skip_n_rows - current_input_rows_processed

                if left_to_skip > 0:
                    batch = batch_tail(batch, input_batch_len - left_to_skip)

                transformed_batch = batch_mapper_fn(batch)
                current_input_rows_processed += input_batch_len
                if transformed_batch is not None:
                    nb_current_samples += batch_length(transformed_batch)
                    mini_batch_results.append(transformed_batch)

                if output_batch_size and nb_current_samples >= output_batch_size:
                    batch_to_write = concat_batches(mini_batch_results)
                    new_state = copy.copy(self.state)
                    new_state.intermediate_index += 1
                    new_state.written_batches_index = batch_idx
                    new_state.input_rows_written = current_input_rows_processed

                    files_path = self.output_dataset_config.write_batch(
                        batch_to_write,
                        (self.state.intermediate_index, iteration_index),
                        metadata=metadata,
                        state_checkpoint=new_state,
                    )
                    self.logger.info(
                        f"Following files has been written: \n {files_path}"
                    )
                    output_written_paths.extend(files_path)
                    nb_current_samples = 0
                    # only update state on successful write
                    self.state = new_state
                    # TODO deal with writting state to checkpoint here
                    mini_batch_results = []
                    gc.collect()

            if len(mini_batch_results) > 0:
                batch_to_write = concat_batches(mini_batch_results)
                new_state = copy.copy(self.state)
                new_state.intermediate_index += 1
                new_state.written_batches_index = batch_idx
                new_state.input_rows_written = current_input_rows_processed
                files_path = self.output_dataset_config.write_batch(
                    batch_to_write,
                    (self.state.intermediate_index, iteration_index),
                    metadata=metadata,
                    state_checkpoint=new_state,
                )
                self.logger.info(f"Following files has been written: \n {files_path}")
                output_written_paths.extend(files_path)
            self.state = None

        return output_written_paths

    def name(self):
        name = (
            self.get_custom_metadata().get("name")
            or str(self.input_dataset_config.input_file)[-250:]
        )
        return name

    def checkpoint(
        self, *args, **kwargs
    ) -> tp.Optional[submitit.helpers.DelayedSubmission]:
        if self.state is not None:
            return submitit.helpers.DelayedSubmission(self)
        return None


class BatchMapper(ABC):
    """
    Abstract class that could be used to structure a Statefull transformation.
    It takes typically a config for the init, loads some models in init.
    Then there a `__call__` method to implement that transform Batch to Batch (potentially small)

    Example:

    ... import librosa
    >>> import whisper
    >>> import torch
    >>> import pandas as pd


    >>> class LIDPredictor(BatchMapper):
    ...     def __init__(self, model_config) -> None:
    ...         self.model_config = model_config
    ...         self.model = whisper.load_model(model_config.get("name", "large-v2"))

    ...     def row_mapper(self, wav: np.ndarray) -> tp.Dict[str, float]:
    ...         audio = whisper.pad_or_trim(wav, length=480000)
    ...         mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
    ...         _, probs = self.model.detect_language(mel)
    ...         return probs

    ...     def __call__(self, batch: BatchType) -> pd.DataFrame:
                if not isinstance(batch, pd.DataFrame):
    ...             batch = batch.to_pandas()
    ...         inp_col = self.model_config.get("input_column", "wavform")
    ...         out_col = self.model_config.get("output_column", "lid_proba")

    ...         with torch.inference_mode():
    ...             batch[f"out_col{output_suffix}"] = batch[inp_col].apply(self.row_mapper)

    ...         return batch

    """

    def __init__(self, config, *args, **kwargs):
        self.config = config
        # self.model = load_model(config.path)

    @abstractmethod
    def __call__(self, batch: tp.Optional[BatchType]) -> tp.Optional[BatchType]:
        if batch is None:
            return None
        # more complex case we want to do
        # return self.model(pandas_to_torch(batch))
        return batch

    def clear_memory(self) -> None:
        pass
        # import gc
        # if self.model is not None:
        #     try:
        #         self.model.cpu()
        #         self.model = None
        #     except:
        #         pass
        # gc.collect()
        # torch.cuda.empty_cache()


@dataclass
class IOConfigWithBatchMapper(PartitionedDataMapperConfig):
    mapper_config: tp.Any = None


def stopes_data_mapper(
    requirements: Requirements,
    metadata: tp.Optional[tp.Dict[str, tp.Any]] = None,
    shards_to_skip: tp.Optional[tp.Dict[int, tp.List[Path]]] = None,
    state: tp.Optional[PartitionedDataMapperState] = None,
):
    """
    A decorator function that can be used to wrap a BatchMapper class into PartitionedDataMapper

    Args:
        - requirements (Requirements):
        - metadata (tp.Optional[tp.Dict[str, tp.Any]], optional):. Metadata to attach to resulting dataset. Defaults to None.

    Example:

        @stopes_data_mapper(requirements=Requirements())
        class LIDPredictor(BatchMapper):
            ...
            def __init__(self, ...):
                ...

            def __call__(self, batch):
                ...
                return transformed_batch

        lid_stopes_module = LIDPredictor(input_config, output_config, mapper_config={"model_name": "large_v2"})

        launcher = Launcher(
                cache=None,
                config_dump_dir=...,
                log_folder=...,
                cluster="slurm",
            )

        results = await launcher.schedule(lid_stopes_module)

    """

    def decorator_mapper(mapper_cls: BatchMapper):
        class WrappedMapper(PartitionedDataMapper):
            def get_batch_mapper(
                self,
            ):
                return mapper_cls(
                    self.config.mapper_config,
                )

            def requirements(self):
                return requirements

            def get_custom_metadata(self, *args, **kwargs) -> tp.Dict[str, tp.Any]:
                if metadata is None:
                    return {}
                return metadata or {}

        def config_assembler(
            input_dataset_config: InputShardingConfig,
            output_dataset_config: OutputDatasetConfig,
            mapper_config: tp.Any,  # should be a BatchedMapper dataclass config
        ):
            io_config_with_mapper = IOConfigWithBatchMapper(
                input_dataset_config=input_dataset_config,
                output_dataset_config=output_dataset_config,
                mapper_config=mapper_config,
            )

            return WrappedMapper(
                io_config_with_mapper,
                state=state,
                shards_to_skip=shards_to_skip,
            )

        return config_assembler

    return decorator_mapper
