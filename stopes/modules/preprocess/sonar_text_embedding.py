# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import gc
import typing as tp
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from fairseq2.assets.error import AssetError
from retrying import retry
from sonar.inference_pipelines.text import (
    EmbeddingToTextModelPipeline,
    TextToEmbeddingModelPipeline,
)

from stopes.core.stopes_module import Requirements
from stopes.modules.partitioned_data_mapper import (
    BatchMapper,
    PartitionedDataMapper,
    PartitionedDataMapperConfig,
)
from stopes.utils.arrow_utils import (
    apply_on_nested_array,
    apply_over_groups,
    numpy_to_fixed_size_pyarrow_array,
    pyarrow_fixed_size_array_to_numpy,
)
from stopes.utils.sharding.abstract_shards import BatchFormat, batch_to_table
from stopes.utils.sharding.parquet_shards import ParquetOutputConfig

fairse2_asset_loading_retry = retry(
    retry_on_exception=lambda exception: isinstance(exception, (AssetError, IOError)),
    stop_max_attempt_number=20,
    wait_random_min=1000,
    wait_random_max=30_000,
)


@dataclass
class LangColumnConfig:
    column: str
    lang_value: tp.Optional[str] = None
    lang_column: tp.Optional[str] = None
    suffix: str = "_sonar_emb"

    """
    1. For multi-lingual setup, provide `lang_column` a column of input dataset containing lang values of `column`
    2. For mono-lingual setup, provide directly the language with `lang_value` (like "eng_Latn")
    """

    def __post_init__(self):
        if (self.lang_value is None) == (self.lang_column is None):
            raise ValueError(
                "Exactly one param out of `lang` and `lang_col` should be provided"
            )

        assert len(self.suffix) > 0


@dataclass
class SonarTextEmbedderConfig:
    column_config: tp.List[LangColumnConfig]
    model_name: str = "text_sonar_basic_encoder"
    tokenizer_name: tp.Optional[str] = None
    device: str = "cuda"
    batch_size: int = 10
    dtype: tp.Optional[str] = None  # "float32"
    """
    This config allow to handle multiple columns and each of columns can be multilingual.
    It also supports columns with nested text values (a list of sentences per row)
    in which case it returns nested embeddings column.
    """

    def __post_init__(self):
        self.tokenizer_name = self.tokenizer_name or self.model_name


class _MonoLangMapperInterface(BatchMapper):
    @torch.inference_mode()
    def _apply_on_simple_column(
        self,
        col: tp.Union[pa.Array, pa.ChunkedArray],
        lang_value: str,
    ) -> tp.Union[pa.Array, pa.ChunkedArray]:
        ...

    def _apply_on_unique_lang_table(
        self, table: pa.Table, config: LangColumnConfig
    ) -> pa.Table:
        if config.lang_column:
            assert (
                len(table[config.lang_column].unique()) == 1
            ), "this method should be called only for unique lang values"
            lang_value = table[config.lang_column][0].as_py()
        else:
            lang_value = config.lang_value

        try:
            col = table[config.column]
        except KeyError:
            # `table.flatten()` allows to access fields from stuct directly
            # with the following name: `{column_name}.{struct_field_name}`
            col = table.flatten()[config.column]

        new_column = apply_on_nested_array(
            partial(self._apply_on_simple_column, lang_value=lang_value),
            col,
        )
        new_name = f"{config.column}{config.suffix}"
        return table.append_column(new_name, new_column)

    def __call__(
        self, table: tp.Optional[tp.Union[pa.Table, pd.DataFrame]]
    ) -> tp.Optional[pa.Table]:
        if table is None:
            return None

        table = batch_to_table(table)

        for current_config in self.config.column_config:
            table = apply_over_groups(
                table,
                [
                    current_config.lang_column
                ],  # note that if `current_config.lang_column` is None function will be applied on the full table
                partial(self._apply_on_unique_lang_table, config=current_config),
            )

        return table


class SonarTextBatchEmbedder(_MonoLangMapperInterface):
    def __init__(self, config: SonarTextEmbedderConfig) -> None:
        super().__init__(config)
        self.dtype = np.dtype(self.config.dtype) if self.config.dtype else None
        self.pipeline = fairse2_asset_loading_retry(TextToEmbeddingModelPipeline)(
            self.config.model_name,
            self.config.tokenizer_name,
            device=torch.device(self.config.device),
        )

    @torch.inference_mode()
    def _apply_on_simple_column(
        self,
        col: tp.Union[pa.Array, pa.ChunkedArray],
        lang_value: str,
    ) -> pa.FixedSizeListArray:

        assert pa.types.is_string(col.type) or pa.types.is_binary(
            col.type
        ), f"unsupported dtype: {col.type}"
        inp = col.to_pandas()
        order = np.argsort([len(x) for x in inp])
        ordered_inp = inp.iloc[order].to_list()
        emb = (
            self.pipeline.predict(
                input=ordered_inp,
                source_lang=lang_value,
                batch_size=self.config.batch_size,
            )
            .cpu()
            .numpy()
        )
        torch.cuda.empty_cache()
        gc.collect()
        if self.dtype:
            emb = emb.astype(self.dtype)
        inv_order = np.argsort(order)
        emb = emb[inv_order]
        return numpy_to_fixed_size_pyarrow_array(emb)


@dataclass
class SonarEmbeddingDecodingConfig(SonarTextEmbedderConfig):
    model_name: str = "text_sonar_basic_decoder"
    generator_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None
    """
    for generator kwargs, please refer to `BeamSearchSeq2SeqGenerator` in fairseq2 :
        https://github.com/facebookresearch/fairseq2/blob/main/src/fairseq2/generation/beam_search.py

    """

    # dont forget to rename column suffix
    # suffix: str = "_sonar_emb"
    def __post_init__(self):
        super().__post_init__()
        for col_config in self.column_config:
            if col_config.suffix == "_sonar_emb":
                print(
                    "Using default ENCODER suffix '_sonar_emb',"
                    f" condsider replace it for column = {col_config.column}"
                )


class SonarEmbeddingToTextMapper(_MonoLangMapperInterface):
    def __init__(self, config: SonarEmbeddingDecodingConfig) -> None:
        super().__init__(config)
        self.pipeline = fairse2_asset_loading_retry(EmbeddingToTextModelPipeline)(
            self.config.model_name,
            self.config.tokenizer_name,
            device=torch.device(self.config.device),
        )

    @torch.inference_mode()
    def _apply_on_simple_column(
        self,
        col: tp.Union[pa.Array, pa.ChunkedArray],
        lang_value: str,
    ) -> tp.Union[pa.Array, pa.ChunkedArray]:
        np_array = pyarrow_fixed_size_array_to_numpy(col)

        text = self.pipeline.predict(
            inputs=torch.from_numpy(np_array),
            target_lang=lang_value,
            batch_size=self.config.batch_size,
            **(self.config.generator_kwargs or {}),
        )
        return pa.array(text, type=pa.string())


@dataclass
class SonarTextEmbedderStopesConfig(PartitionedDataMapperConfig):
    sonar_config: SonarTextEmbedderConfig

    def __post_init__(self):
        super().__post_init__()
        self.input_dataset_config.batch_format = BatchFormat.ARROW
        assert isinstance(
            self.output_dataset_config, ParquetOutputConfig
        ), "Embedding can be serialized only in Parquet"


class SonarTextEmbedderStopes(PartitionedDataMapper):
    def __init__(self, config: SonarTextEmbedderStopesConfig) -> None:
        super().__init__(config)

    def get_custom_metadata(self, *args, **kwargs) -> tp.Dict[str, tp.Any]:
        return {}

    def requirements(self):
        return Requirements(
            nodes=1,
            mem_gb=30,
            tasks_per_node=1,
            gpus_per_node=int(bool(self.config.sonar_config.device.startswith("cuda"))),
            cpus_per_task=(
                4 if self.config.sonar_config.device.startswith("cuda") else 20
            ),
        )

    def get_batch_mapper(self):
        return SonarTextBatchEmbedder(self.config.sonar_config)
