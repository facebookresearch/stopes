# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import logging
import math
import shutil
import typing as tp
from pathlib import Path

import numpy as np
import submitit
from omegaconf.omegaconf import MISSING

import stopes.core
from stopes.core.utils import measure
from stopes.modules.bitext.indexing.train_index import index_to_gpu
from stopes.utils.data_utils import DataConfig
from stopes.utils.embedding_utils import Embedding

if tp.TYPE_CHECKING:
    import faiss


@dataclasses.dataclass
class PopulateFAISSIndexConfig:
    lang: str = MISSING
    output_dir: str = MISSING
    index: str = MISSING
    index_type: str = MISSING
    embedding_files: tp.List[str] = MISSING
    chunk_size: int = 2**14

    use_gpu: bool = False
    num_cpu: int = 40
    embedding_dimensions: int = 1024
    fp16: bool = False

    enable_checkpointing: bool = False
    data: DataConfig = MISSING


@dataclasses.dataclass
class CheckpointSummary:
    """CheckpointSummary is an object that stores a checkpoint of the progress so far in the population process

    Note: The "original index" refers to the original index provided by the config
    Attributes:
        partial_idx (faiss.Index) - Stores the original index with the partial embedding shard that's been populated onto the original index so far. Stored as faiss.index.
        partial_idx_file (Path) - Stores the original index with the partial embedding shard that's been populated onto the original index so far. Stored as faiss.index written to a file.
        idx_size_before_populating_embedding (int) - This is the size of the original index. That is, it's the size of the original index, before we have even started populating the embedding onto it. Specifically, it's calculated as faiss.read_index(str(self.config.index)).ntotal.
        is_partial_file_valid (bool) - boolean to indicate if partial_idx_file is valid or not. The partial_idx_file is considered valid if the index was written to it without interruption - ie the job did not interrupt while doing faiss.write
        is_partial_idx_valid (bool) - boolean to indicate if partial_idx is valid or not. The partial_idx is considered valid if the chunk is added to the idx without interruption - ie the job did not interrupt while doing partial_idx.add(chunk)

    What's the diff between partial_idx and partial_idx_file?
        They both represent the same thing - the partial index (ie. the original index with the partial embedding shard that's been populated onto the original index so far)
        At the end of each iteration in the for loop (within add_embedding_to_index function), these two values should be exactly the same.
        The only difference is the format of storing the index: as a faiss.Index vs as written to a file.
    """

    partial_idx: "faiss.Index"
    partial_idx_file: tp.Optional[Path]
    idx_size_before_populating_embedding: tp.Optional[int]
    is_partial_file_valid: bool
    is_partial_idx_valid: bool


class PopulateFAISSIndexModule(stopes.core.StopesModule):
    def __init__(
        self,
        config: PopulateFAISSIndexConfig = PopulateFAISSIndexConfig(),
        checkpoint_summary: tp.Optional[CheckpointSummary] = None,
    ):
        super().__init__(config, PopulateFAISSIndexConfig)
        self.lang_output_dir = (
            Path(self.config.output_dir) / self.config.lang
        ).resolve()
        self.lang_output_dir.mkdir(exist_ok=True)

        fp16 = getattr(self.config, "fp16", False)
        self.dtype = np.float16 if fp16 else np.float32
        self.checkpoint_summary = (
            CheckpointSummary(
                partial_idx=None,
                partial_idx_file=None,
                idx_size_before_populating_embedding=None,
                is_partial_file_valid=False,
                is_partial_idx_valid=False,
            )
            if checkpoint_summary is None
            else checkpoint_summary
        )

    def requirements(self):
        return stopes.core.Requirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=1 if self.config.use_gpu else 0,
            cpus_per_task=self.config.num_cpu,
            timeout_min=48 * 60,
        )

    def array(self):
        return self.config.embedding_files

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        import faiss

        logger = logging.getLogger("stopes.populate_faiss_index")

        # Calculate original size of index (i.e. size of index before we start populating the embedding onto it)
        self.checkpoint_summary.idx_size_before_populating_embedding = (
            self.checkpoint_summary.idx_size_before_populating_embedding
            or faiss.read_index(str(self.config.index)).ntotal
        )

        lang_output_dir = Path(self.lang_output_dir)
        file_name = f"populate_index.{self.config.index_type}.{self.config.lang}.{iteration_index:03d}.data.idx"
        populated_index_path = lang_output_dir / file_name

        if Path(iteration_value).stat().st_size == 0:
            logger.warning(
                f"Within populate_faiss_index module run: Embedding shard is empty, so None is returned. Embedding Shard path is: {iteration_value}"
            )
            return None

        # If both the partial file and index are None OR
        #    both the partial file and index are INVALID
        if (
            self.checkpoint_summary.partial_idx_file is None
            and self.checkpoint_summary.partial_idx is None
        ) or (
            (not self.checkpoint_summary.is_partial_file_valid)
            and (not self.checkpoint_summary.is_partial_idx_valid)
        ):
            # this means we're entering the job for the very first time (haven't even started the first checkpoint)
            self.checkpoint_summary.partial_idx_file = populated_index_path.with_suffix(
                ".checkpoint"
            )
            # since we're going to start the 1st checkpoint, copy the trained index to the partial_idx_file. This will be populated with a single shard
            shutil.copyfile(self.config.index, self.checkpoint_summary.partial_idx_file)
            self.checkpoint_summary.is_partial_file_valid = True

        # By now, at least one of is_partial_idx_valid or is_partial_file_valid must be true at any one time
        try:
            # If the partial_file is valid, the partial_idx should have the same contents as the file
            if self.checkpoint_summary.is_partial_file_valid:
                self.checkpoint_summary.partial_idx = faiss.read_index(
                    str(self.checkpoint_summary.partial_idx_file)
                )
                # Note that partial_idx is overwritten regardless of whether it is valid from before already or not, because, there's an edge case where:
                # both partial_idx and partial_idx_file are valid but partial_idx is one checkpoint ahead of partial_idx_file. So we calibrate both to be on the same checkpoint.
                # This occurs when the job is interrupted right after adding the chunk to partial index but before writing it to the file.
                self.checkpoint_summary.is_partial_idx_valid = True
            else:  # If the partial_idx is valid and the partial_idx_file isn't we write the partial_idx to the file.
                faiss.write_index(
                    self.checkpoint_summary.partial_idx,
                    str(self.checkpoint_summary.partial_idx_file),
                )
                self.checkpoint_summary.is_partial_file_valid = True

            # Since now both the partial idx and file are valid, we can populate the embedding onto the index:
            add_embedding_to_index(
                self.checkpoint_summary,
                iteration_value,
                self.config.embedding_dimensions,
                dtype=self.dtype,
                gpu=self.config.use_gpu,
                chunk_size=getattr(self.config, "chunk_size", 2**14),
                enable_checkpointing=self.config.enable_checkpointing,
            )

        except Exception as e:
            logger.exception(
                f"Error in index population with embeddings: {iteration_value}, & index: {self.config.index}",
            )
            raise e

        # The process is complete: copying completed (checkpointed) index onto return value file
        populated_index_path = self.checkpoint_summary.partial_idx_file.replace(
            populated_index_path
        )
        self.checkpoint_summary.is_partial_file_valid = False
        logger.info(
            f"Populated index, can be found in output file: {populated_index_path}"
        )
        return populated_index_path

    def name(self):
        return f"populate_index.{self.config.index_type}.{self.config.lang}"

    def checkpoint(
        self, *args: tp.Any, **kwargs: tp.Any
    ) -> submitit.helpers.DelayedSubmission:
        return submitit.helpers.DelayedSubmission(
            PopulateFAISSIndexModule(
                config=self.config,
                checkpoint_summary=self.checkpoint_summary
                if self.config.enable_checkpointing
                else None,
            ),
            *args,
            **kwargs,
        )

    def validate(
        self,
        output: tp.Any,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> bool:
        import faiss

        if Path(iteration_value).stat().st_size == 0:
            assert output is None, "embedding is empty, shouldn't populate anything"
            return True
        assert output.exists(), f"index file {output} is missing"
        idx = faiss.read_index(str(output))
        nbex = len(Embedding(iteration_value))
        assert (
            idx.ntotal == nbex
        ), f"expected {nbex} sentences, only found {idx.ntotal} in index {output} populated from {iteration_value}."

        return True


# Reads embeddings from the given file and add them to the index
def add_embedding_to_index(
    checkpoint_summary: CheckpointSummary,
    embeddings_file: Path,
    dim: int,
    dtype=np.float32,
    gpu: bool = True,
    chunk_size: int = 2**14,
    enable_checkpointing: bool = False,
) -> "faiss.Index":
    import faiss

    logger = logging.getLogger("stopes.populate_faiss_index")
    assert checkpoint_summary is not None, "checkpoint_summary must not be None"
    embedding = Embedding(embeddings_file)
    assert isinstance(checkpoint_summary.partial_idx, faiss.Index)
    partial_idx = checkpoint_summary.partial_idx
    n_total_start = partial_idx.ntotal
    if gpu:
        with measure("index on gpu", logger):
            partial_idx = index_to_gpu(partial_idx)

    with embedding.open_for_read(mode="memory") as data:
        # Below, we calculate how much of the embedding is already populated onto the index
        # checkpointed_embedding_starting_row is the starting row to continue on from (every row before this we've already completed/checkpointed)
        checkpointed_embedding_starting_row = (
            partial_idx.ntotal - checkpoint_summary.idx_size_before_populating_embedding
        )

        # length_of_remaining_embedding is the length of the remaining embedding that still needs to be populated onto index
        length_of_remaining_embedding = (
            len(embedding) - checkpointed_embedding_starting_row
        )

        # embedding_iterator tracks total rows added to index from embedding, after the checkpointed_embedding_starting_row
        embedding_iterator = 0

        if length_of_remaining_embedding > 0:
            num_chunks = math.ceil(length_of_remaining_embedding / chunk_size)

            for chunk in np.array_split(
                data[checkpointed_embedding_starting_row:], num_chunks
            ):
                assert (
                    checkpoint_summary.is_partial_idx_valid
                    and checkpoint_summary.is_partial_file_valid
                ), "Both partial idx and partial idx file must be valid before proceeding to work on current checkpoint"

                embedding_iterator += chunk.shape[0]
                should_log = embedding_iterator % 50_000 == 0

                # fp16 currently not supported by FAISS
                if dtype == np.float16:
                    chunk = chunk.astype(np.float32)
                with measure("normalize", logger, enable_log=should_log):
                    faiss.normalize_L2(chunk)

                # Add chunk to partial index (and while this happens, keep is_partial_idx_valid as False)
                checkpoint_summary.is_partial_idx_valid = False
                with measure(
                    f"adding {chunk.shape[0]}/{length_of_remaining_embedding}",
                    logger,
                    enable_log=should_log,
                ):
                    partial_idx.add(chunk)
                checkpoint_summary.is_partial_idx_valid = True

                if enable_checkpointing:
                    with measure("checkpointing", logger, enable_log=should_log):
                        # this partial checkpointing is needed to protect us from preemption
                        # but is quite slow, and when we aren't afraid of preemption we should skip it.
                        # Write partial index to the partial index file (and while this happens, keep is_partial_file_valid as False)
                        checkpoint_summary.is_partial_file_valid = False
                        partial_cpu_idx = (
                            faiss.index_gpu_to_cpu(partial_idx) if gpu else partial_idx
                        )
                        faiss.write_index(
                            partial_cpu_idx,
                            str(checkpoint_summary.partial_idx_file),
                        )
                        checkpoint_summary.is_partial_file_valid = True

    # Write partial index to the partial index file (and while this happens, keep is_partial_file_valid as False)
    checkpoint_summary.is_partial_file_valid = False
    with measure("writing index", logger):
        if gpu:
            partial_idx = faiss.index_gpu_to_cpu(partial_idx)
        faiss.write_index(
            partial_idx,
            str(checkpoint_summary.partial_idx_file),
        )
        checkpoint_summary.is_partial_file_valid = True

    assert (
        partial_idx.ntotal == n_total_start + embedding_iterator
    ), f"population with {embeddings_file} didn't succeed"

    return partial_idx

    def version(self):
        "0.1"
