# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import filecmp
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import faiss
import omegaconf
import pytest
import submitit

from stopes.core import utils
from stopes.modules.bitext.indexing.populate_faiss_index import (
    PopulateFAISSIndexConfig,
    PopulateFAISSIndexModule,
    add_embedding_to_index,
)
from stopes.utils.embedding_utils import Embedding

from .test_modules_utils import (
    generate_saved_embedding,
    test_dim,
    test_dtype,
    test_idx_type,
    test_lang,
)
from .test_train_index_port import generate_train_index

test_cluster = "local"


def generate_populated_index(
    embedding_outfile: Path, trained_index_out_file: Path, dir_name: Path
) -> Path:
    test_iteration_index = 0
    populated_index = (
        dir_name
        / f"populate_index.{test_idx_type}.{test_lang}.{test_iteration_index}.data.idx"
    )
    shutil.copyfile(trained_index_out_file, populated_index)

    index_loaded = faiss.read_index(str(trained_index_out_file))
    returned_index = add_embedding_to_index(
        index_loaded, embedding_outfile, test_dim, test_dtype
    )

    faiss.write_index(returned_index, str(populated_index))

    return populated_index


@pytest.mark.skip(reason="slow")
def test_generate_populated_index(tmp_path: Path):
    use_gpu = True

    embedding_outfile = tmp_path / "embedding.bin"
    generate_saved_embedding(file=embedding_outfile)

    trained_index_out_file = generate_train_index(tmp_path, use_gpu, embedding_outfile)
    assert (
        trained_index_out_file.exists()
    ), f"index file {trained_index_out_file} missing"

    populated_index = generate_populated_index(
        embedding_outfile, trained_index_out_file, tmp_path
    )
    assert populated_index.exists(), f"populated index file {populated_index} missing"

    read_populated_index = faiss.read_index(str(populated_index))
    nbex = len(Embedding(embedding_outfile, test_dim))

    assert read_populated_index.ntotal == nbex


def _run_populate_job(
    populate_test_config: PopulateFAISSIndexConfig,
    interrupt: bool,
    sleep_time_before_interrupt: int = 100,
):
    """
    This function runs a populate index module job and waits for result.
    If interrupt parameter is True, it will interrupt the job to trigger the checkpoint method
    """
    output_dir = populate_test_config.config.output_dir
    path_for_embedding = populate_test_config.config.embedding_files[0]

    populate = PopulateFAISSIndexModule(populate_test_config.config, None)
    executor = submitit.AutoExecutor(folder=output_dir, cluster=test_cluster)
    executor.update_parameters(timeout_min=60)
    job = executor.submit(populate, path_for_embedding)

    if interrupt:
        # We let the unittest sleep a little to ensure the job is in the midst of running
        time.sleep(sleep_time_before_interrupt)
        assert job.state == "RUNNING"
        # Since now we know the job is running, we can interrupt
        job._interrupt()

    return job.result()


@pytest.mark.skip(reason="slow")
def test_population_checkpointing(tmp_path: Path):
    """
    This test ensures that the outputs of running the job twice,
    (with and without checkpoints) is identical
    """
    populate_test_config = omegaconf.OmegaConf.create(
        {
            "_target_": "examples.stopes.modules.bitext.indexing.populate_faiss_index.PopulateFAISSIndexModule",
            "config": {
                "lang": "bn",
                "index": "???",  # will be generated
                "index_type": test_idx_type,
                "embedding_files": [],  # will be generated
                "output_dir": "???",  # will be tmp dir
                "num_cpu": 40,
                "embedding_dimensions": test_dim,
            },
        }
    )
    test_chkpt_embedding_length = 1310720
    # Embedding is divided into chunks, each of size close to 2^14.
    # So emb length of 1310720 means approx 80 checkpoints (since 1310720/(2^14) = 80).
    # This gives enough time to interrupt in the midst of the job

    # first generate an embedding and index
    embedding_outfile = tmp_path / "embedding.bin"
    generate_saved_embedding(file=embedding_outfile)
    index_out_file = generate_train_index(tmp_path, True, embedding_outfile)
    populate_test_config.config.index = str(index_out_file)

    # generate new embedding to populate onto the index
    generated_embedding_file = tmp_path / "generated_embedding_to_populate.bin"
    generate_saved_embedding(
        file=generated_embedding_file, emb_length=test_chkpt_embedding_length
    )
    populate_test_config.config.embedding_files.append(str(generated_embedding_file))

    # Now run two jobs: one with and one without interruption
    # Job with an interruption
    with utils.clone_config(populate_test_config) as config_with_interrupt:
        output_dir = tmp_path / "job_with_interruption"
        output_dir.mkdir()
        config_with_interrupt.config.output_dir = str(output_dir)
    idx_with_interrupt = _run_populate_job(config_with_interrupt, interrupt=True)

    # Job wihout interruption
    with utils.clone_config(populate_test_config) as config_no_interrupt:
        output_dir = tmp_path / "job_no_interruption"
        output_dir.mkdir()
        config_no_interrupt.config.output_dir = str(output_dir)
    idx_no_interrupt = _run_populate_job(config_no_interrupt, interrupt=False)

    are_files_identical = filecmp.cmp(
        idx_with_interrupt,
        idx_no_interrupt,
        shallow=False,
    )
    assert are_files_identical, "Mismatch between the two indexes."
