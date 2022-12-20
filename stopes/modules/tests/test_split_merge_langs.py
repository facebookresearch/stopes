# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import random
from pathlib import Path

from omegaconf import DictConfig

from stopes.core import utils
from stopes.core.launcher import Launcher
from stopes.core.stopes_module import Requirements
from stopes.modules.bitext.mining.merge_shards import (
    MergeShardsConfig,
    MergeShardsModule,
)
from stopes.modules.preprocess.multiproc_bitext_processor import (
    MultiprocBitextProcessorConfig,
    MultiprocBitextProcessorModule,
)
from stopes.modules.preprocess.multiproc_line_processor import (
    MultiprocLineProcessorConfig,
    MultiprocLineProcessorModule,
)


async def test_split_with_meta(tmp_path: Path):
    n_in_shards = 3
    n_out_shards = 2
    input_sizes = []
    input_shards = []
    input_metas = []
    for i in range(n_in_shards):
        shard_size = i * 2 + 100
        input_sizes.append(shard_size)
        text_name, meta_name = (
            tmp_path / f"texts_{i}.tsv.gz",
            tmp_path / f"meta_{i}.tsv.gz",
        )
        input_shards.append(text_name)
        input_metas.append(meta_name)
        with gzip.open(text_name, mode="wt") as f_text, gzip.open(
            meta_name, mode="wt"
        ) as f_meta:
            for line_id in range(shard_size):
                print("text", i, line_id, file=f_text)
                print("meta", i, line_id, file=f_meta)

    launcher = Launcher(
        config_dump_dir=tmp_path / "conf",
        log_folder=tmp_path / "logs",
        cluster="local",
    )
    reqs = Requirements()
    file_processor = MultiprocBitextProcessorModule(
        config=MultiprocBitextProcessorConfig(
            bitext_processor=DictConfig(
                {
                    "_target_": "stopes.modules.preprocess.split_in_shards.SplitInShardsParallelMC",
                    "nb_shards": n_out_shards,
                }
            ),
            custom_name="split_in_shards",
            output_dir=tmp_path / "results",
            outfile_prefix="outfile_prefix",
            shards=[list(p) for p in zip(input_shards, input_metas)],
            requirements=reqs,
            tmp_dir=tmp_path / "temp",
        )
    )

    results = await launcher.schedule(file_processor)
    assert len(results) == n_in_shards
    # reshape [old_shards, data_type, new_shards] into [data_type, new_shards, old_shards]
    out_shards, out_meta, out_counts = [
        [list(old_shards) for old_shards in zip(*data_type)]
        for data_type in zip(*results)
    ]
    assert len(out_counts) == len(out_shards) == len(out_meta) == n_out_shards
    assert input_sizes == [sum(source_count) for source_count in zip(*out_counts)]

    all_out_texts = set()
    for j, out_shard in enumerate(out_shards):
        for i, filename in enumerate(out_shard):
            with utils.open(filename, "r") as f:
                result_lines = f.readlines()
            with utils.open(out_meta[j][i], "r") as f:
                meta_lines = f.readlines()
            # text and meta lines differ in only first word, so the rest should be the same
            assert [line.split(" ", 1)[1] for line in result_lines] == [
                line.split(" ", 1)[1] for line in meta_lines
            ]
            all_out_texts.update(result_lines)
    assert len(all_out_texts) == sum(input_sizes)


async def test_split_without_meta(tmp_path: Path):
    # this test is deliberately decoupled from `test_split_with_meta``
    n_in_shards = 3
    n_out_shards = 2
    input_sizes = []
    input_shards = []
    for i in range(n_in_shards):
        shard_size = i * 2 + 100
        input_sizes.append(shard_size)
        text_name = tmp_path / f"texts_{i}.tsv.gz"
        input_shards.append(text_name)
        with gzip.open(text_name, mode="wt") as f_text:
            for line_id in range(shard_size):
                print("text", i, line_id, file=f_text)

    launcher = Launcher(
        config_dump_dir=tmp_path / "conf",
        log_folder=tmp_path / "logs",
        cluster="local",
    )
    reqs = Requirements()
    file_processor = MultiprocLineProcessorModule(
        config=MultiprocLineProcessorConfig(
            line_processor=DictConfig(
                {
                    "_target_": "stopes.modules.preprocess.split_in_shards.SplitInShardsMC",
                    "nb_shards": n_out_shards,
                }
            ),
            custom_name="split_in_shards",
            output_dir=tmp_path / "results",
            outfile_prefix="outfile_prefix",
            shards=input_shards,
            requirements=reqs,
            tmp_dir=tmp_path / "temp",
        )
    )

    results = await launcher.schedule(file_processor)
    assert len(results) == n_in_shards
    # reshape [old_shards, data_type, new_shards] into [data_type, new_shards, old_shards]
    out_shards, out_counts = [
        [list(old_shards) for old_shards in zip(*data_type)]
        for data_type in zip(*results)
    ]
    assert len(out_counts) == len(out_shards) == n_out_shards
    assert input_sizes == [sum(source_count) for source_count in zip(*out_counts)]

    all_out_texts = set()
    for j, out_shard in enumerate(out_shards):
        for i, filename in enumerate(out_shard):
            with utils.open(filename, "r") as f:
                all_out_texts.update(f.readlines())
    assert len(all_out_texts) == sum(input_sizes)


async def test_merge_bitext(tmp_path: Path):
    inputs = []
    # create input texts with duplicates to test deduplication
    duplicates = [(f"dup_text_en_{i}", f"dup_text_fr_{i}") for i in range(50)]
    unique_input_pairs = set(duplicates)
    for j in range(3):
        text_name, meta_name = (
            tmp_path / f"bitext_{j}.tsv.gz",
            tmp_path / f"bimeta_{j}.tsv.gz",
        )
        inputs.append((text_name, meta_name))
        with utils.open(text_name, mode="wt") as f_text, utils.open(
            meta_name, mode="wt"
        ) as f_meta:
            unique_texts = [
                (f"unique_text_en_{j}_{i}", f"unique_text_fr_{j}_{i}")
                for i in range(50)
            ]
            unique_input_pairs.update(unique_texts)
            score = 2.0
            for text, meta in unique_texts + duplicates:
                # relative margin scores cannot be smaller than 1, and they should be decreasing
                score = score - random.random() * 0.001
                print(score, text, meta, file=f_text, sep="\t")
                print(score, text, meta, "metadata", file=f_meta, sep="\t")

    launcher = Launcher(
        config_dump_dir=tmp_path / "conf",
        log_folder=tmp_path / "logs",
        cluster="local",
    )
    merge_module = MergeShardsModule(
        MergeShardsConfig(
            src_lang="en",
            tgt_lang="fr",
            output_dir=tmp_path / "output",
            pairs=inputs,
        )
    )
    merged_text, merged_meta = await launcher.schedule(merge_module)

    with utils.open(merged_text, "r") as f:
        lines = f.readlines()
    # check that the number of lines after merge equals the number of unique text pairs
    assert len(lines) == len(
        unique_input_pairs
    ), f"after merging, got { len(lines)} lines, expected {len(unique_input_pairs)}"

    # check that all unique text pairs are preserved after merge
    unique_output_pairs = {tuple(line.strip().split("\t")[1:]) for line in lines}
    assert unique_output_pairs == unique_input_pairs

    # check that bitexts are correctly sorted after merge
    scores = [float(line.split("\t")[0]) for line in lines]
    assert sorted(scores, reverse=True) == scores, "scores are not sorted"

    # check that bitexts are aligned with bimeta after merge
    with utils.open(merged_meta, "r") as f:
        meta_lines = [line.strip() for line in f.readlines()]
    assert [
        line.strip() + "\tmetadata" for line in lines
    ] == meta_lines, "merged bitext and bimeta are not aligned"


async def test_merge_bitext_edge_cases(tmp_path: Path):
    input1, input2 = tmp_path / "text1.tsv", tmp_path / "text2.tsv"
    expected = []
    with utils.open(input1, "w") as f1, utils.open(input2, "w") as f2:
        # full duplicates: only one is preserved
        print("9\ta\tb", file=f1)
        print("9\ta\tb", file=f2)
        print("9\ta\tb", file=f2)
        expected.append("9\ta\tb")

        # duplicates with different scores: the one with the highest score is preserved
        print("8\tc\td", file=f2)
        print("7\tc\td", file=f1)
        print("6\tc\td", file=f1)
        print("5\tc\td", file=f2)
        expected.append("8\tc\td")

        # same score, different texts: all are preserved
        print("4\te\tf", file=f1)
        print("4\te\tg", file=f2)
        print("4\th\tg", file=f2)
        expected.append("4\te\tf")
        expected.append("4\te\tg")
        expected.append("4\th\tg")

    launcher = Launcher(
        config_dump_dir=tmp_path / "conf",
        log_folder=tmp_path / "logs",
        cluster="local",
    )
    merge_module = MergeShardsModule(
        MergeShardsConfig(
            src_lang="en",
            tgt_lang="fr",
            output_dir=tmp_path / "output",
            pairs=[(input1, None), (input2, None)],
        )
    )
    merged_text, merged_meta = await launcher.schedule(merge_module)
    assert merged_meta is None

    with utils.open(merged_text, "r") as f:
        result = [f.strip() for f in f.readlines()]
    assert result == expected
