# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import os
from typing import Dict, Optional, Tuple

from submitit import AutoExecutor

import stopes.pipelines.prepare_data.data_types as data_types
from stopes.pipelines.prepare_data.cache import cache_step, cache_step_sync
from stopes.pipelines.prepare_data.utils import execute_in_shell, split_direction

logger = logging.getLogger(__name__)


def build_preprocess_cmds(
    preprocess_config: data_types.PreprocessingConfig,
    lang: str,
    data_tag: Optional[str] = None,
) -> str:
    """
    Builds custom preprocessing commands based on preprocessing config
    Returns command string
    """
    cmds = []
    if preprocess_config.sample_size:
        cmds.append(f"head -n {preprocess_config.sample_size}")

    if preprocess_config.max_tokens:
        cmds.append(
            f"awk -vLEN={preprocess_config.max_tokens} '{{if (NF <= LEN) print}}'"
        )
    moses_config = preprocess_config.moses_config
    moses_dir = moses_config.script_directory
    if moses_config.normalize_punctuation:
        cmds.append(f"perl {moses_dir}/normalize-punctuation.perl -l {lang}")
    if moses_config.lowercase:
        cmds.append(f"perl {moses_dir}/lowercase.perl")
    if moses_config.remove_non_printing_chars:
        cmds.append(f"perl {moses_dir}/remove-non-printing-char.perl")
    if moses_config.deescape_special_chars:
        cmds.append(f"perl {moses_dir}/deescape-special-chars.perl")
    if preprocess_config.tag_data and data_tag is not None:
        cmds.append(f"sed -e 's/^/{data_tag} /'")
    pipe = " | "
    return f"{pipe}{pipe.join(cmds)}"


@cache_step_sync("retrieve_direction")
def retrieve_direction_step(
    all_corpora_map: Dict[str, data_types.CorporaMap],
    output_prefix: str,
    preprocess_config: data_types.PreprocessingConfig,
    direction: str,
    executor: Optional[AutoExecutor],
    tag: str,
    output_dir: str,
    custom_step_name: str,
):
    source_lang, target_lang = split_direction(direction)
    source_output = f"{output_prefix}.{direction}.{source_lang}"
    target_output = f"{output_prefix}.{direction}.{target_lang}"
    if os.path.exists(source_output):
        execute_in_shell(f"rm {source_output}")
    if os.path.exists(target_output):
        execute_in_shell(f"rm {target_output}")
    jobs = []
    for corpus, corpus_paths in all_corpora_map[direction]["values"].items():
        if os.path.exists(f"{source_output}.{corpus}"):
            execute_in_shell(f"rm {source_output}.{corpus}")
        if os.path.exists(f"{target_output}.{corpus}"):
            execute_in_shell(f"rm {target_output}.{corpus}")
        is_gzip = corpus_paths["is_gzip"]
        local_source = corpus_paths["source"]
        local_target = corpus_paths["target"]
        data_tag = corpus_paths["data_tag"]
        cat_cmd = "zcat" if is_gzip else "cat"
        src_preprocess_cmds = (
            build_preprocess_cmds(preprocess_config, source_lang, data_tag=data_tag)
            if preprocess_config.preprocess_source
            else ""
        )
        tgt_preprocess_cmds = (
            build_preprocess_cmds(preprocess_config, target_lang)
            if preprocess_config.preprocess_target
            else ""
        )
        src_command = (
            f"{cat_cmd} {local_source} {src_preprocess_cmds} >> {source_output}"
        )
        tgt_command = (
            f"{cat_cmd} {local_target} {tgt_preprocess_cmds} >> {target_output}"
        )
        execute_in_shell(src_command)
        execute_in_shell(tgt_command)
    return f"Done {tag} {direction}"


async def retrieve_data(
    all_corpora_map: Dict[str, data_types.CorporaMap],
    output_prefix: str,
    preprocess_config: data_types.PreprocessingConfig,
    tag: str,
    output_dir: str,
    executor: Optional[AutoExecutor],
) -> Tuple[Dict[str, data_types.ParallelDataset]]:
    """
    Retrieve training data from multiple sources
    Returns paths to the retrieved train data
    """

    # submit all directions as part of the same array
    jobs = []
    # with executor.batch():
    if all_corpora_map:
        for direction in all_corpora_map.keys():
            logger.info(f"Scheduling retrieve data for {direction} job")
            job = executor.submit(
                retrieve_direction_step,
                all_corpora_map=all_corpora_map,
                output_prefix=output_prefix,
                preprocess_config=preprocess_config,
                direction=direction,
                tag=tag,
                executor=executor,
                output_dir=output_dir,
                custom_step_name=f"retrieve_data_step.{tag}.{direction}",
            )
            jobs.append(job.result())
    logger.info(f"All jobs for retrieve data are done ... now merging files")

    concatenated_paths = {}
    if all_corpora_map:
        for direction in all_corpora_map.keys():
            source_lang, target_lang = direction.split("-")
            source_output = f"{output_prefix}.{direction}.{source_lang}"
            target_output = f"{output_prefix}.{direction}.{target_lang}"
            for corpus, corpus_paths in all_corpora_map[direction]["values"].items():
                if os.path.exists(f"{source_output}.{corpus}") and os.path.exists(
                    f"{target_output}.{corpus}"
                ):
                    src_command = f"cat {source_output}.{corpus} >> {source_output}"
                    tgt_command = f"cat {target_output}.{corpus} >> {target_output}"
                    execute_in_shell(src_command)
                    execute_in_shell(tgt_command)
                    os.remove(f"{source_output}.{corpus}")
                    os.remove(f"{target_output}.{corpus}")

            concatenated_paths[direction] = data_types.ParallelDataset(
                source=source_output, target=target_output
            )

    if tag.split("_")[0] == "train":
        sampled_concatenated_paths = {}
        if all_corpora_map:
            for direction in all_corpora_map.keys():
                source_lang, target_lang = direction.split("-")
                source_output = f"{output_prefix}.{direction}.{source_lang}"
                target_output = f"{output_prefix}.{direction}.{target_lang}"
                sampled_source_output = f"{output_prefix.replace('train', 'sampled_train')}.{direction}.{source_lang}"
                sampled_target_output = f"{output_prefix.replace('train', 'sampled_train')}.{direction}.{target_lang}"
                src_command = (
                    f'sed -e "1000q" {source_output} > {sampled_source_output}'
                )
                tgt_command = (
                    f'sed -e "1000q" {target_output} > {sampled_target_output}'
                )
                execute_in_shell(src_command)
                execute_in_shell(tgt_command)
                sampled_concatenated_paths[direction] = data_types.ParallelDataset(
                    source=sampled_source_output, target=sampled_target_output
                )
        return (concatenated_paths, sampled_concatenated_paths)
    else:
        return (concatenated_paths,)
