# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import asyncio
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from stopes.core import utils
from stopes.core.launcher import Launcher
from stopes.modules.speech.whisper import WhisperConfig, WhisperModule


@dataclass
class WhisperPipelineConfig:
    launcher: tp.Dict[str, tp.Any]
    whisper: WhisperConfig
    input_manifest: Path
    output_dir: Path
    input_column: int = 0
    sort_inputs: bool = True
    max_shard_size: tp.Optional[int] = None


async def run_asr(
    config: WhisperPipelineConfig,
    launcher: Launcher = None,
):
    """Select the required column from the input manifest, sort it, split into chunks, and feed to Whisper.
    Then combine the results in the original order."""
    config = OmegaConf.merge(OmegaConf.structured(WhisperPipelineConfig), config)  # type: ignore
    if launcher is None:
        launcher = hydra.utils.instantiate(config.launcher)

    # read the input mainfest, extract the right column
    with utils.open(config.input_manifest, "r") as f:
        input_lines = [line.strip().split("\t")[config.input_column] for line in f]
    if config.sort_inputs:
        sort_result = sorted([(line, i) for i, line in enumerate(input_lines)])
        input_columns = [line for line, i in sort_result]
    else:
        sort_result = []
        input_columns = input_lines
    shard_size = config.max_shard_size or len(input_columns)
    input_shards = []
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Split the input into shards
    for i in range(0, len(input_columns), shard_size):
        fname = config.output_dir / f"input_shard_{i}.tsv"
        input_shards.append(str(fname))
        with utils.open(fname, "w") as f:
            for line in input_columns[i : i + shard_size]:
                print(line, file=f)

    # Run the transcription
    config.whisper.shards = input_shards
    config.whisper.output_dir = config.output_dir
    whisper_module = WhisperModule(config.whisper)
    result_shards = await launcher.schedule(whisper_module)

    # join the shards and restore the initial order
    result_texts = []
    for shard in result_shards:
        with utils.open(shard, "r") as f:
            for line in f:
                result_texts.append(line.strip())
    if config.sort_inputs:
        unsorted = sorted(
            [(i, text) for text, (_, i) in zip(result_texts, sort_result)]
        )
        result_texts = [text for i, text in unsorted]

    # write the result
    fname = config.output_dir / "joined_result.tsv"
    with utils.open(fname, "w") as f:
        for line in result_texts:
            print(line, file=f)
    print(
        f"The transcription results ({len(result_texts)} lines) are succesfully stored in {fname}"
    )
    return fname


@hydra.main(config_path="conf", config_name="whisper_asr")
def main(config: WhisperPipelineConfig) -> None:
    asyncio.run(run_asr(config))


if __name__ == "__main__":
    main()
