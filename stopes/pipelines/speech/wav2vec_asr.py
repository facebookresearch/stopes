# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import dataclasses
import typing as tp

import hydra
from omegaconf import OmegaConf

from stopes.modules.speech.wav2vec.asr import ASRConfig, Wav2vecASR


@dataclasses.dataclass
class ASRPipelineConfig(ASRConfig):
    launcher: tp.Any = None


async def run_asr(config: ASRPipelineConfig):
    launcher = hydra.utils.instantiate(config.launcher)
    asr_config_args = OmegaConf.to_container(config, resolve=True)
    assert isinstance(asr_config_args, dict), f"Invalid configuration: {config}"
    asr_config_args.pop("launcher")
    asr_config = ASRConfig(**asr_config_args)  # type: ignore
    asr = Wav2vecASR(asr_config)
    return await launcher.schedule(asr)


@hydra.main(version_base="1.1", config_path="conf", config_name="wav2vec_asr")
def main(config: ASRPipelineConfig) -> None:
    asyncio.run(run_asr(config))


if __name__ == "__main__":
    main()
