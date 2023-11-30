# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import asyncio
import typing as tp
from dataclasses import dataclass

import hydra
import pandas as pd

from stopes.core.launcher import Launcher
from stopes.modules.speech.vits_tts_inference import (
    VITSInferenceConfig,
    VITSInferenceModule,
)
from stopes.ust_common.lib.manifests import save_to_tsv


@dataclass
class VITSInferencePipelineConfig:
    launcher: tp.Dict[str, tp.Any]
    vits_inference: VITSInferenceConfig


async def vits_inference(
    config: VITSInferencePipelineConfig,
    launcher: Launcher = None,
):
    if launcher is None:
        launcher = hydra.utils.instantiate(config.launcher)

    module = VITSInferenceModule(config.vits_inference)
    manifest = await launcher.schedule(module)
    if module.config.nshards > 1:
        manifest = pd.concat(manifest)

    save_to_tsv(module.config.output_dir / "manifest.tsv", manifest)


@hydra.main(config_path="conf", config_name="vits_tts_inference")
def main(config: VITSInferencePipelineConfig) -> None:
    asyncio.run(vits_inference(config))


if __name__ == "__main__":
    main()

"""
# input is Chinese text
python -m stopes.pipelines.speech.vits_tts_inference\
        vits_inference.model_path=/???/proj/vits/logs/zh_tts_golden_char_fixed_punc_lr2e-4/G_779000.pth  \
        vits_inference.input_tsv=/???/tmp/zh.tsv \
        vits_inference.output_dir=/???/tmp/zh_vits \
        vits_inference.nshards=1 \
        vits_inference.text_cleaner="to_simplified_zh simplified_zh_to_pinyin pinyin_r5_to_er5 pinyin_u2_to_v pinyin_u_to_v" \
        vits_inference.text_col=tgt_text

# input is prosody-annotated Chinese text
python -m stopes.pipelines.speech.vits_tts_inference\
        vits_inference.model_path=/???/proj/vits/logs/zh_tts_golden_char_fixed_punc_lr2e-4/G_779000.pth  \
        vits_inference.input_tsv=/???/tmp/zh.tsv \
        vits_inference.output_dir=/???/tmp/zh_tts \
        vits_inference.nshards=200 \
        vits_inference.text_cleaner="to_simplified_zh simplified_zh_to_pinyin pinyin_r5_to_er5 pinyin_u2_to_v pinyin_u_to_v" \
        vits_inference.torch_hub_cache=/???/.cache/torch/hub \
        vits_inference.prosody_manipulate=true \
        vits_inference.text_col=markup
"""
