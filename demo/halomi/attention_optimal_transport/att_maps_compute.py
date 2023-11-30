# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

import stopes
from stopes.core.stopes_module import Requirements, StopesModule
from stopes.eval.alti.alti_metrics.alti_metrics_utils import binarize_pair
from stopes.eval.alti.alti_metrics.nllb_alti_detector import load_nllb_model


def get_attention_maps(data, alti_hub):
    maps = []
    for row in tqdm(data.itertuples(), total=data.shape[0]):
        src_lang, tgt_lang = row.direction.split("-")
        st, pt, tt = binarize_pair(
            alti_hub, row.src, row.mt, src_lang=src_lang, tgt_lang=tgt_lang
        )

        with torch.inference_mode():
            logits, out = alti_hub.models[0].forward(
                src_tokens=st.unsqueeze(0).to(alti_hub.device),
                prev_output_tokens=tt.unsqueeze(0).to(alti_hub.device),
                src_lengths=torch.tensor(st.shape).to(alti_hub.device),
            )
            maps.append(out["attn"][0][0].cpu().numpy().mean(0))
    return maps


@dataclass
class ScoreConfig:
    input_files: tp.List[Path]
    output_dir: Path
    model_data_dir: str
    model_spm_path: str
    model_checkpoint_path: str


class AttentionScoreModule(StopesModule):
    def __init__(self, config):
        super().__init__(config, ScoreConfig)
        self.config: ScoreConfig = config

    def requirements(self) -> Requirements:
        return Requirements(gpus_per_node=1)

    def array(self):
        return self.config.input_files

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> tp.Any:
        if iteration_value is not None:
            input_file = iteration_value
        else:
            input_file = self.config.input_files[0]
        output_file = (
            self.config.output_dir / input_file.with_suffix(".attmaps.npy").name
        )

        data = pd.read_csv(input_file, sep="\t")
        alti_hub = load_nllb_model(
            checkpoint=Path(self.config.model_checkpoint_path),
            data_dir=Path(self.config.model_data_dir),
            spm=Path(self.config.model_spm_path),
            src_lang="eng_Latn",
            tgt_lang="eng_Latn",
        )
        alti_hub.cuda()

        maps = get_attention_maps(data, alti_hub)

        np.save(output_file, maps)
        return output_file

    def validate(
        self,
        output: tp.Any,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> bool:
        return output.exists()


async def main(
    input_dir,
    output_dir,
    model_data_dir,
    model_spm_path,
    model_checkpoint_path,
    launcher_cluster,
    launcher_partiton,
):
    input_files = list(Path(input_dir).iterdir())
    output_dir = Path(output_dir)
    conf = OmegaConf.structured(
        ScoreConfig(
            input_files=input_files,
            output_dir=output_dir,
            model_data_dir=model_data_dir,
            model_spm_path=model_spm_path,
            model_checkpoint_path=model_checkpoint_path,
        )
    )
    scorer = AttentionScoreModule(conf)
    print(f"Processing {len(input_files)} files...")
    print(input_files[:5])
    launcher = stopes.core.Launcher(
        log_folder="executor_logs",
        cluster=launcher_cluster,
        partition=launcher_partiton,
        max_jobarray_jobs=1000,
    )
    shards = await launcher.schedule(scorer)
    return shards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        help="directory with input tsv files (with src, mt and direction columns)",
    )
    parser.add_argument("--output-dir", type=str, help="directory for output files")
    parser.add_argument("--model-data-dir", type=str, help="NLLB data directory")
    parser.add_argument("--model-spm-path", type=str, help="NLLB tokenizer path")
    parser.add_argument(
        "--model-checkpoint-path", type=str, help="NLLB checkpoint path"
    )
    parser.add_argument(
        "--launcher-cluster",
        type=str,
        help="launcher cluster, typically slurm or local",
    )
    parser.add_argument("--launcher-partiton", type=str, help="slurm partition")
    args = parser.parse_args()
    asyncio.run(main(**vars(args)))
