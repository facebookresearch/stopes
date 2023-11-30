# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from dataclasses import dataclass
from pathlib import Path

from omegaconf import MISSING

from stopes.core import Requirements, StopesModule

logger = logging.getLogger("hubert_extraction")

from pathlib import Path

import fairseq
import pandas as pd
import torch
import torch.nn.functional as F
from fairseq.data.audio.audio_utils import get_features_or_waveform
from tqdm import tqdm


def load_hubert_model_eval(hubert_checkpoint_path: Path, device: str = "cuda"):
    (
        model,
        cfg,
        task,
    ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([hubert_checkpoint_path])

    model = model[0].eval().to(device)
    return model, cfg, task


def get_audio_paths(tsv_path: str, column_name: str):
    """Get absolute audio paths for every sample id. This function handles both relative and absolute paths written in TSV manifests.

    Args:
        tsv_path (str): Abosulte path to the tsv manifest;
        column_name (str): Name of the tsv column where the audio path is written.
    """
    manifest = pd.read_csv(tsv_path, sep="\t", quoting=3, index_col="id")
    paths = []
    for idx, row in manifest.iterrows():
        if Path(row[column_name]).is_absolute():
            paths.append([idx, row[column_name]])
        else:
            # convention: the audio is located in the same dir as the tsv file!
            paths.append([idx, (Path(tsv_path).parent / row[column_name]).as_posix()])

    return paths


@dataclass
class HubertExtractionConfig:
    input_tsv_path: str = MISSING
    output_tsv_path: str = MISSING
    input_column_name: str = MISSING
    hubert_checkpoint_path: str = MISSING
    hubert_layer: int = MISSING
    kmeans_path: str = MISSING
    dump_features: bool = False
    dump_meta: bool = False
    min_chunk: int = 300
    max_chunk: int = 1600000
    nshards: int = 1
    requirements: Requirements = Requirements(
        nodes=1,
        tasks_per_node=1,
        gpus_per_node=1,
        cpus_per_task=10,
        timeout_min=240,
    )
    custom_name: str = ""


class HubertExtractionModule(StopesModule):
    def __init__(
        self,
        config: HubertExtractionConfig = HubertExtractionConfig(),
        validate_config: bool = False,
    ):
        super().__init__(
            config,
            # TODO: always validate that config is a LineProcessorConfig
            # This is not possible currently because several config files add extra args
            # to make it easier to type the config
            config_class=HubertExtractionConfig if validate_config else None,
        )
        # we do basic checkpointing with submitit Checkpointable which will store the state of this
        # callable. The basic idea here is to remember the last line processed
        # self.processed_lines = processed_lines
        Path(config.output_tsv_path).parent.mkdir(exist_ok=True)

    def array(self) -> tp.List[str]:
        return [str(i) for i in range(self.config.nshards)]

    def requirements(self) -> Requirements:
        reqs = self.config.requirements
        if not isinstance(reqs, Requirements):
            # Performe conversion if needed
            return Requirements(**reqs)
        return reqs

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> tp.Any:
        shard_id = 0
        if iteration_value is not None:
            shard_id = int(iteration_value)

        Path(self.config.output_tsv_path).parent.mkdir(parents=True, exist_ok=True)

        output_tsv_path = Path(self.config.output_tsv_path)
        output_prefix = output_tsv_path.parent / (
            output_tsv_path.stem + f"_{shard_id}_{self.config.nshards}"
        )

        # picked from examples.hubert.simple_kmeans.feature_utils
        def get_shard_range(tot, nshard, rank):
            assert rank < nshard and rank >= 0, f"invaid rank/nshard {rank}/{nshard}"
            start = round(tot / nshard * rank)
            end = round(tot / nshard * (rank + 1))
            assert start < end, f"start={start}, end={end}"
            return start, end

        model, _, task = load_hubert_model_eval(
            Path(self.config.hubert_checkpoint_path).resolve()
        )
        kmeans_executor = ApplyKmeans(self.config.kmeans_path)

        audio_paths = get_audio_paths(
            self.config.input_tsv_path, self.config.input_column_name
        )
        start, end = get_shard_range(len(audio_paths), self.config.nshards, shard_id)

        logger.info(
            f"rank {shard_id} of {self.config.nshards}, process {end-start} "
            f"({start}-{end}) out of {len(audio_paths)}"
        )

        idx_to_length = []

        if self.config.dump_meta:
            label_entropies = [0.0] * kmeans_executor.km_model.n_clusters
            label_count = [0] * kmeans_executor.km_model.n_clusters
            label_neighbors = torch.zeros(
                (
                    kmeans_executor.km_model.n_clusters,
                    kmeans_executor.km_model.n_clusters,
                )
            )

        labels_dump = NpyAppendArray(output_prefix.as_posix() + f".labels.npy")
        if self.config.dump_features:
            features_dump = NpyAppendArray(output_prefix.as_posix() + f".features.npy")

        start_increment = 0
        for idx, path in tqdm(audio_paths[start:end]):
            wav = get_features_or_waveform(
                path, need_waveform=True, use_sample_rate=task.cfg.sample_rate
            )
            if wav.ndim == 2:
                wav = wav.mean(-1)
            assert wav.ndim == 1, wav.ndim

            with torch.inference_mode():
                x = torch.from_numpy(wav).float().cuda()
                if task.cfg.normalize:
                    x = F.layer_norm(x, x.shape)
                x = x.view(1, -1)

                feat = []
                for start in range(0, x.size(1), self.config.max_chunk):
                    x_chunk = x[:, start : start + self.config.max_chunk]
                    expand_last_chunk = False
                    if (
                        x[:, start + self.config.max_chunk :].shape[1]
                        < self.config.min_chunk
                    ):
                        x_chunk = x[:, start:]
                        expand_last_chunk = True
                    try:
                        feat_chunk, _ = model.extract_features(
                            source=x_chunk,
                            padding_mask=None,
                            mask=False,
                            output_layer=self.config.hubert_layer,
                        )
                    except RuntimeError:
                        raise RuntimeError(
                            f"Maybe input chunk is too small ({x_chunk.shape[1]})."
                        )
                    feat.append(feat_chunk)
                    if expand_last_chunk:
                        break

                feat = torch.cat(feat, dim=1).squeeze(0)  # type: ignore
                labels, meta_labels = kmeans_executor(
                    feat, return_meta=self.config.dump_meta
                )

                if self.config.dump_meta:
                    for label, ent, neighbors in zip(
                        labels, meta_labels["topk_ents"], meta_labels["topk_labels"]
                    ):
                        label_entropies[label] += ent
                        label_count[label] += 1
                        label_neighbors[label, neighbors] += 1

                labels_dump.append(labels)
                idx_to_length.append(
                    {
                        "id": idx,
                        "length": labels.size,
                        "seq_start": start_increment,
                        "seq_end": start_increment + labels.size,
                        f"{self.config.input_column_name.replace('audio', 'text')}": " ".join(
                            [str(i) for i in labels.tolist()]
                        ),
                    }
                )
                start_increment = start_increment + labels.size
                if self.config.dump_features:
                    features_dump.append(feat.cpu().numpy())  # type: ignore

        labels_dump.close()
        if self.config.dump_features:
            features_dump.close()

        if self.config.dump_meta:
            with open(output_prefix.as_posix() + ".top10_entropies.pkl", "wb") as f:
                pickle.dump(label_entropies, f)
            with open(output_prefix.as_posix() + ".counts.pkl", "wb") as f:
                pickle.dump(label_count, f)

            torch.save(label_neighbors, output_prefix.as_posix() + ".top10_labels.pt")

        output_manifest = pd.DataFrame(idx_to_length).set_index("id")

        output_path = (
            output_prefix.as_posix() + Path(self.config.output_tsv_path).suffix
        )
        output_manifest.to_csv(
            output_path,
            sep="\t",
            quoting=3,
        )

        logger.info(
            f"Kmeans Hubert units / labels were extracted and written to {output_path}."
        )
        return output_path

    @staticmethod
    def version() -> str:
        return "0.3"


import logging
import pickle
from argparse import ArgumentParser
from pathlib import Path

import joblib
import pandas as pd
import torch
import torch.nn.functional as F
from fairseq.data.audio.audio_utils import get_features_or_waveform
from npy_append_array import NpyAppendArray
from tqdm import tqdm


class ApplyKmeans(object):
    """Computes l2 distance batched per sequence.
    |feat - cent|_2 = \sqrt( \sum_i feat_i^2 - 2*sum_i feat_i*cent_i + \sum_i cent+i^2 ),
        centroids_squared = \sum_i cent+i^2, and is pre-computed

    code adjusted from fairseq.examples.hubert.simple_kmeans.dump_km_label
    """

    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np**2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x, return_meta=False):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        dist = x.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
        if return_meta:
            topk_dist, topk_labels = dist.topk(k=10, largest=False, dim=1)
            topk_label_distribution = topk_dist.log_softmax(dim=1)
            entropies = (
                -(topk_label_distribution * topk_label_distribution.exp())
                .sum(dim=1)
                .cpu()
                .numpy()
            )
            label_meta = {
                "topk_labels": topk_labels,
                "topk_ents": entropies,
            }
        else:
            label_meta = None

        labels = dist.argmin(dim=1).cpu().numpy()

        return labels, label_meta


def dedup_line(line):
    units = []
    last_unit = None
    for unit in line.strip().split():
        if unit != last_unit:
            last_unit = unit
            units.append(unit)
    return " ".join(units)


def dedup_unit_tsv(
    input_manifest: Path,
    output_manifest: Path,
    unit_column: str = "tgt_text",
) -> Path:
    df = pd.read_csv(input_manifest, sep="\t", quoting=3)
    df[unit_column] = df[unit_column].apply(lambda x: dedup_line(x))
    df.to_csv(output_manifest, sep="\t", quoting=3, index=None)
    logger.info(f"Output {len(df)} rows to {output_manifest}")
    return output_manifest
