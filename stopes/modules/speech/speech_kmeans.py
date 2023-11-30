# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import typing as tp
from dataclasses import dataclass

import faiss
import numpy as np
from omegaconf import MISSING

from stopes.core.stopes_module import Requirements, StopesModule


@dataclass
class SpeechKMeansJob:
    layer: int
    km_size: int
    km_path: str
    feat_path: str


@dataclass
class SpeechKMeansConfig:
    niter: int
    max_points_per_centroid: int
    requirements: Requirements
    speech_kmeans_jobs: tp.List[SpeechKMeansJob] = MISSING


class SpeechKMeans(StopesModule):
    def __init__(
        self,
        config: SpeechKMeansConfig,
    ):
        super().__init__(config, SpeechKMeansConfig)
        self.config: SpeechKMeansConfig

    def array(self):
        return self.config.speech_kmeans_jobs

    def requirements(self):
        return self.config.requirements

    def faiss_kmeans(
        self,
        feat: np.ndarray,
        k: int,
        ngpu: int,
        niter: int,
        max_points_per_centroid: int,
    ):
        """
        Utilizes the FAISS library to perform K-means clustering on a given
        feature matrix (feat) with a specified number of clusters (k) and GPUs (ngpu).
        The clustering process runs for a specified number of iterations (niter).

        Parameters
        ----------
        feat : np.array
            The feature matrix of shape (N, d) to cluster.
        k : int
            The number of clusters to form.
        ngpu : int
            The number of GPUs to use for clustering.
        niter : int
            The number of iterations to run the K-means clustering algorithm.
        max_points_per_centroid : int
            The number of dataset samples per centroid, internally Faiss subsamples
            a dataset of size `max_points_per_centroid * k`.

        Returns
        -------
        result : np.array
            The centroids of the formed clusters, with shape (k, d).
        """
        d = feat.shape[1]
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = niter
        clus.max_points_per_centroid = max_points_per_centroid

        res = [faiss.StandardGpuResources() for _ in range(ngpu)]
        flat_config = []
        for i in range(ngpu):
            cfg = faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = feat.dtype == np.float16
            cfg.device = i
            flat_config.append(cfg)

        if ngpu == 1:
            index = faiss.GpuIndexFlatL2(res[0], d, flat_config[0])
        else:
            indexes = [
                faiss.GpuIndexFlatL2(res[i], d, flat_config[i]) for i in range(ngpu)
            ]
            index = faiss.IndexReplicas()
            for sub_index in indexes:
                index.addIndex(sub_index)

        clus.train(feat, index)
        centroids = faiss.vector_float_to_array(clus.centroids)
        result = centroids.reshape(k, d)
        return result

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        assert isinstance(
            iteration_value, SpeechKMeansJob
        ), "iteration_value must be an instance of SpeechKMeansJob"
        speech_kmeans_job: SpeechKMeansJob = iteration_value
        self.logger = logging.getLogger("stopes.modules.speech.speech_kmeans")
        ngpu = faiss.get_num_gpus()
        self.logger.info(
            f"Training k-means for layer: {speech_kmeans_job.layer} on {ngpu} GPUs."
        )
        feat = np.load(speech_kmeans_job.feat_path, mmap_mode="r")
        self.logger.info(f"{feat.shape=}, {feat.dtype=}")
        result = self.faiss_kmeans(
            feat,
            speech_kmeans_job.km_size,
            ngpu,
            self.config.niter,
            self.config.max_points_per_centroid,
        )
        np.save(speech_kmeans_job.km_path, result)
        return speech_kmeans_job.km_path
