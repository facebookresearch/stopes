# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections
import dataclasses
import logging
import typing as tp
from pathlib import Path

import faiss
import fasttext
import numpy as np
import numpy.typing
import sentencepiece
import xxhash

import stopes.core
import stopes.modules.preprocess.laser_sentence_encoder as laser_sentence_encoder
from stopes.core import utils
from stopes.modules.preprocess.line_processor import (
    LineProcessorCallback,
    LineProcessorConfig,
)
from stopes.pipelines.monolingual.utils import text_normalizer

log = logging.getLogger("stopes.laser_scorer")

FASTTEXT_LABEL_PREFIX = len("__label__")


@dataclasses.dataclass
class LaserScorerConfig:
    input_file: Path
    model_path: Path
    spm_path: Path
    threshold: float = 0.0
    max_tokens: int = 256
    gpu: bool = False
    fp16: bool = False
    laser_margin: str = ""
    laser_margin_knn: int = 5
    src_lang: str = ""
    tgt_lang: str = ""
    fasttext_model: str = ""
    spm_vocab_path: str = ""
    output_file: str = ""
    columns: str = "src_text,tgt_text"


# Note: this notations are for humans, numpy API is mostly untyped.
# hello
Embedding = np.typing.NDArray[np.float32]


class KNN(tp.NamedTuple):
    distances: np.typing.NDArray[np.float32]
    indices: np.typing.NDArray[np.int32]

    @staticmethod
    def zeros(*, n: int, k: int) -> "KNN":
        idx_dtype = np.uint32 if n <= 2**32 else np.uint64
        return KNN(
            np.zeros((n, k), dtype=np.float32), np.zeros((n, k), dtype=idx_dtype)
        )


class Corpus(tp.NamedTuple):
    indices: tp.List[int]
    uniq_sents: tp.List[str]
    uniq_embs: Embedding = None  # type: ignore

    def sent(self, i) -> str:
        return self.uniq_sents[self.indices[i]]

    def emb(self, i) -> np.ndarray:
        return self.uniq_embs[self.indices[i]]

    def append(self, sent: str, cache: tp.Dict[str, int]) -> int:
        n = len(self.uniq_sents)
        idx = cache.get(sent, n)
        self.indices.append(idx)
        if idx == n:
            # This is a new sentence, append it and update the cache
            self.uniq_sents.append(sent)
            cache[sent] = idx

        return idx


class LaserScorerModule(stopes.core.StopesModule):
    """Loads dirty bitext from a file, and uses Laser to clean it up.

    This is a reimplementation of the code in LASER/source/mine_bitexts.py

    Which was called with the following cmd:
    python LASER/source/mine_bitexts.py \
        --src-lang en --trg-lang ha \
        --output train-all.laser2 \
        --mode score --retrieval max --margin ratio -k 4 \
        --verbose --gpu --unify \
        --src-embeddings train-all.laser2.en.emb \
        --trg-embeddings train-all.laser2.ha.emb \
        ../train-all.en ../train-all.ha.no-tag

    Main difference is that here we take care of computing the embeddings on the fly.

    The algorithm is as follow:
        * Take a big corpus of dirty bitext.
        * Dedup bitext.
        * Run Laser based mining on the bitext, using a Faiss flat index, margin ratio,
            4 knn, and "retrieval-max".
        * Use the mining score to decide whether or not to keep the input bitext.

    This algorithm works best with a lot of bitext, otherwise the knn statistics will be noisy.
    So do concatenate severals sources before putting them through.

    This implementation also detects duplicate sentences in the input corpus,
    and only computes embedding and distances once for them.
    """

    def __init__(self, config: "LaserScorerConfig"):
        super().__init__(config, LaserScorerConfig)
        config = self.config
        assert config.input_file.exists(), f"input_file {config.input_file} not found."
        assert config.model_path.exists(), f"model_path {config.model_path} not found."
        if config.output_file:
            self.outfile = Path(config.output_file)
        else:
            self.outfile = config.input_file.with_suffix(
                f".scored{config.input_file.suffix}"
            )
        assert (
            self.outfile.parent.exists()
        ), f"output_file dir {config.output_file} doesn't exist."
        self.sentences: tp.List[str] = []

        lid_params = [config.fasttext_model, config.src_lang, config.tgt_lang]
        self.want_lid_filtering = any(lid_params)
        if self.want_lid_filtering:
            assert all(
                lid_params
            ), "If you want LID filtering, you must provide following config: [fasttext_model, src_lang, tgt_lang]"
            assert Path(config.fasttext_model).exists()
            # OmegaDict is amazingly slow
            self.src_lang = config.src_lang
            self.tgt_lang = config.tgt_lang

    def requirements(self) -> stopes.core.Requirements:
        return stopes.core.Requirements(
            mem_gb=100,
            gpus_per_node=1 if self.config.gpu else 0,
            cpus_per_task=5,
            timeout_min=3 * 24 * 60,
        )

    def _prepare(self):
        config = self.config

        spm_model, spm_vocab = laser_sentence_encoder.LaserSentenceEncoder.gather_spm(
            encoder_model=config.model_path,
            spm_model=str(config.spm_path),
            spm_vocab=config.spm_vocab_path,
        )
        self.spm = sentencepiece.SentencePieceProcessor()
        self.spm.Load(spm_model)
        self.encoder = laser_sentence_encoder.SentenceEncoder(
            config.model_path,
            max_tokens=config.max_tokens,
            spm_vocab=spm_vocab,
            sort_kind="quicksort",
            cpu=not config.gpu,
        )

        if self.want_lid_filtering:
            self.lid = fasttext.load_model(config.fasttext_model)

        self.stats: tp.Dict[str, int] = collections.defaultdict(int)

    def run(self, *args):
        self._prepare()

        x, y = self._read_sents()
        log.info(
            f"Loaded {len(x.indices)} bitext. {len(x.uniq_sents)} unique source sentences, {len(y.uniq_sents)} unique target sentences"
        )
        log.info(f"Stats: {self.stats}")

        x = x._replace(uniq_embs=self.embed_sentences(x.uniq_sents, normalize=True))
        y = y._replace(uniq_embs=self.embed_sentences(y.uniq_sents, normalize=True))

        retrieval = self.config.laser_margin
        if retrieval != "bwd":
            log.info("perform k-nn source against target")
            x2y_sim, _ = self.knn(x.uniq_embs, y.uniq_embs)
            x2y_mean = x2y_sim.mean(axis=1)

        if retrieval != "fwd":
            log.info("perform k-nn target against source")
            y2x_sim, _ = self.knn(y.uniq_embs, x.uniq_embs)
            y2x_mean = y2x_sim.mean(axis=1)

        log.info(f"Writing results to {self.outfile}")
        with utils.open(self.outfile, "w") as o:
            # TODO: vectorize
            for i, j in zip(x.indices, y.indices):
                score = margin_ratio(
                    x.uniq_embs[i], y.uniq_embs[j], x2y_mean[i], y2x_mean[j]
                )
                print(score, x.uniq_sents[i], y.uniq_sents[j], sep="\t", file=o)

    def _read_sents(self) -> tp.Tuple[Corpus, Corpus]:
        """Load all sentences in memory."""
        x = Corpus([], [])
        x_sents_cache: tp.Dict[str, int] = {}
        y = Corpus([], [])
        y_sents_cache: tp.Dict[str, int] = {}
        sent_cols = self.config.columns.split(",")
        want_lid_filtering = self.want_lid_filtering
        seen_pair_hashes: tp.Set[int] = set()

        with utils.open(self.config.input_file, "r") as f:
            header_line = f.readline()
            try:
                header_cols = header_line.rstrip().split("\t")
                (src_col, tgt_col) = [header_cols.index(c) for c in sent_cols]
            except:
                raise ValueError(
                    f"Didn't find excepted columns ({sent_cols}) in header {header_cols}"
                )
            for line in f:
                self.stats["line"] += 1
                columns = line.rstrip().split("\t")
                src = columns[src_col]
                tgt = columns[tgt_col]

                norm_src = text_normalizer.normalize_for_dedup(src)
                norm_tgt = text_normalizer.normalize_for_dedup(tgt)
                pair_hash = xxhash.xxh3_64_intdigest("\t".join((norm_src, norm_tgt)))
                if pair_hash in seen_pair_hashes:
                    self.stats["dedup_line"] += 1
                    continue
                seen_pair_hashes.add(pair_hash)

                if want_lid_filtering and not self.accept_bitext(src, tgt):
                    continue

                self.stats["accepted"] += 1
                # TODO: use the normalized src/tgt for the translation caching
                x.append(src, x_sents_cache)
                y.append(tgt, y_sents_cache)

        return x, y

    def accept_bitext(self, src: str, tgt: str) -> bool:
        """We keeep a bitext only if the identified language on one side is the
        expected language on the other side.
        LID can be a bit noisy at the sentence level, so we allow some miss LID
        (The documents were already run through LID before).
        This should remove the most proeminent case of English on both sides.
        """
        (label_src,), _ = self.lid.predict(src, k=1)
        if label_src[FASTTEXT_LABEL_PREFIX:] == self.tgt_lang:
            self.stats["bad_lid_src"] += 1
            return False
        (label_tgt,), _ = self.lid.predict(tgt, k=1)
        if label_tgt[FASTTEXT_LABEL_PREFIX:] == self.src_lang:
            self.stats["bad_lid_tgt"] += 1
            return False
        return True

    def embed_sentences(
        self, sentences: tp.Iterable[str], normalize: bool = True
    ) -> np.ndarray:
        max_tokens = self.config.max_tokens
        # TODO: it's not very efficient to tokenize, then join, then later retokenize
        tokenized_sents = [
            " ".join(self.spm.EncodeAsPieces(sent)[:max_tokens]) for sent in sentences
        ]
        embeddings = self.encoder.encode_sentences(tokenized_sents)
        # TODO: reuse normalize parameter of sentence encoder after https://github.com/fairinternal/nllb/pull/79
        if normalize:
            embeddings /= np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
        return embeddings

    def knn(self, x: np.ndarray, y: np.ndarray) -> KNN:
        k = min(self.config.laser_margin_knn, y.shape[0])
        # If running on GPU, we might need to split the data in several batches
        mem_gb = 5 if self.config.gpu else 50
        dim = x.shape[1]
        batch_size = int(mem_gb * 1024 * 1024 * 1024) // (dim * 4)
        return knn_batched(x, y, k=k, batch_size=batch_size, gpu=self.config.gpu)


def knn_batched(
    x: Embedding, y: Embedding, k: int, batch_size: int = 0, gpu: bool = False
) -> KNN:
    """k-NN implementation using Faiss and batches.

    Batching avoid using more than mem_gb of memory for the index,
    especially relevant when running on GPU.
    """
    if batch_size == 0:
        batch_size = x.shape[0]
    if x.shape[0] <= batch_size:
        return faiss_knn(x, y, k=k, gpu=gpu)

    knn = KNN.zeros(n=x.shape[0], k=k)
    for xfrom in range(0, x.shape[0], batch_size):
        xto = min(xfrom + batch_size, x.shape[0])
        small_knns = []
        # Compares 'batch_size' embs from 'x' and 'batch_size' embs from 'y'.
        for yfrom in range(0, y.shape[0], batch_size):
            yto = min(yfrom + batch_size, y.shape[0])
            small_knn = faiss_knn(
                x[xfrom:xto], y[yfrom:yto], k=min(k, yto - yfrom), gpu=gpu
            )
            # returned indices are relative into y[yfrom:yto], make them absolute.
            small_knn.indices.__iadd__(yfrom)
            small_knns.append(small_knn)

        # Aggregate all results for the 'batch_size' embs from 'x' across all 'y'
        x_knn = KNN(
            np.concatenate([s.distances for s in small_knns], axis=1),
            np.concatenate([s.indices for s in small_knns], axis=1),
        )
        assert x_knn.distances.shape == x_knn.indices.shape
        assert x_knn.distances.shape == (xto - xfrom, k * len(small_knns))
        neighbors = np.argsort(-x_knn.distances, axis=1)[:, :k]
        knn.distances[xfrom:xto, :] = np.take_along_axis(
            x_knn.distances, neighbors, axis=1
        )
        knn.indices[xfrom:xto, :] = np.take_along_axis(x_knn.indices, neighbors, axis=1)
    return knn


def faiss_knn(x: Embedding, y: Embedding, k: int, gpu: bool = False) -> KNN:
    """k-NN using Faiss."""
    dim = x.shape[1]
    idx = faiss.IndexFlatIP(dim)
    if gpu:
        idx = faiss.index_cpu_to_all_gpus(idx)
    idx.add(y)
    return KNN(*idx.search(x, k))


def margin_ratio(
    x: np.ndarray, y: np.ndarray, x2y_mean: float, y2x_mean: float
) -> float:
    """Computes normalized cosimiliarity."""
    return x.dot(y) / ((x2y_mean + y2x_mean) / 2)


def main(file: Path, lang: str = "swh"):
    MODEL_DIR = Path("/private/home/kevinheffernan/nllb.200.models")

    cfg = LaserScorerConfig(
        input_file=file,
        model_path=MODEL_DIR / f"laser3-{lang}_Latn.v1.pt",
        spm_path=MODEL_DIR / f"laser2.spm",
        spm_vocab_path=str(MODEL_DIR / "laser2.cvocab"),
        # spm_vocab_path="/checkpoint/guw/laser2/spm100v1.xx.50k.cvocab",
        fasttext_model="/large_experiments/seamless/nllb/mmt/lidruns/lid_models/2022-02-18_ft_model.bin",
        src_lang=lang,
        tgt_lang="eng",
        gpu=True,
    )

    scorer = LaserScorerModule(cfg)
    scorer.run()


if __name__ == "__main__":
    import func_argparse

    func_argparse.single_main(main)
