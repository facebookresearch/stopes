# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
This pipeline processes a parallel corpus and extracts the words
that are most frequently associated with added toxicity.
This is useful both for understanding the corpus
and for updating toxicity wordlists.

The pipeline consists of 4 steps, one module for each:
1. Reading and sampling some sentence pairs from a parallel corpus
2. Finding all pairs with detected toxicity on either or both sides
3. Applying a word aligner to find words translated as something toxic
4. Aggregating this statistics to get the most frequent word pairs with examples
"""

import asyncio
import logging
import os
import random
import typing as tp
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import hydra
import pandas as pd
from tqdm.auto import tqdm, trange

from stopes.core.stopes_module import Requirements, StopesModule
from stopes.core.utils import promote_config
from stopes.eval.toxicity.toxicity_list import ToxicityList
from stopes.eval.word_alignment.aligners.awesome_align_wrapper import (
    AwesomeAlignWrapper,
)
from stopes.eval.word_alignment.alignment_utils import alignments_to_text
from stopes.pipelines.monolingual.utils.word_tokenization import get_word_tokenizer

logger = logging.getLogger(__name__)

MATCH_SEP = "|"
NOT_FOUND = "*"


@dataclass
class BatchSampledCorpusConfig:
    input_dir: Path
    output_dir: Path
    directions: tp.List[
        str
    ]  # list of directions in NLLB format, such as eng_Latn-spa_Latn
    max_n_per_source: int = 10_000_000
    max_out_shard_size: int = 100_000
    fast_sample: bool = False


class BatchSampledCorpusModule(StopesModule):
    """
    Find all file pairs in the `input_dir` that contain the word `train` and the language names from one of the `directions`.
    Each direction is a pair like `lang1-lang2`, and the input filenames are supposed
    to look like `prefix-lang1-lang2.lang1` and `prefix-lang1-lang2.lang2`.
    From each file pair, sample at most `max_n_per_source` sentence pairs.
    Then recombine the sentence pairs into shards of size at most `max_out_shard_size` and save as .tsv files to `output_dir`.
    """

    def __init__(
        self,
        config: BatchSampledCorpusConfig,
    ):
        super().__init__(config, BatchSampledCorpusConfig)
        self.config: BatchSampledCorpusConfig

    def array(self) -> tp.Optional[tp.List[tp.Any]]:
        return self.config.directions

    def requirements(self):
        return Requirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=10,
            timeout_min=24 * 60,
        )

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        root = self.config.input_dir
        lang1, lang2 = iteration_value.split("-")  # type: ignore
        if lang1 > lang2:
            lang1, lang2 = lang2, lang1
        logger.info(f"Obtaining source data for {lang1}->{lang2}")
        all_train_files = root.iterdir()
        fn2sample = {}
        random.seed(1)
        for fn in all_train_files:
            # TODO these are filenames of the NLLB training corpus; we need to make the filtering more general
            if "train" not in str(fn) or "sampled" not in str(fn):
                continue
            if str(fn).endswith(f"{lang1}-{lang2}.{lang1}") or str(fn).endswith(
                f"{lang2}-{lang1}.{lang1}"
            ):
                s = fn.stat().st_size / 1024**3
                logger.info(f"{fn:50s}  {s:10.3f} GB")
                fn2 = Path(fn).with_suffix(f".{lang2}")
                assert fn2.exists()

                pairs: tp.List[tp.Tuple[str, str]] = []
                with fn.open("r") as f1, fn2.open("r") as f2:
                    for i, (line1, line2) in enumerate(zip(f1, f2)):
                        # if this is smaller than 1, this is the probability that the pair is selected.
                        ratio = self.config.max_n_per_source / (i + 1)
                        if random.random() < ratio:
                            pair = (line1.strip(), line2.strip())
                            if len(pairs) < self.config.max_n_per_source:
                                pairs.append(pair)
                            elif self.config.fast_sample:
                                break
                            else:
                                pairs[
                                    random.randint(0, self.config.max_n_per_source - 1)
                                ] = pair

                logger.info(f"Total input size: {len(pairs)}")
                fn2sample[fn] = pairs

        sampled_dataframes = []
        for filename, list_of_pairs in fn2sample.items():
            tmp = pd.DataFrame(list_of_pairs, columns=["text1", "text2"])
            tmp["source"] = filename
            sampled_dataframes.append(tmp)
        # TODO: fail gracefully if sampled_dataframes is an empty list
        big_sample = pd.concat(sampled_dataframes, ignore_index=True).sample(
            frac=1.0, random_state=1
        )

        out_files = []
        for shard_id, shard_start in enumerate(
            range(0, big_sample.shape[0], self.config.max_out_shard_size)
        ):
            shard = big_sample.iloc[
                shard_start : shard_start + self.config.max_out_shard_size
            ]
            out_filename = (
                self.config.output_dir
                / f"sampled_pairs-{lang1}-{lang2}-shard_{shard_id:04d}.tsv"
            )
            shard.to_csv(out_filename, index=None, sep="\t")
            out_files.append(out_filename)

        return out_files


@dataclass
class AddedToxicityDetectionConfig:
    input_files: tp.List[Path]
    toxlist_dir: Path
    direction: tp.Optional[
        str
    ] = None  # by default, it is inferred from the input filenames
    save_only_toxic: bool = True


class AddedToxicityDetectionModule(StopesModule):
    """
    For each of `input_files` (expected to be a .tsv file with columns "text1" and "text2"), run toxic word search.
    The languages are by default decoded from the filename or can be passed as the `direction` argument.
    Found bad words are saved as "matched1_string" and "matched2_string"; their counts, as "ntox1" and "ntox2" columns.
    The resulting .tsv file is written into the same directory with a new name.
    If `save_only_toxic` is true (by default), only the rows with some detected toxicity are saved.
    """

    def __init__(
        self,
        config: AddedToxicityDetectionConfig,
    ):
        super().__init__(config, AddedToxicityDetectionConfig)
        self.config: AddedToxicityDetectionConfig

    def array(self) -> tp.Optional[tp.List[tp.Any]]:
        return self.config.input_files

    def requirements(self):
        return Requirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=10,
            timeout_min=24 * 60,
        )

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        iteration_value = Path(iteration_value)  # type: ignore
        df = pd.read_csv(iteration_value, sep="\t")

        if self.config.direction is not None:
            lang1, lang2 = self.config.direction.split("-")
        else:
            parts = iteration_value.name.split("-")
            lang1, lang2 = parts[1], parts[2]

        etoxes = {
            lang: ToxicityList([str(self.config.toxlist_dir / f"{lang}_twl.txt")])
            for lang in [lang1, lang2]
        }  # TODO: apply language-specific detection, where appropriate

        df["matched1"] = [etoxes[lang1].get_toxic_words(s) for s in tqdm(df.text1)]
        df["matched2"] = [etoxes[lang2].get_toxic_words(s) for s in tqdm(df.text2)]

        df["matched1_string"] = df["matched1"].apply(lambda x: MATCH_SEP.join(x))
        df["matched2_string"] = df["matched2"].apply(lambda x: MATCH_SEP.join(x))

        df["ntox1"] = df["matched1"].apply(len)
        df["ntox2"] = df["matched2"].apply(len)

        if self.config.save_only_toxic:
            df_out = df[(df.ntox1 > 0) | (df.ntox2 > 0)]
        else:
            df_out = df
        df_out = df_out.drop(["matched1", "matched2"], axis=1)

        out_filename = iteration_value.parent / iteration_value.name.replace(
            "sampled_pairs-", "sampled_pairs_etox-"
        )
        df_out.to_csv(out_filename, index=None, sep="\t")

        return out_filename


@dataclass
class ToxicityAttributionConfig:
    input_files: tp.List[Path]
    direction: tp.Optional[
        str
    ] = None  # by default, it is inferred from the input filenames


class ToxicityAttributionModule(StopesModule):
    """
    For each of the `input_files` (.tsv files obtained from AddedToxicityDetectionModule), run word tokenization and alignment.
    For each detected toxic word, find its match on the other side of the translation (unmatched words are matched to "*").
    Save the new .tsv file with added columns: "words1", "words2", "word_alignment", "crossmatch1", "crossmatch2".
    """

    def __init__(
        self,
        config: ToxicityAttributionConfig,
    ):
        super().__init__(config, ToxicityAttributionConfig)
        self.config: ToxicityAttributionConfig

    def requirements(self):
        return Requirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=1,
            cpus_per_task=10,
            timeout_min=24 * 60,
        )

    def array(self) -> tp.Optional[tp.List[tp.Any]]:
        return self.config.input_files

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        iteration_value = Path(iteration_value)  # type: ignore
        df = pd.read_csv(iteration_value, sep="\t")
        df.matched1_string = df.matched1_string.fillna("")
        df.matched2_string = df.matched2_string.fillna("")

        word_aligner = AwesomeAlignWrapper()

        if self.config.direction is not None:
            lang1, lang2 = self.config.direction.split("-")
        else:
            parts = iteration_value.name.split("-")
            lang1, lang2 = parts[1], parts[2]

        logger.info("Tokenization...")
        tok1 = get_word_tokenizer(lang1)
        tok2 = get_word_tokenizer(lang2)
        tokens1 = [tok1.tokenize(text) for text in tqdm(df.text1)]
        tokens2 = [tok2.tokenize(text) for text in tqdm(df.text2)]

        logger.info("Word alignment...")
        strong_alignments = []
        weak_alignments = []
        for idx in trange(len(tokens1)):
            sa, wa = word_aligner.force_align_single_pair(tokens1[idx], tokens2[idx])
            strong_alignments.append(sa)
            weak_alignments.append(wa)

        df["words1"] = tokens1
        df["words2"] = tokens2
        df["word_alignment"] = [
            alignments_to_text(s, w) for s, w in zip(strong_alignments, weak_alignments)
        ]

        df["crossmatch1"] = [
            align_bad_words(
                row.matched1_string.split(MATCH_SEP),
                tokens1[i],
                tokens2[i],
                strong_alignments[i],
                reverse=False,
            )
            for i, row in enumerate(tqdm(df.itertuples(), total=df.shape[0]))
        ]
        df["crossmatch2"] = [
            align_bad_words(
                row.matched2_string.split(MATCH_SEP),
                tokens2[i],
                tokens1[i],
                strong_alignments[i],
                reverse=True,
            )
            for i, row in enumerate(tqdm(df.itertuples(), total=df.shape[0]))
        ]
        # join lists of words into strings
        for column in ["words1", "words2", "crossmatch1", "crossmatch2"]:
            df[column] = [MATCH_SEP.join(words) for words in df[column]]

        out_filename = iteration_value.parent / iteration_value.name.replace(
            "sampled_pairs_etox-", "sampled_pairs_etox_aligned-"
        )
        df.to_csv(out_filename, index=None, sep="\t")

        return out_filename


@dataclass
class ToxicityAggregationConfig:
    input_shard_groups: tp.List[tp.List[Path]]
    min_word_pair_count: int = 5
    examples_per_pair: int = 3
    random_proportion: float = 0


class ToxicityAggregationModule(StopesModule):
    """
    For each of the `input_shard_groups` (lists of .tsv files obtained from ToxicityAttributionModule),
    extract the pairs of the detected toxic word and the word aligned to it.
    Produce two .tsv files, with the word pairs most commonly associated
    with imbalanced toxicity, either positive (added) or negative (removed).
    Use only the word pairs appearing at least `min_word_pair_count` times.
    For each, sample `examples_per_pair` examples of sentence pairs.
    """

    def __init__(
        self,
        config: ToxicityAggregationConfig,
    ):
        super().__init__(config, ToxicityAggregationConfig)
        self.config: ToxicityAggregationConfig

    def array(self) -> tp.Optional[tp.List[tp.Any]]:
        return self.config.input_shard_groups

    def requirements(self):
        return Requirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=10,
            timeout_min=24 * 60,
        )

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        if len(iteration_value) == 0:  # type: ignore
            return
        dataframes = []
        for filename in iteration_value:  # type: ignore
            filename = Path(filename)  # type: ignore
            dataframes.append(pd.read_csv(filename, sep="\t"))
            parts = filename.name.split("-")
            lang1, lang2 = parts[1], parts[2]
        df = pd.concat(dataframes, ignore_index=True)
        top_negative = get_unbalanced_top(
            df,
            negative_balance=True,
            min_word_pair_count=self.config.min_word_pair_count,
            examples_per_pair=self.config.examples_per_pair,
            random_proportion=self.config.random_proportion,
        )
        top_positive = get_unbalanced_top(
            df,
            negative_balance=False,
            min_word_pair_count=self.config.min_word_pair_count,
            examples_per_pair=self.config.examples_per_pair,
            random_proportion=self.config.random_proportion,
        )

        prefix = f"top_unbalanced-{lang1}-{lang2}"
        root = filename.parent
        name_pos = root / f"{prefix}-positive.tsv"
        name_neg = root / f"{prefix}-negative.tsv"
        top_positive.to_csv(name_pos, index=None, sep="\t")
        top_negative.to_csv(name_neg, index=None, sep="\t")

        return [name_pos, name_neg]


def find_word(word: str, text_tokens: tp.List[str]) -> tp.Optional[tp.List[int]]:
    if word in text_tokens:
        return [text_tokens.index(word)]
    word_tokens = word.lower().split()
    n = len(word_tokens)
    text_tokens = [t.lower() for t in text_tokens]
    for i in range(len(text_tokens)):
        if text_tokens[i : i + n] == word_tokens:
            return list(range(i, i + n))
    return None


def align_bad_words(
    bad_words, words1, words2, strong_alignment, reverse=False, not_found=NOT_FOUND
) -> tp.List[str]:
    # turning the alignment into a dict
    id2align = defaultdict(list)
    for x, y in strong_alignment:
        if reverse:
            x, y = y, x
        id2align[x].append(y)

    # finding translations of l1 words
    trans = []
    for word in bad_words:
        word_indices = find_word(word, words1)
        if word_indices is None:
            matched_exp = not_found
        else:
            matched_exp = (
                " ".join(words2[y] for idx in word_indices for y in id2align[idx])
                or not_found
            )
        trans.append(matched_exp)
    return trans


def sample_or_all(data, n_samples: int):
    return random.sample(data, min(n_samples, len(data)))


def interleave(*sources) -> tp.List:
    """Pool a few iterable sources in a way that the order within each is preseved."""
    indices = [i for i, source in enumerate(sources) for _ in source]
    random.shuffle(indices)
    result = []
    pointers = [0] * len(sources)
    for i in indices:
        result.append(sources[i][pointers[i]])
        pointers[i] += 1
    return result


def unpack(joint_string) -> tp.List[str]:
    if isinstance(joint_string, str):
        return joint_string.split(MATCH_SEP)
    return []


def get_unbalanced_top(
    df: pd.DataFrame,
    min_word_pair_count: int = 5,
    examples_per_pair: int = 3,
    negative_balance: bool = True,
    random_proportion: float = 0,
) -> pd.DataFrame:
    """
    Given a dataframe with translation pairs and aligned bad words, extract the word pairs that are most commonly associated with added toxicity.
    If `negative_balance` is True, do this for removed toxicity; otherwise, for added toxicity.
    If `random_proportion` is positive, return not only the most frequent word pairs, but some randomly sampled ones.
    """
    word_pairs: tp.Counter[tp.Tuple[str, str]] = Counter()
    pair2id = defaultdict(list)
    for i, row in enumerate(tqdm(df.itertuples(), total=df.shape[0])):
        # case 1: removed toxicity
        if negative_balance and row.ntox1 > 0 and row.ntox2 == 0:
            for w1, w2 in zip(unpack(row.matched1_string), unpack(row.crossmatch1)):
                word_pairs.update([(w1, w2)])
                pair2id[(w1, w2)].append(i)
        # case 2: added toxicity
        if not negative_balance and row.ntox1 == 0 and row.ntox2 > 0:
            for w1, w2 in zip(unpack(row.matched2_string), unpack(row.crossmatch2)):
                word_pairs.update([(w1, w2)])
                pair2id[(w1, w2)].append(i)

    top_items = []
    tail_items = []
    for pair, pair_cnt in word_pairs.most_common():
        is_top = pair[1] != NOT_FOUND and pair_cnt >= min_word_pair_count
        ids = sample_or_all(pair2id[pair], examples_per_pair if is_top else 1)
        if negative_balance:
            w1, w2 = pair
        else:
            w2, w1 = pair

        for idx in ids:
            row = df.iloc[idx]
            item = {
                "w1": w1,
                "w2": w2,
                "n": pair_cnt,
                "e1": row.text1,
                "e2": row.text2,
                "ntox1": row.ntox1,
                "ntox2": row.ntox2,
            }
            if is_top:
                top_items.append(item)
            else:
                tail_items.append(item)
    if random_proportion == 0:
        merged = top_items
    elif random_proportion == 1:
        merged = top_items + tail_items
        random.shuffle(merged)
    else:
        n_tail_items = int(len(top_items) * random_proportion / (1 - random_proportion))
        merged = interleave(top_items, sample_or_all(tail_items, n_tail_items))
    df_diff = pd.DataFrame(merged)
    return df_diff


@dataclass
class AddedToxicityMiningPipelineConfig:
    launcher: tp.Any
    # input files are expected to be in paired files "{source}.{lang1}-{lang2}.{lang}", where {source} has "train" in it
    input_dir: Path
    output_dir: Path
    directions: tp.List[str]
    # wordlists are expected to be in files "{lang}_twl.txt"
    toxlist_dir: Path
    max_n_per_source: int = 5_000_000
    fast_sample: bool = False
    min_word_pair_count: int = 5
    examples_per_pair: int = 3
    random_proportion: float = 0


async def run_added_toxicity_mining(config_raw):
    config: AddedToxicityMiningPipelineConfig = promote_config(
        config_raw, AddedToxicityMiningPipelineConfig
    )
    launcher = hydra.utils.instantiate(config.launcher)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Data sampling step...")
    shards_raw: tp.List[tp.List[Path]] = await launcher.schedule(
        BatchSampledCorpusModule(
            BatchSampledCorpusConfig(
                input_dir=config.input_dir,
                output_dir=config.output_dir,
                directions=config.directions,
                max_n_per_source=config.max_n_per_source,
                fast_sample=config.fast_sample,
            )
        )
    )
    logger.info("Etox step...")
    shards_etoxed = await launcher.schedule(
        AddedToxicityDetectionModule(
            AddedToxicityDetectionConfig(
                input_files=[file for files in shards_raw for file in files],
                toxlist_dir=config.toxlist_dir,
            )
        )
    )
    logger.info("Alignment step...")
    shards_aligned = await launcher.schedule(
        ToxicityAttributionModule(
            ToxicityAttributionConfig(
                input_files=shards_etoxed,
            )
        )
    )
    # Grouping the shards with the same language pair together
    language_groups = defaultdict(list)
    for filename in shards_aligned:
        parts = filename.name.split("-")
        lang1, lang2 = parts[1], parts[2]
        language_groups[(lang1, lang2)].append(filename)
    counts = {
        lang_pair: len(filenames) for lang_pair, filenames in language_groups.items()
    }
    logger.info(f"Grouped language shards to aggregate: {counts}")

    shards_aggregated = await launcher.schedule(
        ToxicityAggregationModule(
            ToxicityAggregationConfig(
                input_shard_groups=list(language_groups.values()),
                min_word_pair_count=config.min_word_pair_count,
                examples_per_pair=config.examples_per_pair,
                random_proportion=config.random_proportion,
            )
        )
    )
    logger.info("Success! The results are saved to the following files:")
    for direction in shards_aggregated:
        for filename in direction:
            logger.info(str(filename))
    return shards_aggregated


@hydra.main(config_path="conf", config_name="mine_added_toxicity")
def main(config) -> None:
    asyncio.run(run_added_toxicity_mining(config))


if __name__ == "__main__":
    main()
