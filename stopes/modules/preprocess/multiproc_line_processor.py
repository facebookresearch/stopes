# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import multiprocessing
import os
import random
import shlex
import string
import subprocess
import tempfile
import typing as tp
from abc import abstractmethod
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import hydra
from fairseq.file_chunker_utils import Chunker, find_offsets
from joblib import Parallel, delayed
from omegaconf import MISSING, DictConfig

from stopes.core import utils
from stopes.core.stopes_module import Requirements, StopesModule
from stopes.modules.preprocess.line_processor import (
    LineProcessorCallback,
    LineProcessorConfig,
)


class MultiprocLineProcessorCallback(LineProcessorCallback):
    """
    beware: line numbers will only be local to a chunk, not global to your file
    use self.offset_start in combination to know in what block that line number is.
    If the offset is None, you are in the merge mode
    """

    def __init__(
        self,
        outfile_prefix: str,
        input_file: str,
        input_file_idx: int,
        output_dir: str,
        offset_start: tp.Optional[int],
        offset_end: tp.Optional[int],
        merging: bool = False,
    ) -> None:
        super().__init__(
            outfile_prefix=outfile_prefix,
            input_file=input_file,
            input_file_idx=input_file_idx,
            output_dir=output_dir,
        )
        self.offset_start = offset_start
        self.offset_end = offset_end
        self.merging = merging

    @abstractmethod
    def merge_results(self, splits: tp.List[tp.Any]) -> tp.Any:
        """
        Multiple version of this CB will be used concurently to process splits of the original input. At the end
        we need to merge (reduce) the list of parallel results into a single results. The CB is responsible to do that.

        A separate instance will be created to do this.

        Takes a list of results from a single threaded LineProcessorCallback and merge them.
        """
        pass


def _process_file_chunk(
    config: LineProcessorConfig,
    input_file: Path,
    input_file_idx: int,
    output_dir: Path,
    offset_start: int,
    offset_end: int,
):
    processor: MultiprocLineProcessorCallback = hydra.utils.instantiate(
        config.line_processor,
        outfile_prefix=config.outfile_prefix,
        input_file=str(input_file),
        input_file_idx=input_file_idx,
        output_dir=str(output_dir),
        offset_start=offset_start,
        offset_end=offset_end,
        merging=False,
    )

    with Chunker(str(input_file), offset_start, offset_end) as line_iterator:
        with processor as proc:
            proc.process_lines(enumerate(line_iterator))

    return processor.final_result()


@dataclass
class MultiprocLineProcessorConfig(LineProcessorConfig):
    tmp_dir: str = MISSING


class MultiprocLineProcessorModule(StopesModule):
    def __init__(
        self, config: LineProcessorConfig = LineProcessorConfig(), processed_lines=0
    ):
        super().__init__(config)

    def array(self):
        return self.config.shards

    def requirements(self):
        reqs = self.config.requirements
        if not isinstance(reqs, Requirements):
            # Performe conversion if needed
            return Requirements(**reqs)
        return reqs

    def name(self):
        return getattr(
            self.config,
            "custom_name",
            "_".join(
                ["MultiLineProc", self.config.line_processor._target_, self.sha_key()]
            ),
        )

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        input_file = Path(iteration_value)
        assert input_file.exists()

        reqs = self.requirements()
        num_workers = reqs.cpus_per_task

        slurm_env = os.environ.get("SLURM_JOB_ID", None)
        tmp_dir = Path(self.config.tmp_dir)
        if slurm_env:
            tmp_dir = tmp_dir / slurm_env
        tmp_dir.mkdir(parents=True, exist_ok=True)

        decompressed_tmp = utils.expand_if_compressed(input_file, tmp_dir)
        if decompressed_tmp:
            offsets = find_offsets(str(decompressed_tmp), num_workers)
            read_from = decompressed_tmp
        else:
            offsets = find_offsets(str(input_file), num_workers)
            read_from = input_file

        # find_offsets returns a list of position [pos1, pos2, pos3, pos4] but we would want pairs:
        # [(pos1, pos2), (pos2, pos3), (pos3, pos4)] to process the chunks with start/end info
        # we zip the list with itself shifted by one to get all the pairs.

        file_chunks = list(zip(offsets, offsets[1:]))

        proc_cb = partial(
            _process_file_chunk,
            self.config,
            str(read_from),
            iteration_index,
            tmp_dir,
        )

        print(f"starting jobs")
        final_results = Parallel(n_jobs=num_workers)(
            [delayed(proc_cb)(start, end) for (start, end) in file_chunks]
        )
        # with multiprocessing.get_context('spawn').Pool(num_workers) as p:
        #     # submit work
        #     results = [p.apply_async(proc_cb, offsets) for offsets in file_chunks]

        #     # wait for work
        #     final_results = [r.get() for r in results]

        merging_processor: MultiprocLineProcessorCallback = hydra.utils.instantiate(
            self.config.line_processor,
            outfile_prefix=self.config.outfile_prefix,
            input_file=str(input_file),
            input_file_idx=iteration_index,
            output_dir=self.config.output_dir,
            offset_start=None,
            offset_end=None,
            merging=True,
        )

        print(f"merging")
        full_rez = merging_processor.merge_results(final_results)

        if decompressed_tmp:
            decompressed_tmp.unlink()

        return full_rez


############
# Testing
############


class TestIdentityMultiLP(MultiprocLineProcessorCallback):
    def __init__(
        self,
        outfile_prefix: str,
        input_file: str,
        input_file_idx: int,
        output_dir: str,
        offset_start: tp.Optional[int],
        offset_end: tp.Optional[int],
        merging: bool = False,
    ) -> None:
        super().__init__(
            outfile_prefix,
            input_file,
            input_file_idx,
            output_dir,
            offset_start,
            offset_end,
            merging,
        )

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        input = Path(input_file)
        if merging:
            self.output_file = out_dir / f"{input.stem}_{input_file_idx}.txt"
        else:
            self.output_file = (
                out_dir
                / f"{input.stem}_{input_file_idx}_{offset_start}_{offset_end}.txt"
            )

    def __enter__(self):
        self._outf = self.output_file.open("w", encoding="utf-8")
        return self

    def __exit__(self, *exc):
        self._outf.close()

    def process_lines(self, lines_with_number: tp.Iterator[tp.Tuple[int, str]]) -> None:
        """
        process a batch of lines and write them to an output_file the way you want.
        The input is an iterator of lines with their line number in the input file
        """
        for _, l in lines_with_number:
            strip_l = l.rstrip()
            print(strip_l, file=self._outf)

    def final_result(self) -> tp.Any:
        """
        return whatever this callback created, probably a file name
        """
        return self.output_file

    def merge_results(self, splits: tp.List[tp.Any]) -> tp.Any:
        subprocess.run(
            shlex.join(["cat"] + [str(f) for f in splits])
            + " > "
            + shlex.quote(str(self.output_file)),
            shell=True,
            check=True,
        )
        return self.output_file


POPULATION = string.ascii_letters + string.digits


def make_sentence() -> tp.List[str]:
    length = random.randint(10, 50)
    return random.choices(
        population=POPULATION, k=length, weights=range(1, len(POPULATION) + 1)
    )


def make_data(length=1000, out_file=None) -> tp.List[tp.List[str]]:
    data = (
        [make_sentence() for _ in range(0, length)]
        # add all the symbols at least once
        + [list(POPULATION)]
    )
    if out_file is not None:
        with open(out_file, "w", encoding="utf-8") as out:
            for s in data:
                print(" ".join(s), file=out)

    return data


def test_multiproc_line_processor():

    num_splits = 5
    num_lines = 1000

    with tempfile.TemporaryDirectory() as tmpdirname:

        input_file = Path(tmpdirname) / "test_multiproc_line_processor.txt"

        sents = make_data(
            length=num_lines,
            out_file=input_file,
        )

        file_processor = MultiprocLineProcessorModule(
            config=MultiprocLineProcessorConfig(
                line_processor=DictConfig(
                    {
                        # this will eventually initialize SplitNormalizeFilterLID above
                        "_target_": f"{TestIdentityMultiLP.__module__}.TestIdentityMultiLP",
                    }
                ),
                output_dir=tmpdirname,
                outfile_prefix="",
                shards=[input_file],
                requirements=Requirements(
                    cpus_per_task=num_splits,
                ),
                tmp_dir=tmpdirname,
            )
        )

        final_file = file_processor(iteration_value=input_file, iteration_index=0)

        with final_file.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                strip_l = line.rstrip()
                exp = " ".join(sents[idx])
                assert (
                    exp == strip_l
                ), f'line {idx} doesn\'t match\nfound: "{strip_l}"\nexpec: "{exp}"'
