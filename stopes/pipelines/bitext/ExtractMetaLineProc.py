# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import glob
import logging
import lzma
import shutil
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import hydra
from omegaconf import MISSING, DictConfig, OmegaConf

from stopes.core import stopes_module, utils
from stopes.core.launcher import Launcher
from stopes.core.stopes_module import Requirements
from stopes.modules.preprocess.multiproc_line_processor import (
    MultiprocLineProcessorCallback,
    MultiprocLineProcessorConfig,
    MultiprocLineProcessorModule,
)

logger = logging.getLogger("meta_extractor")


class ExtractMetaLineProc(MultiprocLineProcessorCallback):
    """
    this is a cb for extracting metadata and text line by line in parallel with
    MultiprocLineProcessor.
    Responsible for extracting information from a chunk of the file
    """

    def __init__(
        self,
        # set by LineProcessorModule
        outfile_prefix: str,
        input_file: str,
        input_file_idx: int,
        output_dir: str,
        offset_start: tp.Optional[int],
        offset_end: tp.Optional[int],
        merging: bool,
        # our params
        #  the index of the field where we find the text, we assume everything before is metadata
        # and take from that index to the end of the line, splitting by tab
        text_starts_at_col: int,
    ):
        super().__init__(
            outfile_prefix=outfile_prefix,
            input_file=input_file,
            input_file_idx=input_file_idx,
            output_dir=output_dir,
            offset_start=offset_start,
            offset_end=offset_end,
            merging=merging,
        )
        self.output_basename = str(
            Path(self.output_dir)
            / f"{self.outfile_prefix}{Path(input_file).stem}.{input_file_idx:03d}.{offset_start}_{offset_end}"
        )
        self.meta_output_file = Path(f"{self.output_basename}.meta.tsv")
        self.text_output_file = Path(f"{self.output_basename}.text.tsv")
        self.nl_output_file = Path(f"{self.output_basename}.xxx.nl")
        self.text_starts_at_col = text_starts_at_col
        self.input_file = input_file

    def __enter__(self):
        self._meta_outf = self.meta_output_file.open("w", encoding="utf-8")
        self._text_outf = self.text_output_file.open("w", encoding="utf-8")
        return self

    def __exit__(self, *exc):
        self._meta_outf.close()
        self._text_outf.close()
        with self.nl_output_file.open("w", encoding="utf-8") as nl_outf:
            print(self.nb_lines, file=nl_outf)

    def process_lines(self, lines_with_number: tp.Iterator[tp.Tuple[int, str]]) -> None:
        """
        process a batch of lines, extracting metadata and text,
        and write that information to the output file
        """
        self.nb_lines = 0
        for num, line in lines_with_number:
            fields = line.strip().split("\t", maxsplit=self.text_starts_at_col)
            if len(fields) < 2:
                logger.info(f"Found meaningless text on line: {num} with text: {line}")
                continue
            print(*fields[: self.text_starts_at_col], file=self._meta_outf, sep="\t")
            print(*fields[self.text_starts_at_col :], file=self._text_outf, sep="\t")
            self.nb_lines += 1
            if self.nb_lines % 100_000 == 0:
                print(self.nb_lines, self.output_basename, sep="\t")

    def final_result(self) -> tp.Tuple[Path]:
        """
        called once all batches have been processed, right before collecting them
        """
        logger.info(
            f"finished processing to: {self.meta_output_file},"
            f" {self.text_output_file} and {self.nl_output_file}"
        )
        return (self.meta_output_file, self.text_output_file, self.nl_output_file)

    def merge_results(self, splits: tp.List[Path]) -> tp.Tuple[str]:
        """
        taking all batches and stitching them together in dedicated output files
        """
        basepath = str(
            Path(self.output_dir) / f"{self.outfile_prefix}{Path(self.input_file).stem}"
        )
        merge_meta = f"{basepath}.meta.xz"
        merge_text = f"{basepath}.text.xz"
        merge_nl = f"{basepath}.xxx.nl"

        meta_outfd = lzma.open(merge_meta, "wb")
        text_outfd = lzma.open(merge_text, "wb")
        nl_outfd = open(merge_nl, "wt")
        total_nb_lines = 0

        for meta, text, nl in splits:
            with meta.open("rb") as infd:
                shutil.copyfileobj(infd, meta_outfd)
            with text.open("rb") as infd:
                shutil.copyfileobj(infd, text_outfd)
            with nl.open("rt") as infd:
                total_nb_lines += int(infd.read().strip())
        print(str(total_nb_lines), file=nl_outfd)

        meta_outfd.close()
        text_outfd.close()
        nl_outfd.close()

        return (merge_meta, merge_text, merge_nl)


async def launch_processor(
    launcher: Launcher,
    raw_files: tp.List[Path],
    output_dir: Path,
    text_starts_at_col: int,
    requirements: Requirements,
    tmp_dir: Path,
) -> tp.List[Path]:
    """
    configure the MultiprocLineProcessorModule for you
    """

    file_processor = MultiprocLineProcessorModule(
        config=MultiprocLineProcessorConfig(
            line_processor=DictConfig(
                {
                    # this will eventually initialize SplitNormalizeFilterLID above
                    "_target_": "stopes.pipelines.bitext.ExtractMetaLineProc.ExtractMetaLineProc",
                    "text_starts_at_col": text_starts_at_col,
                }
            ),
            custom_name=f"extract_meta",
            output_dir=str(output_dir),
            outfile_prefix="",
            shards=[str(f) for f in raw_files],
            requirements=requirements,
            tmp_dir=str(tmp_dir),
        )
    )

    return await launcher.schedule(file_processor)


@hydra.main(config_path="conf", config_name="extract_meta")
def main(config: DictConfig) -> None:
    launcher = hydra.utils.instantiate(config.launcher)

    logger.info(f"Starting metadata extraction to {config.output_dir}")
    files = list(glob.glob(config.input_files_glob))

    nb_files = len(files)
    if nb_files == 0:
        logger.warning("There were no files to process")
        return

    logger.info(f"Found {nb_files} shards to extract from: {files}")
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    res = asyncio.run(
        launch_processor(
            launcher=launcher,
            raw_files=files,
            output_dir=config.output_dir,
            text_starts_at_col=config.text_starts_at_col,
            requirements=Requirements(**config.requirements),
            tmp_dir=config.tmp_dir,
        )
    )
    logger.info(f"Done extracting metadata, text and computing nl {res}")


if __name__ == "__main__":
    main()
