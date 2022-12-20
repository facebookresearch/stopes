# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import glob
import lzma
import shutil
import typing as tp
from pathlib import Path

import hydra
from omegaconf import DictConfig

from stopes.core.launcher import Launcher
from stopes.core.stopes_module import Requirements
from stopes.modules.preprocess.multiproc_line_processor import (
    MultiprocLineProcessorCallback,
    MultiprocLineProcessorConfig,
    MultiprocLineProcessorModule,
)
from stopes.utils.demojizer import Demojizer


class DemojizeLineProc(MultiprocLineProcessorCallback):
    """
    this is a cb for cleaning emojis line by line in parallel with MultiprocLineProcessor.
    Responsible for cleaning a chunk of the file
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
        self.demojizer = Demojizer()
        self.output_file = Path(self.output_dir) / (
            f"{self.outfile_prefix}.{input_file_idx:03d}.{offset_start}_{offset_end}.tsv"
        )
        self.text_starts_at_col = text_starts_at_col

    # context manager for the TextEncoder which deals with
    # opening and closing the output file the right way

    def __enter__(self):
        self._outf = self.output_file.open("w", encoding="utf-8")
        return self

    def __exit__(self, *exc):
        self._outf.close()

    def process_lines(self, lines_with_number: tp.Iterator[tp.Tuple[int, str]]) -> None:
        """
        process a batch of lines, filter them, dedup them locally
        and write them to the output file
        """
        for num, line in lines_with_number:
            cols = line.split(sep="\t", maxsplit=self.text_starts_at_col)
            text = self.demojizer(cols[-1], "")
            print(*cols[:-1], text, sep="\t", file=self._outf)

    def final_result(self) -> Path:
        print(f"finished processing to: {self.output_file}")
        return self.output_file

    def merge_results(self, splits: tp.List[Path]) -> Path:
        merge = Path(self.output_dir) / (
            f"{self.outfile_prefix}.{self.input_file_idx:03d}.demojize.xz"
        )
        # TODO replace lzma for utils.open
        with lzma.open(str(merge), "wb") as outfd:
            for f in splits:
                # TODO replace lzma for utils.open
                with f.open("rb") as infd:
                    shutil.copyfileobj(infd, outfd)
                    outfd.write("\n".encode("utf-8"))
        return merge


async def launch_processor(
    launcher: Launcher,
    raw_files: tp.List[Path],
    out_dir: Path,
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
                    "_target_": "stopes.pipelines.bitext.DemojizeLineProc.DemojizeLineProc",
                    "text_starts_at_col": text_starts_at_col,
                }
            ),
            custom_name=f"demojize",
            output_dir=str(out_dir),
            outfile_prefix="",
            shards=[str(f) for f in raw_files],
            requirements=requirements,
            tmp_dir=str(tmp_dir),
        )
    )

    return await launcher.schedule(file_processor)


@hydra.main(config_path="conf", config_name="demojize")
def main(config: DictConfig) -> None:
    launcher = hydra.utils.instantiate(config.launcher)
    files = config.files or list(glob.glob(config.file_glob))
    print(f"processing {len(files)} shards")
    Path(config.out_dir).mkdir(parents=True, exist_ok=True)
    res = asyncio.run(
        launch_processor(
            launcher=launcher,
            raw_files=files,
            out_dir=config.out_dir,
            text_starts_at_col=config.text_starts_at_col,
            requirements=Requirements(**config.requirements),
            tmp_dir=config.tmp_dir,
        )
    )
    print(f"done processing {res}")


if __name__ == "__main__":
    main()
