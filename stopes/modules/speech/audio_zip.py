# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import dataclasses
import logging
import typing as tp
import zipfile
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from stopes.core import Requirements, StopesModule, utils
from stopes.modules.speech import speech_units
from stopes.modules.speech import utils as speech_utils
from stopes.utils.data_utils import DictWithOverrides
from stopes.utils.shards import (
    Shard,
    make_one_shard,
    make_shards,
    parse_header,
    resolve_output,
)

log = logging.getLogger("stopes.speech.audio_zip")


@dataclasses.dataclass
class AudioZipConfig:
    tsv_file: Path

    # column index (0,1) of column name ("src_audio", "tgt_audio",..)
    column: tp.Union[int, str]
    output_dir: Path
    output_prefix: tp.Optional[str] = None
    sample_rate: int = 16_000
    audio_format: str = "ogg"
    store_num_frames: bool = False
    store_input_line: bool = False
    output_validation_token: bool = False
    # runtime requirements:
    # nshards = no. of shards to split the inputs
    nshards: int = 1

    # Whether to provide zip for duplicate segments
    no_duplicate: bool = True


class AudioZipModule(StopesModule):
    """
    Read a manifest TSV file, where each line corresponds to an audio. Resample and put
    the output audios into one or a number of zip files that is defined by `nshards`.

    Output: 'nshards' zip files and 'nshards' manifest files of the format:
        <zip file>:<start offset>:<offset len>

    TODO: Combine all manifest file into one ?

    Example command to run audiozip from a manifest file on
    content of the 3rd column = 2 (0 index)
    python -m stopes.modules +audio_zip=base \
        audio_zip.tsv_file=myfile.tsv \
        audio_zip.output_zip=myoutput.zip \
        audio_zip.column=2 \
        launcher.cluster=debug
    """

    config: AudioZipConfig

    def __init__(self, config: AudioZipConfig, **kwargs):
        super().__init__(config, AudioZipConfig)
        if not self.config.tsv_file.exists():
            raise ValueError(f"Input tsv_file not found: {self.config.tsv_file}")

        self.output_dir = Path(self.config.output_dir).resolve()
        self.output_dir.mkdir(exist_ok=True)
        self.output_prefix = (
            self.config.output_prefix
            if self.config.output_prefix is not None
            and len(self.config.output_prefix) > 0
            else self.config.tsv_file.stem.replace(".tsv", "")
            .replace(".gz", "")
            .strip()
        )
        if self.config.nshards > 1:
            (self.output_dir / "workdir").mkdir(exist_ok=True)
        self.output_zip = self.output_dir / f"{self.output_prefix}_audio.zip"
        self.manifest_tsv = self.output_dir / f"{self.output_prefix}_zipped.tsv.gz"
        self.kwargs = kwargs
        self.header = (
            isinstance(self.config.column, str) and not self.config.column.isdecimal()
        )

    def requirements(self) -> Requirements:
        return Requirements(timeout_min=240)

    def array(self) -> tp.List[Shard]:
        return list(
            make_shards(
                self.config.tsv_file,
                nshards=self.config.nshards,
                algo="sort",
                cache_dir=self.output_dir / "workdir",
                header=self.header,
                sep="\t",
                col=self.config.column,
                no_duplicate=self.config.no_duplicate,
            )
        )

    def run(
        self, iteration_value: tp.Any = None, iteration_index: int = 0
    ) -> tp.Tuple[Path, Path]:
        shard = iteration_value

        # Shard is None when the run() is called directly, e.g. when running the
        # module from command line: python -m stopes.modules.speech.audio_zip ...
        # In this case we create a dummy Shard object with index = None
        if shard is None:
            cols = parse_header(self.config.tsv_file, "\t") if self.header else None
            shard = Shard(self.config.tsv_file, cols=cols, sep="\t")
        output_zip = resolve_output(shard, Path(self.output_zip), suffix=".zip")
        output_manifest = resolve_output(shard, Path(self.manifest_tsv), suffix=".tsv")
        assert output_zip and output_manifest

        column_offset = shard.resolve_column_index(self.config.column)
        with contextlib.ExitStack() as stack:
            progress = stack.enter_context(shard)
            manifest_o = stack.enter_context(utils.open(output_manifest, "w"))
            zip_o = stack.enter_context(
                zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_STORED)
            )
            zip_o.comment = bytes(
                f"Audio segments extracted from {shard.input_file} [{shard.index}]",
                encoding="utf-8",
            )
            lines = iter(progress)
            sample_rate = self.config.sample_rate
            audio_format = self.config.audio_format

            for line, audio in tqdm(
                speech_units.parallel_audio_read(
                    lines,
                    column_offset,
                    sampling_factor=int(sample_rate / 1000),
                    **self.kwargs,
                ),
                unit="segment",
            ):
                columns = line.rstrip("\n").split("\t")
                audio = torch.tensor(audio, dtype=torch.float)
                if audio.ndim == 1:
                    audio = audio.unsqueeze(
                        0
                    )  # input needs to be 2D for torchaudio.save
                frame_length = audio.size(-1)
                # The file names inside the zip archive look like this (with the spaces in them):
                # /path/to/audio.mp3 1981501 2033072 16.ogg
                sample_name = f"{columns[column_offset]}.{audio_format}"

                with zip_o.open(sample_name, mode="w") as sample_o:
                    sample_start = int(sample_o._fileobj.tell())  # type: ignore[attr-defined]

                    torchaudio.save(sample_o, audio, sample_rate, format=audio_format)
                    sample_end = int(sample_o._fileobj.tell())  # type: ignore[attr-defined]

                # Note: this can also be recreated from the .zip using zip_o.getinfo(sample_name)
                # the lines look like this:
                # /abs/output_dir/mined.en-de.en.zip:121:14700
                outputs = [f"{output_zip}:{sample_start}:{sample_end - sample_start}"]
                if self.config.store_num_frames:
                    outputs.append(str(frame_length))
                if self.config.store_input_line:
                    outputs.append(line.rstrip("\n"))
                print(
                    "\t".join(outputs),
                    flush=True,
                    file=manifest_o,
                )
        validate(
            output_manifest,
            output_zip,
            self.config.audio_format,
            self.config.output_validation_token,
        )
        return output_zip, output_manifest


def validate(
    manifest: Path,
    audio_zip: Path,
    audio_format: str,
    output_validation_token: tp.Optional[bool] = False,
) -> None:
    with zipfile.ZipFile(audio_zip) as z:
        num_audio_files = len(
            [i for i in z.infolist() if i.filename.endswith(f".{audio_format}")]
        )

    num_lines = 0
    for line in utils.open(manifest):
        audio = speech_utils.parse_audio(
            line.rstrip("\n").split("\t")[0], sampling_factor=16
        )
        assert isinstance(audio, speech_utils.AudioBytes)
        audio.load()
        num_lines += 1

    assert (
        num_lines == num_audio_files
    ), f"Found {num_lines} lines in {manifest}, but {num_audio_files} audio files in {audio_zip}"
    if output_validation_token:
        # persist validation token
        open(f"{manifest}.validated-ok", "w").close()


class PostAudioZipModule(AudioZipModule):
    """
    A post-processor of AudioZipModule in case `nshards` > 1.

    This module reads the input TSV file, check intermediate zip files from AudioZip and returns
    a manifest file that converts all segment in the input file into zipped segment in the intermediate files.
    This module also performs internal re-organisation of zip files to save disk spaces.

    Output is a final manifest file in TSV format:
    <zipped segment 1> TAB <zipped segment 2> .... TAB <zipped segment n>

    where n is the number of columns in `config`.`column`.

    Additional args:
        intermedia_zips: the list of intermedia zip files to be combined (output of AudioZipModule)
    """

    def __init__(
        self,
        config: AudioZipConfig,
        intermediate_zips: tp.List[Path],
    ):
        super().__init__(config)

        # post audiozip works with columns as a list
        if type(self.config.column) == str:
            self.columns = self.config.column.strip().strip("[]").split(",")
        elif type(self.config.column) == int:
            self.columns = [self.config.column]  # type: ignore[list-item]
        else:
            raise ValueError("AudioZip config only accepts column of type int or str")

        if len(intermediate_zips) <= 1:
            log.warning(
                "This module is called within a pipeline with one shard."
                "Normally this means the mining file is not big enough."
                "Your pipeline might be simpler without the PostAudioZip module."
            )
        self.intermediate_zips = intermediate_zips
        self.zip_dict = DictWithOverrides()

    def array(self):
        # Make PostAudioZip a virtual array module to access function 'resolve_column_index'.
        # TODO: Put `resolve_column_index()` to a general utils module
        header = isinstance(self.columns[0], str) and not self.columns[0].isdecimal()
        return make_one_shard(self.config.tsv_file, header, sep="\t")

    def run(self, iteration_value: tp.Any = None, iteration_index: int = 0):
        input_file = iteration_value
        assert isinstance(input_file, Shard)
        column_offsets = [input_file.resolve_column_index(col) for col in self.columns]

        # combine all zips into a zip files saved in self.output_zip_{0000x})
        compacted_zip_files = self.combine_zips()
        self.load_zip_info(compacted_zip_files)

        with input_file as progress, utils.open(self.manifest_tsv, "wt") as f_out:
            lines = iter(progress)
            for line in tqdm(lines, unit="segment"):
                # replace each segment in the manifest by its zipped segment
                columns = line.rstrip("\n").split("\t")
                out_cols = []
                for col_n, col in enumerate(columns):
                    if col_n in column_offsets:
                        res = self.zip_dict.get(col)
                        if res is None:
                            raise RuntimeError(f"Unable to find file info for {col}")
                    else:
                        res = col
                    out_cols.append(res)
                print(*out_cols, sep="\t", file=f_out)

    def combine_zips(self):
        output_zip_finals = []
        current_combined_zip_file_no = 0
        current_combined_zip_filename = self.output_zip.with_stem(  # type: ignore[attr-defined]
            self.output_zip.stem + f"_{current_combined_zip_file_no:05}"
        )
        current_combined_zip_file = zipfile.ZipFile(current_combined_zip_filename, "w")
        byte_offset = 0
        for zip_filepath in tqdm(self.intermediate_zips):
            with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
                for file in zip_ref.filelist:
                    extracted_file = zip_ref.read(file.filename)
                    file_totalsize = len(file.FileHeader()) + file.file_size
                    byte_offset += file_totalsize
                    if byte_offset > 2**31:
                        current_combined_zip_file.close()
                        output_zip_finals.append(current_combined_zip_file.filename)
                        current_combined_zip_file_no += 1
                        current_combined_zip_filename = self.output_zip.with_stem(  # type: ignore[attr-defined]
                            self.output_zip.stem + f"_{current_combined_zip_file_no:05}"
                        )
                        current_combined_zip_file = zipfile.ZipFile(
                            current_combined_zip_filename, "w"
                        )
                        byte_offset = file_totalsize
                    current_combined_zip_file.writestr(file, extracted_file)
        current_combined_zip_file.close()
        output_zip_finals.append(current_combined_zip_file.filename)
        return output_zip_finals

    def load_zip_info(self, output_zip_finals):
        """
        this function reads the list of headers from all created audio_zips for each shard
        and created dictionary where segments in old format (space sep format) are mapped to segments in zipped format
        we can do this by only relying on the header of the zipped files
        """
        for output_zip in output_zip_finals:
            byte_offset = 0
            z = zipfile.ZipFile(output_zip)

            for f in z.filelist:
                # remove ogg from filename inside zip to endup with the orig segment name eg:
                # /path/to/audio.mp3 44474880 44693472 1054
                k = f.filename.replace("." + self.config.audio_format, "")

                # skip the file header usually ~133 bytes and output format is path:byte_offset:byte_size e.g.
                # /path/to/zip_file.zip:1377741:30157
                v = f"{output_zip}:{byte_offset +len(f.FileHeader())}:{str(f.file_size)}"

                self.zip_dict[k] = v
                byte_offset = byte_offset + len(f.FileHeader()) + f.file_size


if __name__ == "__main__":
    import func_argparse

    # Simple command line to run the audio zipping without post-processing
    logging.basicConfig(level=logging.INFO)
    # python -m stopes.modules.speech.audio_zip --tsv_file mining_results.tsv --column 1 --output_dir /myoutput
    cfg = func_argparse.single_main(AudioZipConfig)
    AudioZipModule(cfg).run()
