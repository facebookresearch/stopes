# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import dataclasses
import logging
import typing as tp
import uuid
import zipfile
from pathlib import Path

import hydra
import torch
import torchaudio
from tqdm import tqdm

from stopes.core import Requirements, StopesModule, utils
from stopes.core.utils import open as stopes_open
from stopes.modules.speech import audio_load_utils
from stopes.modules.speech import utils as speech_utils
from stopes.utils.config_utils import parse_hydra_list
from stopes.utils.data_utils import DictWithOverrides
from stopes.utils.sharding.text_shards import (
    TextShard,
    make_one_text_shard,
    make_text_file_shards,
    parse_header,
    resolve_output,
)


@dataclasses.dataclass
class AudioZipConfig:

    input_file: Path

    # column index (0,1) or column name ("src_audio", "tgt_audio",..)
    # Accepted values:
    # - column index: 0, 1, ...
    # - column headers: "src_audio", "tgt_audio",..
    # - List of column indicdes: '[0,1]', '[2,3]', ...
    # - List of column headers: '[es_audio,en_audio]', '[src_audio,tgt_audio]', ...
    column: tp.Any
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

    # Whether to perform internal sorting to speed up audio reading
    # (But will change the order of audios in the ip files)
    sorted: bool = True

    # Whether to provide zip for duplicate segments
    no_duplicate: bool = True

    # Custom reading logic in audio zip, applied to each shard at runtime
    # Default is None, meaning using the builtin reader of the shard
    reader_wrapper: tp.Optional[str] = None


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
        audio_zip.input_file=myfile.tsv \
        audio_zip.output_zip=myoutput.zip \
        audio_zip.column=2 \
        launcher.cluster=debug
    """

    config: AudioZipConfig

    def __init__(self, config: AudioZipConfig, **kwargs):
        super().__init__(config, AudioZipConfig)
        if not self.config.input_file.exists():
            raise ValueError(f"Input input_file not found: {self.config.input_file}")

        self.output_dir = Path(self.config.output_dir).resolve()
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.output_prefix = (
            self.config.output_prefix
            if self.config.output_prefix is not None
            and len(self.config.output_prefix) > 0
            else self.config.input_file.stem.replace(".tsv", "")
            .replace(".gz", "")
            .strip()
        )

        self.columns = parse_hydra_list(self.config.column)
        assert (
            self.columns
        ), f"Expect config.column to be a list or string, get {self.config.column}"
        self.multi_column_mode = len(self.columns) > 1
        self.header = not str(self.columns[0]).isdecimal()

        self._prepare_tmp_dir()
        self.output_zip = self.output_dir / f"{self.output_prefix}_audio.zip"
        self.manifest_tsv = self.output_dir / f"{self.output_prefix}_zipped.tsv.gz"
        self.logger = logging.getLogger("stopes.speech.audio_zip")
        self.kwargs = kwargs

    def _prepare_tmp_dir(self):
        """Prepare a temporary directory if needed"""
        if self.config.nshards > 1 and (
            self.config.input_file.suffix.endswith(".gz")
            or self.config.input_file.suffix.endswith(".zip")
        ):
            (self.output_dir / "workdir").mkdir(exist_ok=True)

        # In multi-column mode, we need a working dir to store intermediate file
        if self.multi_column_mode:
            (self.output_dir / "workdir").mkdir(exist_ok=True)

    def _combine_columns(self) -> Path:
        """
        combine multiple audio columns into ones. Return the intermediate file with
        one column containing all audio paths
        """
        assert (
            self.columns
        ), f"Expect config.column to be a list or string, get {self.config.column}"

        combined_file = (
            self.output_dir
            / "workdir"
            / (Path(self.config.input_file).name + ".columns_merged")
        )
        with (
            stopes_open(self.config.input_file) as reader,
            stopes_open(combined_file, "a+") as writer,
        ):
            if self.header and isinstance(self.header, tp.Iterable):
                cols = next(reader).rstrip("\n").split("\t")
                col_offsets = [cols.index(c) for c in self.columns]
            else:
                col_offsets = [int(c) for c in self.columns]
            for line in reader:
                columns = line.rstrip("\t").split("\t")
                for col in col_offsets:
                    writer.write(f"{ columns[col]}\n")
        return combined_file

    def requirements(self) -> Requirements:
        return Requirements(timeout_min=1440)

    def reader_wrapper(self) -> tp.Callable[..., tp.ContextManager]:
        """Provide a reader wrapper around a Shard, optionally with extra logic"""
        if self.config.reader_wrapper is None:
            return lambda x, *unused, **kwargs: x
        else:
            return hydra.utils.get_method(self.config.reader_wrapper)

    def array(self) -> tp.List[TextShard]:
        if self.multi_column_mode:
            input_file: Path = self._combine_columns()
            col = 0
        else:
            input_file = self.config.input_file
            col = self.columns[0]  # type: ignore
        return list(
            make_text_file_shards(
                input_file,
                nshards=self.config.nshards,
                algo="sort" if self.config.sorted else "chunk",
                cache_dir=self.output_dir / "workdir",
                header=self.header,
                sep="\t",
                col=col,
                no_duplicate=self.config.no_duplicate,
            )
        )

    def is_zip_complete(self, shard: TextShard, manifest: Path):
        """ """

    def run(
        self, iteration_value: tp.Any = None, iteration_index: int = 0
    ) -> tp.Tuple[Path, Path]:
        shard = iteration_value

        # Shard is None when the run() is called directly, e.g. when running the
        # module from command line: python -m stopes.modules.speech.audio_zip ...
        # In this case we create a dummy Shard object with index = None
        if shard is None:
            cols = parse_header(self.config.input_file, self.header, "\t")
            shard = TextShard(
                input_file=self.config.input_file, columns=cols, sep="\t", filter=None
            )

        assert isinstance(
            shard, TextShard
        ), "Each audio zip task expects input to be sharded, or convert to a Shard"

        output_zip = resolve_output(shard, Path(self.output_zip), suffix=".zip")
        output_manifest = resolve_output(shard, Path(self.manifest_tsv), suffix=".tsv")
        error_manifest = resolve_output(shard, Path(self.manifest_tsv), suffix=".err")
        assert output_zip and output_manifest

        # IF there are more than one audio columns, we are actually handling the
        # combined tsv file here which is a one-column manifest
        if self.multi_column_mode:
            column_offset = 0
        else:
            assert isinstance(
                self.config.column, (int, str)
            ), f"invalid column setting: {self.config.column} (Expect int or str)"
            column_offset = shard.resolve_column_index(self.config.column)
        sample_rate = self.config.sample_rate
        audio_format = self.config.audio_format

        # Before running, check if the manifest exists and skip if the content is full
        # (Useful in rerunning the job)
        skipped = False
        if validate(
            output_manifest,
            output_zip,
            self.config.audio_format,
            self.config.output_validation_token,
        ):
            self.logger.info(f"Skip shard #{iteration_index}")
            skipped = True

        if not skipped:
            with contextlib.ExitStack() as stack:
                writer = stack.enter_context(
                    AudioZipWriter(
                        output_zip=output_zip,
                        manifest_path=output_manifest,
                        audio_format=audio_format,
                        add_header=False,
                        zip_comment=f"Audio segments extracted from {shard.input_file} [{shard.index}]",
                    )
                )
                reader = stack.enter_context(  # type: ignore
                    self.reader_wrapper()(  # type: ignore
                        shard,
                        column_offset=column_offset,
                        logger=self.logger,
                        **self.kwargs,
                    )
                )
                err = stack.enter_context(stopes_open(error_manifest, "wt"))  # type: ignore
                lines = iter(reader)

                for line, audio in tqdm(
                    audio_load_utils.parallel_audio_read(
                        lines,
                        column_offset,
                        sampling_factor=int(sample_rate / 1000),
                        **self.kwargs,
                    ),
                    unit="segment",
                ):
                    try:
                        line, audio = audio_load_utils.load_audio(
                            column_offset=column_offset,
                            gpu=False,
                            fp16=False,
                            line=line,
                            sampling_factor=int(sample_rate / 1000),
                        )
                        columns = line.rstrip("\n").split("\t")
                        audio = torch.tensor(audio, dtype=torch.float)
                        metadata_column_values = []
                        if self.config.store_num_frames:
                            metadata_column_values.append(str(audio.size(-1)))
                        if self.config.store_input_line:
                            metadata_column_values.append(line.rstrip("\n"))
                        writer.append(
                            audio=audio,
                            sampling_rate=sample_rate,
                            filepath=f"{columns[column_offset]}.{audio_format}",
                            metadata_column_values=metadata_column_values,
                        )
                    except Exception:  # type: ignore
                        self.logger.warning(f"Error in line {line}")
                        err.write(line)

            if not validate(
                output_manifest,
                output_zip,
                self.config.audio_format,
                self.config.output_validation_token,
            ):
                self.logger.warning(f"Shard {iteration_index} is not complete")
        return output_zip, output_manifest

    def name(self):
        """
        implement this if you want to give a fancy name to your job
        """
        # TODO ideally use hydra override_dirname here
        return "_".join(
            [
                self.__class__.__name__,
                str(self.config.output_prefix),
                str(self.config.nshards),
            ]
        )

    def should_retry(
        self,
        ex: Exception,
        attempt: int,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> bool:
        return True


def validate(
    manifest: Path,
    audio_zip: Path,
    audio_format: str,
    output_validation_token: tp.Optional[bool] = False,
) -> bool:
    if not Path(audio_zip).exists():
        return False

    try:
        with zipfile.ZipFile(audio_zip) as z:
            num_audio_files = len(
                [i for i in z.infolist() if i.filename.endswith(f".{audio_format}")]
            )
    except IOError as exc:
        return False

    num_lines = 0
    for line in utils.open(manifest):
        audio = speech_utils.parse_audio(
            line.rstrip("\n").split("\t")[0], sampling_factor=16
        )
        assert isinstance(audio, speech_utils.AudioBytes)
        audio.load()
        num_lines += 1

    if num_lines != num_audio_files and num_lines > 0:
        return False
    else:
        if output_validation_token:
            # persist validation token
            open(f"{manifest}.validated-ok", "w").close()
        return True


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

        self.columns = parse_hydra_list(self.config.column)
        if len(intermediate_zips) <= 1:
            self.logger.warning(
                "This module is called within a pipeline with one shard."
                "Normally this means the mining file is not big enough."
                "Your pipeline might be simpler without the PostAudioZip module."
            )
        self.intermediate_zips = intermediate_zips
        self.zip_dict = DictWithOverrides()

    def array(self):
        # Make PostAudioZip a virtual array module to access function 'resolve_column_index'.
        # TODO: Put `resolve_column_index()` to a general utils module
        header = isinstance(self.columns[0], str) and not self.columns[0].isdecimal()  # type: ignore
        return make_one_text_shard(self.config.input_file, header, sep="\t")

    def run(self, iteration_value: tp.Any = None, iteration_index: int = 0):
        input_file = iteration_value
        assert isinstance(input_file, TextShard)
        column_offsets = [input_file.resolve_column_index(col) for col in self.columns]  # type: ignore

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
        current_combined_zip_filename = self.output_zip.parent / (
            self.output_zip.stem
            + f"_{current_combined_zip_file_no:05}"
            + self.output_zip.suffix
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
                        current_combined_zip_filename = self.output_zip.parent / (
                            self.output_zip.stem
                            + f"_{current_combined_zip_file_no:05}"
                            + self.output_zip.suffix
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

    def name(self):
        """
        implement this if you want to give a fancy name to your job
        """
        # TODO ideally use hydra override_dirname here
        return "_".join([self.__class__.__name__, str(self.config.output_prefix)])


class AudioZipWriter:
    def __init__(
        self,
        output_zip: Path,
        manifest_path: Path,
        audio_column_name="audio",
        audio_format: str = "ogg",
        metadata_column_names: tp.Optional[tp.List[str]] = None,
        add_header: bool = True,
        zip_comment: tp.Optional[str] = None,
    ):
        """
        output_zip:
            The path to save audio zip
        manifest_path:
            The path to save the manifest file
            each line in the manifest file has at least the following format:
                /path/to/your/zip:bytes_start:bytes_length
            where:
                `bytes_start` is the start position in bytes of the audio in this line,
                `bytes_length` is the length in bytes of the audio in this line.
        audio_column_name:
            The name for the audio path column (default "audio").
        audio_format:
            The format for the audio (default "ogg").
        metadata_column_names:
            Not required if add_header is False.
            By default the manifest has only one column name "audio" indicating the location of the audio.
            Specify `metadata_column_names` if you want to have more meta information.
            The values of the metadata (`metadata_column_values` in append()) should be appended in the same order as the `metadata_column_names`
        add_header:
            Whether to add the header for the manifest file (default True).
        """
        self.output_zip = output_zip
        self.manifest_path = manifest_path
        self.audio_column_name = audio_column_name
        self.audio_format = audio_format
        self.metadata_column_names = (
            metadata_column_names if metadata_column_names is not None else []
        )
        self.add_header = add_header
        self.zip_comment = zip_comment

    def __enter__(self):
        self.zip_file = zipfile.ZipFile(
            self.output_zip,
            mode="w",
            compression=zipfile.ZIP_STORED,
        )
        if self.zip_comment is not None:
            self.zip_file.comment = bytes(
                self.zip_comment,
                encoding="utf-8",
            )
        self.manifest_file = open(
            self.manifest_path,
            mode="w",
            encoding="utf-8",
        )
        if self.add_header:
            col_names = [
                self.audio_column_name,
            ] + self.metadata_column_names
            header_line = "\t".join(col_names)
            self.manifest_file.write(f"{header_line}\n")
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.zip_file.close()
        self.manifest_file.close()

    def _append_audio(
        self, audio: torch.Tensor, sampling_rate: int, filename: tp.Optional[str] = None
    ) -> speech_utils.AudioBytes:
        audio_name = (
            filename
            if filename is not None
            else f"{str(uuid.uuid4())}.{self.audio_format}"
        )
        with self.zip_file.open(audio_name, mode="w") as audio_file:  # type: ignore
            bytes_start = int(audio_file._fileobj.tell())  # type: ignore[attr-defined]
            torchaudio.save(
                audio_file,
                audio,
                sampling_rate,
                format=self.audio_format,
            )
            bytes_length = int(audio_file._fileobj.tell()) - bytes_start  # type: ignore[attr-defined]
        return speech_utils.AudioBytes(
            path=audio_name,
            byte_offset=bytes_start,
            length=bytes_length,
            sample_rate=sampling_rate,
        )

    def _get_manifest_line(
        self,
        audio_bytes: speech_utils.AudioBytes,
        metadata_column_values: tp.Optional[tp.List[str]] = None,
    ):
        if metadata_column_values is None:
            metadata_column_values = []
        zipaudio_filename = ":".join(
            [
                str(self.output_zip.absolute()),
                str(audio_bytes.byte_offset),
                str(audio_bytes.length),
            ]
        )
        return "\t".join([zipaudio_filename] + metadata_column_values)

    def _append_to_manifest(
        self,
        audio_bytes: speech_utils.AudioBytes,
        metadata_column_values: tp.Optional[tp.List[str]] = None,
    ):
        line = self._get_manifest_line(
            audio_bytes,
            metadata_column_values,
        )
        self.manifest_file.write(f"{line}\n")

    def append(
        self,
        audio: torch.Tensor,
        sampling_rate: int,
        filepath: tp.Optional[str] = None,
        metadata_column_values: tp.Optional[tp.List[str]] = None,
    ):
        """
        audio: The audio tensor that you want to append into the audiozip.
        sampling_rate: Sampling rate.
        filepath: The original audio file path if applicable, when it is None, a random name would be used.
        metadata_column_values: A list of metadata values you want to append for the manifest file,
            the size of `metadata_column_values` should be equal to the size of `self.metadata_column_names`
            when add_header is True
        """
        if self.add_header and metadata_column_values is not None:
            assert len(metadata_column_values) == len(self.metadata_column_names), (
                f"The number of metadata values should be equal to the number of metadata columns, "
                f"got number of metadata values: {len(metadata_column_values)}, "
                f"number of metadata columns: {len(self.metadata_column_names)}"
            )
        if len(audio.size()) == 1:
            # torchaudio.save expects to have a 2D tensor
            audio = audio.unsqueeze(0)
        if audio.is_cuda:
            audio = audio.cpu()
        audio_bytes = self._append_audio(audio, sampling_rate, filepath)
        self._append_to_manifest(audio_bytes, metadata_column_values)


if __name__ == "__main__":
    import func_argparse

    # Simple command line to run the audio zipping without post-processing
    logging.basicConfig(level=logging.INFO)
    # python -m stopes.modules.speech.audio_zip --input_file mining_results.tsv --column 1 --output_dir /myoutput
    cfg = func_argparse.single_main(AudioZipConfig)
    AudioZipModule(cfg).run()
