# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import typing as tp
from pathlib import Path

import numpy as np


class MissingHeaderError(Exception):
    pass


class Embedding:
    def __init__(self, file_path: tp.Union[str, Path]):
        """Class to provide coherent workflow to load embeddings
        file_path (str or Path): path to the source file

        If the file is not compatible with numpy.load, will look for
        a header file (.hdr). See stopes.utils.embedding_utils.create_header
        to learn more on how to create a header file for your needs.

        """
        self.legacy_mode = False
        if isinstance(file_path, Path):
            self.file_path = file_path
        else:
            self.file_path = Path(file_path)
        try:
            with open(self.file_path, mode="rb") as fp:
                version = np.lib.format.read_magic(fp)
                np.lib.format._check_version(version)
                shape, _, dtype = np.lib.format._read_array_header(fp, version)
        except ValueError:
            try:
                self.legacy_mode = True
                header_path = _hdr_filepath(self.file_path)
                with open(header_path, mode="rb") as fp:
                    version = np.lib.format.read_magic(fp)
                    np.lib.format._check_version(version)
                    shape, _, dtype = np.lib.format._read_array_header(fp, version)

            except FileNotFoundError:
                raise MissingHeaderError(
                    """Could not load embedding file.
                    Please refer to stopes.utils.embedding_utils.create_header
                    to create a companion header file for your embedding or update
                    your file to integrate header
                    """
                )
        self.length = shape[0]
        self.embedding_dimensions = shape[1]
        self.dtype = dtype

    def __len__(self) -> int:
        return self.length

    @contextlib.contextmanager
    def open_for_read(self, mode="mmap") -> tp.Iterator[np.ndarray]:
        emb = None
        IMPLEMENTATION_ERROR_TEXT = (
            "open_for_read was called with not"
            + f" implemented option: {mode}. "
            + "Currently accepted modes: mmap, memory, or an empty mode."
        )
        try:
            if self.legacy_mode:
                if mode == "mmap":
                    emb = np.memmap(
                        self.file_path,
                        dtype=self.dtype,
                        mode="r",
                        shape=(len(self), self.embedding_dimensions),
                    )
                elif mode == "memory":
                    emb = np.fromfile(self.file_path, dtype=self.dtype, count=-1)
                    emb.resize(len(self), self.embedding_dimensions)
                else:
                    raise NotImplementedError(IMPLEMENTATION_ERROR_TEXT)
            else:
                if mode == "memory":
                    mmap = None
                elif mode == "mmap":
                    mmap = "r"
                else:
                    raise NotImplementedError(IMPLEMENTATION_ERROR_TEXT)
                emb = np.load(self.file_path, mmap_mode=mmap)
            yield emb
        finally:
            # looks like there is no clean way to close an mmap in numpy
            # and that the best way is to remove the ref and hope for it
            # to be GC
            if emb is not None:
                del emb

    def save(
        self,
        output_file: tp.Union[str, Path],
        sample: tp.Union[tp.List[int], np.ndarray] = None,
        fp16: bool = False,
        mode: str = "mmap",
    ) -> None:
        """Save embeddings to out_file (in fp32 by default)
        Useful to convert a file to fp16 or  write only parts of a file.

        Usage example:
        >>> input_file = "test.npy"
        >>> output_file = "out.npy"
        >>> data = np.random.randn(100, 5).astype(np.float32)
        >>> sample = np.random.choice(len(data), 5, replace=False)
        >>> emb = Embedding(input_file)
        >>> emb.save(output file,sample)
        """
        with self.open_for_read(mode=mode) as data:
            if sample is not None:
                data = data[sample]
            dtype = np.float16 if fp16 else np.float32
            np.save(file=output_file, arr=data.astype(dtype))


class EmbeddingConcatenator:
    def __init__(
        self,
        output_file: tp.Union[str, Path],
        fp16: bool = False,
    ):
        """Helper class to merge embeddings,
        for example when creating shards

        output_file (str or Path): path to the output file
        fp16 (bool, optional): whether the embeddings
            should be saved as float16 (True) or float32 (False, default).

        Usage example:
        >>> with EmbeddingConcatenator(out_file, self.config.fp16) as combined_fp:
        >>>    combined_fp.append_embeddings(sample_shards)
        """
        self.output_file = output_file
        self.shape = (0, 0)
        self.fp16 = fp16

        # header info
        self.dtype = np.float16 if fp16 else np.float32

        self.header = {
            "descr": np.lib.format.dtype_to_descr(np.dtype(self.dtype)),
            "fortran_order": False,
            "shape": self.shape,
        }

    def __enter__(self):
        self.output_file_pointer = open(self.output_file, "wb")
        # add temporary header at the beginning of the file
        # the header will be replaced with correct info when exiting,
        # without the need to reload or rewrite the entire file
        np.lib.format._write_array_header(
            self.output_file_pointer, self.header, version=(1, 0)
        )
        return self

    def __exit__(self, *_exc):
        self.output_file_pointer.seek(0)
        # rewrite header with correct shape
        header = self.header
        header["fortran_order"] = False  # C-order is ensured by contiguous array
        header["shape"] = self.shape
        np.lib.format._write_array_header(
            self.output_file_pointer, header, version=(1, 0)
        )
        self.output_file_pointer.close()

    def _update_shape(self, shape: tp.Tuple[int, int]) -> None:
        if self.shape[1] != 0:
            # check that all embeddings have the same dimension
            assert (
                shape[1] == self.shape[1]
            ), f"Embeddings do not have the same dimension: found {self.shape[1]} and {shape[1]}"
        self.shape = (self.shape[0] + shape[0], shape[1])

    def append_embedding_from_array(self, embedding: np.ndarray) -> None:
        self._update_shape(embedding.shape)
        np.ascontiguousarray(embedding, dtype=self.dtype).tofile(
            self.output_file_pointer
        )

    def append_files(self, emb_files: tp.List[tp.Union[str, Path]]) -> None:
        for file in emb_files:
            emb = Embedding(file)
            with emb.open_for_read(mode="memory") as data:
                self.append_embedding_from_array(data)


def _hdr_filepath(fp: Path) -> Path:
    """Create the hdr filepath from an embedding path"""
    return fp.with_suffix(".hdr")


def create_header(
    fp: tp.Union[str, Path],
    shape: tp.Tuple,
    dtype: type = np.float32,
    fortran_order: bool = False,
    inplace=False,
) -> tp.Union[str, Path]:
    """Create a companion header file that can be used to load embeddings
    created with the previous versions of stopes.

    fp (str or Path): path to the embeddings file
    shape (tuple): shape of the embedding file
    dtype (default: np.float32): type of the embedding
    fortran_order (bool, default: False):
        The array data will be written out directly if it is either
        C-contiguous or Fortran-contiguous. Otherwise, it will be made
        contiguous before writing it out.)
    inplace (bool, default: False): should the header be written directly in the
        file (True) or as a companion file (False, default).
        The companion file will have the same path as the provided file,
        but the extension will be replaced with '.hdr'.
        N.B.: Currently supposes the file can be loaded in memory
    """
    if inplace:
        data = np.fromfile(fp, dtype=dtype, count=-1)
        data.resize(shape)
        np.save(fp, data)
        return fp
    else:
        fp = Path(fp)
        hdr_fp = _hdr_filepath(fp)
        d = {
            "descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
            "fortran_order": fortran_order,
            "shape": shape,
        }
        with open(hdr_fp, mode="wb") as f:
            np.lib.format._write_array_header(f, d, version=(1, 0))
        return hdr_fp
