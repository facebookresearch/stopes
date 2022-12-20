# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path


def chunk_files_exist(tsv_file: Path, num_chunks: int) -> bool:
    """
    Returns True only if all chunk files exist.
    """
    for chunk_id in range(num_chunks):
        chunk_file = tsv_file.parent / f"{tsv_file.stem}_{chunk_id}.tsv"
        if not chunk_file.is_file():
            return False
    return True


def split_tsv_files(data_dir: str, lang_dirs: str, num_chunks=16) -> None:
    """
    Splits tsv files into chunks where each tsv file is of the format:
    <root_dir>
    <filename>:<offset>:<length>\t<num_frames>
    <filename>:<offset>:<length>\t<num_frames>
    ...

    Args:
        data_dir (str): str path of the parent directory of the tsv file.
        lang_dir (str): Comma separated string of language directions.
            Ex: "hr-en,ro-en,es-en"
        num_chunks (int): Number of chunks you want to split the tsv file into.

    We write num_chunks output tsv files of the same format, with each filename
    <lang_dir>_<lang>_<chunk_id>.tsv
    """
    for lang_dir in lang_dirs.split(","):
        src, tgt = lang_dir.split("-")
        for lang in (src, tgt):
            tsv_file = Path(data_dir) / f"{lang_dir}_{lang}.tsv"
            if chunk_files_exist(tsv_file, num_chunks):
                continue
            num_lines = sum(1 for _ in open(tsv_file))
            with open(tsv_file, "r") as f:
                # First read the root_dir of the tsv file.
                root_dir = f.readline().strip()
                chunk_size = num_lines // num_chunks
                chunk_id = 0
                cur_chunk_size = 0
                for i, line in enumerate(f):
                    # Starting a new chunk: Open the file and write the root_dir
                    if cur_chunk_size == 0:
                        o = open(
                            Path(data_dir) / f"{lang_dir}_{lang}_{chunk_id}.tsv", "w"
                        )
                        print(root_dir, file=o)
                    # Write the current line.
                    print(line.strip(), file=o)
                    # Increment the chunk size.
                    cur_chunk_size += 1
                    # When we haven't hit the last chunk, and we've hit the last line of the chunk.
                    # We're keeping all chunk sizes constant, and dump the residue into the last chunk.
                    if chunk_id < num_chunks - 1 and cur_chunk_size == chunk_size - 1:
                        o.close()
                        cur_chunk_size = 0
                        chunk_id += 1
                    # Close the last chunk file only when we hit the last line of the parent tsv file.
                    elif chunk_id == num_chunks - 1 and i == num_lines - 1:
                        o.close()
