import subprocess
from pathlib import Path

COPYRIGHT = "Copyright (c) Meta Platforms, Inc. and affiliates"
FB_COPYRIGHT = "Copyright (c) Facebook, Inc. and its affiliates"

PY_HEADER = """# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""


def check_file(file: Path, autofix: bool = False) -> bool:
    full_text = file.read_text()
    if COPYRIGHT in full_text:
        return True

    # Either returns immediatly or first tries to fix things.
    if not autofix:
        return False

    if FB_COPYRIGHT in full_text:
        file.write_text(full_text.replace(FB_COPYRIGHT, COPYRIGHT))
        return True

    if file.suffix == ".py":
        file.write_text(PY_HEADER + full_text)
        return True

    return False


def test_all_files_have_a_copyright_header(autofix: bool = False):
    root = Path(__file__).resolve().parents[3]
    assert (root / ".git").is_dir()
    ls_tree = subprocess.check_output(
        ["git", "ls-tree", "-r", "HEAD", "--name-only"],
        encoding="utf-8",
    )
    files = ls_tree.strip().splitlines()
    failed = []
    for f in files:
        file = root / f
        if any(part.startswith("fb_") for part in file.parts):
            continue
        if file.suffix in (
            ".png",
            ".ico",
            ".json",
            ".jsonl",
            ".yaml",
            ".md",
            ".tsv",
            ".svg",
            ".txt",
            ".toml",
            ".ipynb",
            ".csv",
        ):
            continue
        if file.name in (
            ".gitignore",
            ".prettierignore",
            ".prettierrc",
            ".nojekyll",
            "moses-config.lowercase",
        ):
            continue
        if file.is_symlink():
            continue
        try:
            license = check_file(file, autofix=autofix)
        except:
            license = False
        if not license:
            print(file)
            failed.append(file)

    assert not failed


if __name__ == "__main__":
    test_all_files_have_a_copyright_header(autofix=True)
