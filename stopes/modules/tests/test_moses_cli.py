# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path

import hydra

from stopes.core import utils
from stopes.core.stopes_module import StopesModule
from stopes.modules.preprocess.moses_cli_module import MosesPreprocessModule
from stopes.pipelines.tests import test_configs

# we use this as the base input
CLEAN_SENTENCE = "'the quick brown fox jumps over the lazy & dog'"


def test_moses_cli_module(tmp_path: Path):

    input_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    input_file = input_dir / "data.xz"
    input_dir.mkdir(exist_ok=True)

    with utils.open(input_file, "w") as f:
        print(CLEAN_SENTENCE, file=f)
        # this doesn't cover all of the moses changes, but mostly covers the call to all scripts
        print(CLEAN_SENTENCE.replace("'", "‘"), file=f)
        print(CLEAN_SENTENCE.replace(" ", "   "), file=f)
        print(CLEAN_SENTENCE.replace("&", "&amp;"), file=f)
        print(CLEAN_SENTENCE.upper(), file=f)

    mod: MosesPreprocessModule = test_configs.instantiate_conf(
        test_configs.CONF / "moses" / "standard_conf.yaml",
        "moses.lang=en",
        f"moses.shards=[{input_file}]",
        f"moses.output_dir={output_dir}",
    )

    outfile = mod.run(iteration_value=input_file, iteration_index=0)

    with utils.open(outfile) as out:
        for idx, line in enumerate(out):
            line = line.rstrip("\n")
            assert line == CLEAN_SENTENCE, f"{line}:{idx} did not match"
