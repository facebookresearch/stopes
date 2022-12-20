# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Example usage (run it on the NLLB-compatible branch of the fairseq-py library):
```
python compute_nllb_alti.py \
    input_filename=$(pwd)/test_input.tsv \
    metrics_filename=$(pwd)/test_output.tsv \
    alignment_filename=$(pwd)/test_output_alignments.jsonl \
    src_lang=fra_Latn \
    tgt_lang=eng_Latn \
    +preset=nllb_demo +demo_dir=$(pwd)/nllb

```
"""
import hydra

from stopes.eval.alti.alti_metrics.nllb_alti_detector import (
    ALTIMetricsConfig,
    compute_nllb_alti,
)


@hydra.main(config_path=".", config_name="conf.yaml")
def main(config: ALTIMetricsConfig) -> None:
    return compute_nllb_alti(config)


if __name__ == "__main__":
    main()
