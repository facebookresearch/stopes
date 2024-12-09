# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from botok import WordTokenizer as BodWordTokenizer
from botok import sentence_tokenizer as bod_sentence_tokenizer


class BodSentenizer:
    """
    BodSentenizer is a small wrapper around botok for seamless sentence splitting.
    What it does:
    1. Apply botok word tokenization to the text.
    2. Apply botok sentenization to the tokens (it requires the words to be already identified).
    3. Based on the tokens identified as a sentence, extract the raw text corresponding to that sentence.
    """

    def __init__(self):
        self.tokenizer = BodWordTokenizer()

    def __call__(self, text: str) -> List[str]:
        tokens = self.tokenizer.tokenize(text, split_affixes=False)
        sents_raw = bod_sentence_tokenizer(tokens)
        sents_new = []
        for sent in sents_raw:
            first_tok = sent["tokens"][0]
            last_tok = sent["tokens"][-1]
            sents_new.append(
                text[first_tok["start"] : last_tok["start"] + last_tok["len"]].strip()
            )

        return sents_new
