# mtoxicity-alti-holisticbias
MT toxicity at scale: deep detection and analysis. Subfolders:
- `alti/`: contains (1) the outputs of the translation models  and (2) the source contributions and word alignments for the MT outputs of holisticbias with the NLLB 3B dense model. We used the github repository: https://github.com/mt-upc/transformer-contributions
- `analysis/`: scripts for calculating/plotting toxicity results, given (1) toxicities precomputed with ETOX and (2) ALTI+ scores.
- `annotation/`: contains the false positive and the false negative analysis conducted for 8 outputs on the holisticbias toxicity detection.
- `ETOX/`: contains the tool for detecting toxicity
