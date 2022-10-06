# mtoxicity-alti-holisticbias
MT toxicity at scale: deep detection and analysis. Subfolders:
- `alti/`: contains two folders: one with the code to extract source contributions and word alignments for a given source and target sentence pair; another with (1) the outputs of the translation models  and (2) the source contributions and word alignments for the MT outputs of holisticbias with the NLLB 3B dense model.
- `analysis/`: scripts for calculating/plotting toxicity results, given (1) toxicities precomputed with ETOX and (2) ALTI+ scores.
- `annotation/`: contains the false positive and the false negative analysis conducted for 8 outputs on the holisticbias toxicity detection.
- `ETOX/`: contains the tool for detecting toxicity
