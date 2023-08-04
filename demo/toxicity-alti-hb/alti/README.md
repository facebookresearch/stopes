# mtoxicity-alti-holisticbias

Data can be download from:

wget --trust-server-names https://tinyurl.com/toxtranslationaltioutputs


`HB-dense3B-outputs/`: 164 folder from English to 164 languages. Each folder has the non-tokenized and the spm translation output (of HolisticBias) for the NLLB 3B dense model . Example of folder for LANGX in the 164 languages:

    eng_Latn-LANGX/holistic.eng_Latn-LANGX

    eng_Latn-LANGX/spm_holistic.eng_Latn-LANGX


`HB-distilled600M-outputs`:   164 folder from English to 164 languages. Each folder has the non-tokenized and the spm translation output (of HolisticBias) for the NLLB 600M distilled model. Example of 1 of these folders for LANGX in the 164 languages:

    eng_Latn-LANGX/holistic.eng_Latn-LANGX

    eng_Latn-LANGX/spm_holistic.eng_Latn-LANGX


`alti-outputs/`: 164 folder from English to 164 languages. Each folder has two files: the outputs of the source contributions and alignments for the MT outputs of HolisticBias with the NLLB 3B dense model. Example 1 of these folders for LANGX in the 164 languages:

    eng_Latn-LANGX/output.eng_Latn-LANGX

    eng_Latn-LANGX/align.eng_Latn-LANGX
