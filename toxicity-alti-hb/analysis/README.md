# Analysis code

Contains scripts for calculating/plotting toxicity results, given precomputed toxicities and ALTI+ scores. Prerequisites:
- Install the HolisticBias module ([setup instructions](https://github.com/facebookresearch/ResponsibleNLP/tree/main/holistic_bias))
- Define paths for loading in pre-existing files of source/target sentences, toxicity results, ALTI+ source contribution scores, etc. in `util.py` (see `'TODO'`s)

Scripts:
- `00_compile_toxicity_stats.py`: compute the course-grained analysis of toxicity as a function of language, axis, noun, template, etc.
- `00c_plot_toxicity_per_lang.py`: plot the breakdown of toxicity across HolisticBias axes as a function of language
- `01_sample_high_risk_translations.py`: sample translations likely to be toxic despite not being labeled as toxic, for the false negative analysis
- `02_count_toxicity_sources.py`: analyze the breakdown of detected toxic words into those aligned to descriptor words, template words, or the noun in the original HolisticBias sentence
- `02b_plot_alignment_type_breakdown.py`: plot the breakdown of which types of words in the original sentence toxic words are aligned to, as a function of language
- `03_measure_source_contributions.py`: perform all analyses relating to ALTI+ source contribution scores in the paper
- `03b_make_toxicity_heatmap.py`: plot the translation counts and rate of toxicity as a function of ALTI+ source contribution score and the robustness of translation
