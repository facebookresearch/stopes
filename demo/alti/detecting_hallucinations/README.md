# Token contributions for hallucination detection

This folder contains the code for reproducing the experiments from the paper 
[Detecting and Mitigating Hallucinations in Machine Translation: Model Internal Workings Alone Do Well, Sentence Similarity Even Better](https://arxiv.org/abs/2212.08597).

The structure is following:
- `annotated_data`: 
    - `guerreiro2022_corpus_w_annotations.csv`: the corpus from the [Guerreiro, 2022 repository](https://github.com/deep-spin/hallucinations-in-nmt) with German-English translations annotated by pathology type.
    - `annotate_hallucination_mitigation_v7_stacked.tsv`: a subset of this corpus translated by 3 improved systems and re-annottated. 
- `computed_data`: should contain the files created by our code. They can either be downloaded from [this link](https://dl.fbaipublicfiles.com/nllb/hallucination_detection_data.zip) or re-computed from scratch by the code below.
- `experiments`: the code for the experiments, organized in 5 Jupyter notebooks:
    - `01_Detection.ipynb`: computing various metrics of translation quality on the aforementioned corpus.
    - `02_Detection_analysis.ipynb`: evaluating the metrics computed above.
    - `03_Mitigation.ipynb`: translating German sentences to English by generating multiple hypotheses with various methods and reranking them with various scores.
    - `04_Mitigation_more_hypotheses.ipynb`: the same experiments as above, with fewer generation methods and a larger pool of hypotheses.
    - `05_Mitigation_analysis.ipynb`: evaluating the translations computed above.

The notebooks `02_Detection_analysis.ipynb` and `05_Mitigation_analysis.ipynb` reproduce most of the figures and tables mentioned in the paper.


# Setup

1. Prepare the environment by installing Fairseq, Stopes, and some extra libraries:
```
pip install fairseq==0.12.1
git clone https://github.com/facebookresearch/stopes.git
cd stopes
pip install -e '.[alti]'
cd demo/alti/detecting_hallucinations
pip install -r requirements.txt
```

2. Set up the translation model

    2.1. Download the translation [model](https://www.mediafire.com/file/mp5oim9hqgcy8fb/checkpoint_best.tar.xz/file) and [data](https://www.mediafire.com/file/jfl7y6yu7jqwwhv/wmt18_de-en.tar.xz/file) from https://github.com/deep-spin/hallucinations-in-nmt and put them in the `model` directory.

    2.2. Run the following commands to unpack the data:
    ```
    tar -xvf model/checkpoint_best.tar.xz
    tar -xvf model/wmt18_de-en.tar.xz
    ```

    2.3. Run the following command to download the tokenizers: 
    ```
    wget -P model https://github.com/deep-spin/hallucinations-in-nmt/raw/main/sentencepiece_models/sentencepiece.joint.bpe.model 
    wget -P model https://github.com/deep-spin/hallucinations-in-nmt/raw/main/sentencepiece_models/sentencepiece.joint.bpe.vocab 
    ```

3. Download LASER2 encoder (see https://github.com/facebookresearch/LASER/tree/main/nllb) by running:
```
mkdir laser
cd laser
curl -sSl 'https://raw.githubusercontent.com/facebookresearch/LASER/main/nllb/download_models.sh' | bash -s ''
cd ..
```

4. Optionally, download the computed translations and scores (instead of re-computing it by notebooks 1, 3, and 4):
```
wget https://dl.fbaipublicfiles.com/nllb/hallucination_detection_data.zip
unzip hallucination_detection_data.zip
```

Now you can run any notebook from the `experiments` folder.


# Citation
If you use refer to this code or results in your work, please cite:

```bibtex
@article{dale2022detecting,
    title={Detecting and Mitigating Hallucinations in Machine Translation: Model Internal Workings Alone Do Well, Sentence Similarity Even Better},
    author={Dale, David and Voita, Elena and Barrault, Lo{\"\i}c and Costa-juss{\`a}, Marta R},
    journal={arXiv preprint arXiv:2212.08597},
    url={https://arxiv.org/abs/2212.08597},
    year={2022}
}
```
