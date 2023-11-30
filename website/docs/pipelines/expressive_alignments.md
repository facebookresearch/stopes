---
sidebar_position: 1
---

# Expressive parallel alignments

We extend the parallel speech alignment method described in Seamless Communication et al. (2023) to align pairs not only in terms of meaning, but also in terms of expressivity. Specifically, the pipeline is extended to allow for additional processing of the k-nearest-neighbors and introduces an option to add `auxiliary_embeddings`, which are expected to be a complementary source of prosodic input to the traditional semantic-based inputs e.g. [SONAR speech embeddings](https://github.com/facebookresearch/SONAR). The prosodic scores/similarity from the auxiliary embeddings are then "blended" together with the traditional margin-based scores using the formula:

$$ \text{blended-score}(x, y) = \alpha \cdot \text{margin} + (1 - \alpha) \cdot \text{auxiliary-score} $$

where $\alpha$ controls how much to weight the `margin` scores against the `auxiliary-score` (i.e. the prosodic similarity score). By default, the `auxiliary-score` will be calculated as the cosine between the source and candidate neighbors using the auxiliary embeddings. However, there is also an option to perform PCP inference using AutoPCP (Seamless Communication et al., 2023). In this instance, the `auxiliary-score` will be the AutoPCP outputs. 

---
**NOTE**

if no auxiliary-embeddings are provided, then the logic will not branch from the existing speech alignment pipeline.

---

Below is an example configuration of how to provide the `auxiliary_embeddings` for two language configurations: English (en) and Spanish (es). For `auxiliary_embeddings` which are already pre-computed, you can simply pass the option: `existing_aux_embedding_glob` using a similar structure to that shown below. For other options, please see the README for the existing alignment pipeline.

```
data_dir: /path/to/data
lang_configs:
  es:
    existing_embedding_glob: ${data_dir}/sonar-speech-embeddings/es_emb.[0-9][0-9][0-9].npy
    existing_aux_embedding_glob: ${data_dir}/aux_embeddings/es_emb.[0-9][0-9][0-9].npy
  en:
    existing_embedding_glob: ${data_dir}/sonar-speech-embeddings/en_emb.[0-9][0-9][0-9].npy
    existing_aux_embedding_glob: ${data_dir}/aux_embeddings/en_emb.[0-9][0-9][0-9].npy
```
