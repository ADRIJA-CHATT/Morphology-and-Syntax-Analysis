# Citations and Data Sources

## Primary data sources

### FLEURS

Conneau, Alexis, et al. (2022). *FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech*. arXiv:2205.12446.

Hugging Face dataset: `google/fleurs`.

The current dataset card describes FLEURS as a 102-language speech benchmark built from 2009 n-way parallel sentences, with training data of roughly 10 hours of supervision per language. The Hugging Face distribution is provided as Parquet with language-specific configurations and is licensed CC-BY-4.0. The pipeline uses individual language train shards rather than the special `all` configuration.

### Universal Dependencies

Nivre, Joakim, et al. (2020). *Universal Dependencies v2: An Evergrowing Multilingual Treebank Collection*. LREC 2020, pp. 4034–4043.

Hugging Face dataset: `universal-dependencies/universal_dependencies`.

The current official HF distribution packages the UD 2.18 release snapshot as Parquet configurations, one configuration per treebank, while preserving CoNLL-U annotations. Treebanks have their own source corpora and licenses; the selected treebank name is recorded in the output for reproducibility.

## Locality and information-theoretic methodology

Hahn, Michael, Richard Futrell, and others. Work on information locality and surprisal-based characterizations of information distribution in language.

Cover, Thomas M. & Joy A. Thomas. *Elements of Information Theory*. Wiley.

Kneser, Reinhard & Hermann Ney (1995). Improved backing-off for m-gram language modeling. In *ICASSP*.

Chen, Stanley F. & Joshua Goodman (1999). An empirical study of smoothing techniques for language modeling. Harvard Computer Science Technical Report TR-10-99.

## Morphological / word-order decomposition

Koplenig, Alexander, Peter Meyer, Sascha Wolfer, and Carolin Müller-Spitzer (2017). *The statistical trade-off between word order and word structure — Large-scale evidence for the principle of least effort*. PLOS ONE, 12(3), e0173614. The preprint appeared in 2016. The implementation uses the following operational quantities:

`D_order = H_order - H_original`

`D_structure = H_structure - H_original`

`M = D_structure / (D_order + D_structure)` when the non-negativity/identification conditions are satisfied.

The project additionally uses Universal Dependencies annotation fields to obtain directly annotated morphology and dependency predictors, rather than treating the perturbation index as a complete morphology measure.

## Source-selection note

This rewrite intentionally does **not** use Mozilla Common Voice as a Hugging Face source. The current Common Voice HF repository is empty and Mozilla states that, effective October 2025, Common Voice datasets are exclusively available through Mozilla Data Collective. Therefore Common Voice/MDC is removed from this project's production data path rather than pretending that an obsolete HF Common Voice mirror is current.

## Software / implementation

- Python 3
- pandas
- NumPy
- SciPy
- scikit-learn
- PyArrow
- Hugging Face Hub (`huggingface_hub`)
- requests
- PyYAML
- matplotlib
- python-dotenv

The primary locality estimator is a held-out character-level interpolated Kneser–Ney model. The production pipeline intentionally avoids a neural LM dependency so that 102-language processing is practical on a disk-limited laptop.
