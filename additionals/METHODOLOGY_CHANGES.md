# Methodology and Code Changes

## 1. Dataset source change

**Before:** Common Voice 26 was downloaded through Mozilla Data Collective (MDC), with additional MASSIVE/MLS/Speech-MASSIVE paths in the project.

**Now:** the production speech corpus is `google/fleurs` on Hugging Face. The new registry uses one language configuration per available FLEURS language, covering all 102 configurations currently documented by the dataset card. The `all` configuration is excluded.

**Why:** Common Voice's Hugging Face repository is currently empty and Mozilla states that, since October 2025, Common Voice datasets are exclusively distributed through Mozilla Data Collective. Using an old Common Voice HF URL would therefore be misleading.

## 2. New language list

The old fixed 20-language list has been removed. `src/language_registry.py` contains a new 102-language registry derived from current FLEURS configuration availability, and `additionals/LANGUAGE_COVERAGE.md` gives the exact config, language code, ISO-639-3 code and broad geographic group.

## 3. Speech/transcript population

FLEURS provides paired `transcription` and `num_samples` fields at 16 kHz. The pipeline therefore computes utterance duration directly as:

`duration_s = num_samples / 16000`.

This is preferable to inferring time from a phoneme-duration model when real audio duration is available.

The raw `audio` payload is not decoded. This keeps the implementation lighter and makes `num_samples` sufficient for temporal calibration.

## 4. Separate structural/morphological evidence

The previous main path could calculate morphology-like statistics on the same Common Voice utterances that supplied the speech-time and locality estimates. The rewritten project restores the earlier research-design separation:

- FLEURS = speech, transcript, duration, locality/time calibration.
- Universal Dependencies 2.18 = independently annotated structural/morphological predictors when a matching treebank exists.

This avoids making a single short speech corpus carry every theoretical variable.

UD fields used are `text`, `tokens`, `upos`, `feats`, `head`, and `deprel`.

The Koplenig-style entropy perturbation index is retained as an operational proxy rather than being presented as identical to a hand-annotated morphological richness scale.

## 5. Locality estimator

The primary estimator is a held-out character-level interpolated Kneser–Ney model. For each lag/order `t`, a model is trained on one set of utterances and evaluated at the same held-out target positions.

The information curve is defined from corrected cross-entropies:

`I_t = S_{t-1} - S_t`.

The `S_t` sequence receives a monotone non-increasing correction before differencing. Contexts do not cross utterance boundaries.

The previous neural LSTM implementation remains conceptually useful but has been removed from the production execution path. Running 102 languages does not require two competing production estimators, and removing PyTorch substantially reduces environment size and operational complexity.

## 6. Decay fitting

The exponential model is fitted only to the contiguous positive prefix of `I_t` beginning at lag 1:

`I_t = C exp(-lambda t)`.

Non-positive values or values created by the monotonic correction are not treated as evidence for an additional decay tail.

The conversion between character and time rates is explicit:

`lambda_per_s = lambda_per_char * speech_char_rate_cps`.

Consequently, the secondary time-domain regression excludes `speech_char_rate_cps` to prevent algebraic coupling of the predictor and outcome.

## 7. Morphology measures

Retained:

`D_order = H_order - H_original`

`D_structure = H_structure - H_original`

`M = D_structure / (D_order + D_structure)` when the identification conditions hold.

Added directly annotated UD predictors:

- proportion of tokens with non-empty FEATS;
- mean number of morphological features per token;
- dependency-direction entropy;
- mean dependency distance.

## 8. Speech-rate measures

The primary temporal calibration is measured audio duration divided by transcript character count. Word/syllable/character rates are also reported.

Syllable count is intentionally marked as an **approximate orthographic heuristic** because a single pronunciation backend is not equally reliable across all 102 languages and scripts. Syllable rate is therefore secondary, not the definition of time.

## 9. Disk constraint

The code enforces the requested storage pattern:

1. query HF metadata only;
2. download exactly one Parquet file;
3. validate its schema;
4. process it with PyArrow record batches;
5. delete it in a `finally` block;
6. move to the next file.

No Hugging Face dataset cache is used, and the temporary directory is asserted to contain no file between source files and after each language.

## 10. Fault tolerance

Each completed language is checkpointed in `results/checkpoints/`. The derived feature/curve/coverage rows are written incrementally. An interrupted full run can therefore be restarted without redownloading completed languages.

Failures do not leave partial Parquet files intentionally: `.part` and destination files are removed on download failure.

## 11. Reproducibility

The run metadata records the HF repository IDs, revisions, language order, source files, byte sizes, schema columns and storage policy. The FLEURS revision is `main`; UD is pinned to release revision `2.18`.
