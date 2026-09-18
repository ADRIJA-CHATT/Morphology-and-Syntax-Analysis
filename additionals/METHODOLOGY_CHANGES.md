# Methodology and Code Changes

This document records the current project version. It supersedes the older Common Voice/MDC-oriented execution path.

## 1. Dataset source change

The production speech corpus is `google/fleurs` on Hugging Face. The current registry uses one FLEURS language configuration per language and covers 102 language rows. The special `all` configuration is excluded.

Common Voice/MDC is not part of the current production data path.

## 2. Language registry

The old fixed 20-language list has been replaced by the 102-language FLEURS-derived registry in `src/language_registry.py`. `additionals/LANGUAGE_COVERAGE.md` records the configuration, language code, ISO-639-3 code, and broad geographic group for each row.

## 3. Speech, transcript, and time source

FLEURS supplies paired transcription and audio metadata. The pipeline uses `transcription` for text-based locality and uses `num_samples` at 16 kHz to obtain utterance duration:

`duration_s = num_samples / 16000`.

The raw audio payload is not decoded for the production analysis.

## 4. Separate UD annotation layer

Universal Dependencies release 2.18 is used as a separate annotation source when a matching treebank exists. UD contributes annotation-dependent predictors such as morphological-feature density, mean FEATS per token, dependency-direction entropy, and mean dependency distance.

The entropy-perturbation morphology index is mathematically text-based and does not require UD. In the current computational wiring, however, that perturbation calculation is reached from the UD-processing branch. Consequently, the current 102-row result contains 58 materialized morphology-index values and 44 `NA` values, with the missing rows marked `ud_status = no_matching_treebank`. This is a processing-coverage condition rather than a claim that those languages lack morphology or that the index is undefined in principle.

## 5. Locality estimator

The primary estimator is a held-out character-level interpolated Kneser–Ney model. For each lag/order `t`, the model is trained on one utterance subset and evaluated on shared held-out target positions.

The information-locality curve is defined from corrected held-out cross-entropies:

`I_t = S_{t-1} - S_t`.

The empirical `S_t` sequence is made monotone non-increasing before differencing. Contexts do not cross utterance boundaries.

The previous neural LSTM execution path is not part of the current production workflow.

## 6. Decay fitting and time conversion

The positive locality prefix is fit with

`I_t = C exp(-lambda t)`.

The character-domain rate is `lambda_per_char`. Using the measured speech character rate `speech_char_rate_cps`, the corresponding time-domain parameter is

`lambda_per_s = lambda_per_char * speech_char_rate_cps`.

The time-domain regression excludes `speech_char_rate_cps` because it is already algebraically linked to the response.

## 7. Morphology perturbation measures

The project retains the following operational decomposition:

`D_order = H_order - H_original`

`D_structure = H_structure - H_original`

`M = D_structure / (D_order + D_structure)`

when the required non-negativity and positive-denominator conditions hold.

The implemented entropy calculation uses a Lempel–Ziv longest-previous-factor estimator. The index is treated as an operational perturbation statistic rather than a complete typological morphology scale.

## 8. Speech-rate measures

The primary temporal calibration is based on measured audio duration and transcript character count. Character, word, and syllable rate variables are also reported. Syllable counting is treated as an approximate orthographic heuristic and is not the definition of time.

## 9. Unified plotting workflow

The two plotting scripts have been merged into `src/plots.py`.

The unified script produces the complete plot set in `results/plots/`. It also writes the derived plotting data file `derived_metrics_used_in_plots.csv` and the automatically generated `PLOT_CATALOG.md`.

The two new family-coded morphology/locality views are:

- `41_morphology_index_vs_lambda_character_102_family.png`
- `42_morphology_index_vs_lambda_time_102_family.png`

They retain all 102 language rows. Languages with a current morphology-index value are shown at their measured coordinate; the 44 current `NA` rows are retained in a hatched strip at their observed locality coordinate.

## 10. Disk-space policy

The production pipeline lists metadata first, downloads one Parquet file at a time, validates it, processes it with PyArrow batches, and deletes it in a `finally` block before moving to the next file. No persistent Hugging Face dataset cache is used by the production workflow.

## 11. Restartability

Completed languages are recorded in `results/checkpoints/`. Derived rows are written incrementally so an interrupted full run can resume without repeating completed languages.

## 12. Reproducibility

Run metadata records repository identifiers, revisions, language ordering, source files, byte sizes, schema information, and the storage policy. The current production run uses FLEURS revision `main` and UD release `2.18`.
