# Cross-Linguistic Information Locality Pipeline (Hugging Face)

This project estimates how rapidly character-level predictive information decays with temporal distance across the 102 language configurations currently exposed by the Hugging Face distribution of Google FLEURS. It combines:

1. **FLEURS speech + paired transcription** for the locality curve and speech-time calibration.
2. **Universal Dependencies (UD) 2.18**, from the official `universal-dependencies/universal_dependencies` Hugging Face distribution, for independently annotated structural/morphological predictors when a matching treebank exists.

The language list is intentionally **not inherited from the previous 20-language MDC/Common Voice run**. It is a new registry based on the current FLEURS language availability. The special FLEURS `all` configuration is never used.

## What changed from the previous pipeline

The previous code mixed Common Voice/MDC, MASSIVE, MLS and Speech-MASSIVE sources and, in the main run, used the same Common Voice utterance pool for transcript entropy, locality and speech timing. The rewritten pipeline makes the conceptual roles explicit:

- **Speech/time source:** FLEURS, with its paired `transcription` and `num_samples` fields; the current Hub card lists 102 language configurations.
- **Structural/morphological source:** a separately selected UD treebank, when one is available for the FLEURS language. This restores the earlier research design in which morphology/syntax is not inferred solely from the same speech sample used for time calibration.
- **Locality estimator:** held-out character-level interpolated Kneser–Ney is the primary estimator. The neural LSTM estimator and PyTorch dependency were removed from the production run because the KN estimator was already the primary scientific estimator and is substantially lighter for a 102-language run.
- **Time conversion:** `lambda_per_s = lambda_per_char * speech_char_rate_cps`; the time-domain regression therefore does **not** use `speech_char_rate_cps` as a predictor, avoiding algebraic leakage.
- **Morphology:** the Koplenig-style entropy perturbations are retained as an operational proxy, and UD annotation-derived measures are added: morphological feature density, mean FEATS per token, dependency-direction entropy and mean dependency distance.
- **Storage:** exactly one Parquet file is downloaded to `data/hf_one_file_tmp/` at a time, processed in Arrow batches, and deleted in a `finally` block. No persistent Hugging Face dataset cache is used.
- **Restartability:** after each completed language, derived CSVs and a per-language checkpoint are written. A future run skips completed languages unless `--overwrite` is supplied.

## Data formats handled explicitly

### FLEURS (`google/fleurs`)

The current Hugging Face distribution is Parquet and exposes separate language configurations. The pipeline discovers `train-*.parquet` shards for one FLEURS configuration at a time.

Required fields checked after download:

- `transcription`: normalized transcript text used for locality.
- `num_samples`: number of audio samples. FLEURS audio is 16 kHz, so duration is `num_samples / 16000` seconds.

Optional fields such as `speaker_id` are preserved when present. The raw `audio` column is not materialized by the pipeline; `num_samples` is sufficient for timing and avoids decoding the audio payload into memory. Schema validation is performed on every downloaded FLEURS file and recorded in `results/source_coverage.csv`.

### Universal Dependencies (`universal-dependencies/universal_dependencies`)

The current official HF distribution packages UD release 2.18 as Parquet configurations, one configuration per treebank. The pipeline discovers treebanks by FLEURS language code or ISO-639-3 code and deterministically chooses the best matching configuration.

Required fields checked after download:

- `text`
- `tokens`
- `upos`
- `feats`
- `head`
- `deprel`

These fields preserve the CoNLL-U linguistic annotation needed for morphology and dependency-based predictors. Every UD file is also deleted immediately after processing.

## Scientific definition

For a character stream `U`, define

`S_t = H(U_i | U_{i-t}, ..., U_{i-1})`

and estimate

`I_t = S_{t-1} - S_t`.

The primary estimator fits an order-`t` interpolated Kneser–Ney character model for each `t`, always evaluates the orders on the **same held-out target positions**, and then applies a monotone non-increasing correction to `S_t` before differencing. Context never crosses an utterance boundary.

The fitted positive locality tail is modeled as

`I_t = C exp(-lambda t)`

with `t` converted from characters to seconds using the measured FLEURS character rate. The decay fit uses only the contiguous positive-information prefix; artificially zeroed monotonic tails are not treated as additional observations.

## Morphology/syntax predictors

The pipeline retains the earlier Koplenig-style decomposition:

- `D_order = H_order - H_original`
- `D_structure = H_structure - H_original`
- `M = D_structure / (D_order + D_structure)` when both redundancies are non-negative and the denominator is positive.

The entropy-rate calculation is an operational proxy rather than an exact reproduction of every implementation detail in the literature. UD annotations then supply independent, directly annotated structural measures.

## Outputs

`results/language_features.csv` — one row per processed language.

`results/locality_curves.csv` — one row per language and character lag.

`results/source_coverage.csv` — exact FLEURS/UD files used, byte sizes and validated schema columns.

`results/checkpoints/*.done.json` — per-language restart checkpoints.

`results/plots/*.png` — the 42 generated figures from the unified plotting script, including the two family-coded all-language morphology/locality views:

- `41_morphology_index_vs_lambda_character_102_family.png`
- `42_morphology_index_vs_lambda_time_102_family.png`

`results/plots/derived_metrics_used_in_plots.csv` — the derived variables used by the plotting workflow.

`results/plots/PLOT_CATALOG.md` — automatically generated figure descriptions.

`results/plots/multilingual_locality_plots.pdf` — optional multipage PDF generated with `--pdf`.

`results/regression/` — cross-language RidgeCV regression and sensitivity analyses for completed full runs.

`results/run_metadata.json` — repositories, revisions, language order and the one-file storage policy.

## Disk-space policy

The pipeline never downloads all languages first. It lists repository metadata, chooses a processing order without downloading data, then processes one language shard/file at a time. Within a language, a single Parquet file is opened, streamed through Arrow batches and deleted before the next file is downloaded.

A pre-download disk check uses the largest FLEURS train shard for the language plus the configured free-space reserve. This is conservative: the pipeline does not assume that compressed Parquet size equals the eventual in-memory analysis size.

## Hugging Face authentication

Public FLEURS and UD data should work without a token in ordinary environments. If your network requires authenticated Hub access, create a local `.env` from `.env.example` and set `HF_TOKEN`. Never commit the token.

The previous project archive contained credential-looking values in its environment files; those values have deliberately been removed from this rewrite.

## Recommended WSL execution

```bash
cd ~/path/to/zippedcode
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r additionals/requirements.txt

# Optional, only if a token is needed:
cp additionals/.env.example additionals/.env
nano additionals/.env

python src/run_pipeline.py --mode pilot --max-languages 3
python src/check_hf_access.py
python src/run_pipeline.py --mode full --overwrite
```

For a disk-constrained full run, use the same command without `--overwrite` after an interruption. Completed languages are checkpointed and skipped.

Regenerate the unified plot set after results have been produced:

```bash
python src/plots.py --results-dir results --output-dir results/plots
```

To also assemble all generated PNGs into a multipage PDF:

```bash
python src/plots.py --results-dir results --output-dir results/plots --pdf
```

To run selected languages from the **new 102-language registry**, pass their exact names, for example:

```bash
python src/run_pipeline.py --mode full --languages Bengali Yoruba Finnish Zulu
```

Use only names present in `src/language_registry.py`; the registry is generated from the current 102 FLEURS configurations.

## Current morphology-index coverage

The current result table contains 102 language rows. All 102 have FLEURS locality estimates, while 58 currently have a materialized entropy-perturbation morphology index and 44 have `morphology_index = NA`. The 44 missing rows are associated with `ud_status = no_matching_treebank`. This is a property of the current pipeline wiring: the text-based perturbation calculation is invoked from the UD-processing branch. The mathematical definition of the index itself does not require UD. The two new family-coded morphology/locality plots keep all 102 rows visible by placing the 44 current missing-index languages in a hatched NA strip at their observed locality coordinate.

## Important interpretation limits

FLEURS is built from parallel sentences and controlled recording conditions, so its speech sample is much more domain-controlled than a large spontaneous-speech corpus. The locality estimate should therefore be interpreted as predictive-information behavior in the FLEURS speech domain, not as a universal estimate for every speaking context.

Syllable rate is estimated from Unicode orthography because a single phonological pronunciation pipeline is not equally reliable across all 102 languages. It is therefore a secondary feature, not the primary time conversion. The primary temporal conversion uses measured audio duration and character counts.

A language with no compatible UD treebank remains usable for FLEURS locality/time analysis, but its UD morphology fields are left missing rather than silently substituting a different language or an unannotated corpus.
