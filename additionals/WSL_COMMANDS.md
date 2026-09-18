# WSL command sequence

From WSL, enter the project directory:

```bash
cd ~/path/to/information_locality_time_bundle
```

Create and activate the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r additionals/requirements.txt
```

Run the local synthetic smoke test:

```bash
PYTHONPATH=src python src/smoke_test.py
```

Check the current Hugging Face metadata before downloading dataset files:

```bash
PYTHONPATH=src python src/check_hf_access.py
```

Run the small pilot (first three languages in the 102-language registry):

```bash
python src/run_pipeline.py --mode pilot
```

Run a pilot for one named language:

```bash
python src/run_pipeline.py --mode pilot --languages Bengali
```

Run the full 102-language workflow:

```bash
python src/run_pipeline.py --mode full --overwrite
```

Resume an interrupted full workflow without deleting existing results:

```bash
python src/run_pipeline.py --mode full
```

Run a selected set from the current registry:

```bash
python src/run_pipeline.py --mode full --languages Bengali Yoruba Finnish Zulu
```

Regenerate the complete unified plot set after results have been produced:

```bash
python src/plots.py --results-dir results --output-dir results/plots
```

Also assemble the plots into a multipage PDF:

```bash
python src/plots.py --results-dir results --output-dir results/plots --pdf
```

The unified plot script writes the 42 PNG figures, `derived_metrics_used_in_plots.csv`, and `PLOT_CATALOG.md` under `results/plots/`.

Check that the temporary storage directory is empty after a run:

```bash
find data/hf_one_file_tmp -maxdepth 1 -type f -printf '%f\n'
```

Inspect progress:

```bash
tail -n 5 results/language_features.csv
tail -n 5 results/source_coverage.csv
ls results/checkpoints | wc -l
```

If authenticated Hugging Face access is required:

```bash
cp additionals/.env.example additionals/.env
nano additionals/.env
```

Put the token only in the local `.env` file as `HF_TOKEN=...`. Do not commit the file.

## Important storage behavior

Do **not** run `huggingface-cli download` or `datasets.load_dataset()` separately before the pipeline. Those workflows can create their own caches. The production pipeline deliberately uses Hub metadata plus direct streaming download into `data/hf_one_file_tmp/` and deletes each Parquet file after processing.
