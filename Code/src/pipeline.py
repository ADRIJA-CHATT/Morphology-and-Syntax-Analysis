from __future__ import annotations

from pathlib import Path
import re
import time
import shutil
import traceback
import csv
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import load_settings
from corpora import load_fleurs_samples, discover_ud_treebank, load_ud_sentences
from decay import fit_decay, bootstrap_lambda, memory_cost
from hf_sources import HuggingFaceSources, FLEURS_REPO, UD_REPO, FLEURS_REVISION, UD_REVISION
from language_registry import FLEURS_LANGUAGES
from locality import estimate_information_locality, choose_max_lag, InsufficientDataError
from morphology import ud_annotated_features
from regression import fit_models
from speech_features import aggregate_speech_features
from utils import free_bytes, stable_seed, write_json

ANALYSIS_VERSION = "2026-09-16-hf-fleurs-ud-v1"


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _checkpoint_path(results_root: Path, language: str) -> Path:
    path = results_root / "checkpoints" / f"{_safe_name(language)}.done.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _completed(results_root: Path, language: str) -> bool:
    return _checkpoint_path(results_root, language).exists()


def _quarantine_file(path: Path, reason: str) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.stem}.invalid_{stamp}{path.suffix}")
    path.rename(backup)
    print(f"[results] quarantined {path.name}: {reason} -> {backup.name}")
    return backup


def _csv_is_well_formed(path: Path) -> tuple[bool, str]:
    if not path.exists() or path.stat().st_size == 0:
        return True, ""
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if not header:
                return False, "missing header"
            expected = len(header)
            for line_no, row in enumerate(reader, start=2):
                if len(row) != expected:
                    return False, f"line {line_no} has {len(row)} fields; header has {expected}"
    except Exception as exc:
        return False, f"CSV parse error: {type(exc).__name__}: {exc}"
    return True, ""


def _append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    if path.exists() and path.stat().st_size:
        ok, reason = _csv_is_well_formed(path)
        if not ok:
            _quarantine_file(path, reason)
            frame.to_csv(path, index=False)
            return
        try:
            existing = pd.read_csv(path)
        except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
            _quarantine_file(path, f"pandas could not read existing CSV: {exc}")
            frame.to_csv(path, index=False)
            return
        # Rebuild rather than append raw fields. This guarantees a stable
        # column union when optional UD/speech fields differ between languages.
        combined = pd.concat([existing, frame], ignore_index=True, sort=False)
        combined.to_csv(path, index=False)
    else:
        frame.to_csv(path, index=False)


def _prepare_results_store(out_root: Path) -> None:
    """Quarantine malformed legacy CSVs before a run can read them.

    If any result table is corrupt, remove completion checkpoints as well: a
    checkpoint without its corresponding result row would silently create a
    missing-language analysis on resume.
    """
    result_names = ["language_features.csv", "locality_curves.csv", "source_coverage.csv"]
    invalid = []
    for name in result_names:
        path = out_root / name
        ok, reason = _csv_is_well_formed(path)
        if not ok:
            invalid.append((path, reason))
    for path, reason in invalid:
        _quarantine_file(path, reason)
    if invalid:
        ck_root = out_root / "checkpoints"
        if ck_root.exists():
            for ck in ck_root.glob("*.done.json"):
                ck.unlink()
        # Also remove plots that were generated from the invalid tables.
        plots = out_root / "plots"
        for name in ("locality_curves.png", "morphology_vs_locality.png"):
            (plots / name).unlink(missing_ok=True)


def _read_result_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    ok, reason = _csv_is_well_formed(path)
    if not ok:
        _quarantine_file(path, reason)
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        _quarantine_file(path, f"pandas could not read result CSV: {exc}")
        return pd.DataFrame()


def _write_checkpoint(results_root: Path, language: str, payload: dict) -> None:
    write_json(_checkpoint_path(results_root, language), {"language": language, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **payload})


def _assert_one_file_or_none(temp_root: Path) -> None:
    leftovers = [p for p in temp_root.iterdir() if p.is_file()] if temp_root.exists() else []
    if leftovers:
        raise RuntimeError(f"One-file invariant violated: {leftovers}")


def _plot_curves(curves: pd.DataFrame, out_path: Path) -> None:
    if curves.empty or not {"language", "lag_characters"}.issubset(curves.columns):
        return
    info_col = None
    for candidate in ("information_bits", "I_t_bits"):
        if candidate in curves.columns:
            info_col = candidate
            break
    if info_col is None:
        return
    clean = curves.copy()
    clean["lag_characters"] = pd.to_numeric(clean["lag_characters"], errors="coerce")
    clean[info_col] = pd.to_numeric(clean[info_col], errors="coerce")
    clean = clean.dropna(subset=["lag_characters", info_col, "language"])
    if clean.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for language, group in clean.groupby("language"):
        ax.plot(group["lag_characters"], group[info_col], alpha=0.55, linewidth=1.0, label=language)
    if len(curves["language"].unique()) <= 15:
        ax.legend(fontsize=8, ncol=2)
    ax.set_xlabel("Context lag (characters)")
    ax.set_ylabel("I_t (bits/target character)")
    ax.set_title("Character-level information locality by language")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_decay(features: pd.DataFrame, out_path: Path) -> None:
    if features.empty or "locality_lambda_per_s" not in features.columns:
        return
    x = pd.to_numeric(features["locality_lambda_per_s"], errors="coerce")
    y = pd.to_numeric(features["morphology_index"], errors="coerce") if "morphology_index" in features.columns else None
    if y is None:
        return
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x[keep], y[keep], alpha=0.75)
    for _, row in features.loc[keep].iterrows():
        ax.annotate(str(row["language"]), (row["locality_lambda_per_s"], row["morphology_index"]), fontsize=7, alpha=0.7)
    ax.set_xlabel("Information-locality decay rate (1/s)")
    ax.set_ylabel("Morphology index (Koplenig-style proxy)")
    ax.set_title("Morphological redundancy proxy and information-locality decay")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _run_one(language: str, lang_meta: dict, settings, sources: HuggingFaceSources, seed: int) -> tuple[dict, list[dict], dict]:
    sampling = settings.sampling()
    temp_root = settings.temp_root
    temp_root.mkdir(parents=True, exist_ok=True)
    # FLEURS: actual paired speech/transcript observations.
    utterances, fleurs_meta = load_fleurs_samples(
        sources=sources,
        language=language,
        config=lang_meta["fleurs_config"],
        work_dir=temp_root,
        max_utterances=int(sampling.get("max_utterances_per_language", 1200)),
        min_chars=int(settings.raw["sampling"].get("min_utterance_chars", 5)),
        max_chars=int(settings.raw["sampling"].get("max_utterance_chars", 600)),
        min_duration_s=float(settings.raw["speech"].get("min_duration_s", 0.4)),
        max_duration_s=float(settings.raw["speech"].get("max_duration_s", 30.0)),
        seed=seed,
    )
    _assert_one_file_or_none(temp_root)
    if len(utterances) < 20:
        raise RuntimeError(f"Only {len(utterances)} usable FLEURS utterances for {language}; too few for the held-out locality estimate.")

    texts = [u.text for u in utterances]
    durations = [u.duration_s for u in utterances]
    speech = aggregate_speech_features(texts, durations, language_code=lang_meta["iso639_3"])
    n_chars = int(sum(sum(not c.isspace() for c in t) for t in texts))
    max_lag, lag_reason = choose_max_lag(n_chars, settings.raw["locality"])
    locality = estimate_information_locality(
        language=language,
        utterance_texts=texts,
        seconds_per_character=speech["seconds_per_character"],
        max_lag=max_lag,
        params=settings.raw["locality"],
        seed=seed,
    )
    decay = fit_decay(
        locality.lags,
        locality.information,
        seconds_per_character=speech["seconds_per_character"],
        positive_floor=float(settings.raw["locality"].get("min_positive_it", 1e-5)),
        min_decay_points=int(settings.raw["locality"].get("min_decay_points", 5)),
    )
    boot = bootstrap_lambda(
        locality.lags,
        locality.information,
        seconds_per_character=speech["seconds_per_character"],
        reps=int(settings.raw["locality"].get("bootstrap_reps", 300)),
        seed=stable_seed(language, "bootstrap", base=seed),
        positive_floor=float(settings.raw["locality"].get("min_positive_it", 1e-5)),
        min_decay_points=int(settings.raw["locality"].get("min_decay_points", 5)),
    )

    # UD is deliberately a separate source. It provides the structural /
    # morphosyntactic measurements that should not be inferred from the same
    # short parallel speech prompts used for time calibration.
    ud_spec = discover_ud_treebank(
        sources=sources,
        fleurs_lang_code=lang_meta["fleurs_lang_code"],
        iso639_3=lang_meta["iso639_3"],
        fleurs_config=lang_meta["fleurs_config"],
    )
    ud_features: dict = {}
    ud_meta: dict = {"status": "no_matching_treebank", "treebank": ""}
    if ud_spec is not None:
        ud_sentences, ud_meta_load = load_ud_sentences(
            sources=sources,
            language=language,
            spec=ud_spec,
            work_dir=temp_root,
            max_sentences=int(sampling.get("max_ud_sentences", 5000)),
            max_chars=int(sampling.get("max_ud_characters", 1_000_000)),
            seed=seed,
        )
        _assert_one_file_or_none(temp_root)
        if len(ud_sentences) >= 20:
            ud_features = ud_annotated_features(ud_sentences, seed=seed)
            ud_meta = {"status": "loaded", "treebank": ud_spec.config, **ud_meta_load}
        else:
            ud_meta = {"status": "treebank_too_small", "treebank": ud_spec.config, **ud_meta_load}

    feature_row = {
        "language": language,
        "fleurs_config": lang_meta["fleurs_config"],
        "fleurs_lang_code": lang_meta["fleurs_lang_code"],
        "iso639_3": lang_meta["iso639_3"],
        "region_group": lang_meta["region_group"],
        "analysis_version": ANALYSIS_VERSION,
        "fleurs_revision": FLEURS_REVISION,
        "ud_revision": UD_REVISION,
        "fleurs_n_utterances": len(utterances),
        "fleurs_train_files": len(fleurs_meta["files"]),
        "fleurs_downloaded_bytes": fleurs_meta["downloaded_bytes"],
        "ud_status": ud_meta.get("status"),
        "ud_treebank": ud_meta.get("treebank", ""),
        "ud_downloaded_bytes": sum(f["size_bytes"] for f in ud_meta.get("files", [])),
        "max_lag_characters": max_lag,
        "max_lag_rule": lag_reason,
        "memory_cost_bits_char": memory_cost(locality.lags, locality.information),
        **speech,
        **decay,
        **boot,
        **ud_features,
        "locality_estimator": "held-out character-level interpolated Kneser-Ney",
        "locality_effective_train_chars": int(locality.effective_train_chars),
        "locality_effective_valid_chars": int(locality.effective_valid_chars),
        "locality_eval_positions": int(locality.eval_positions),
        "locality_requested_train_chars": int(settings.raw["locality"].get("kn_train_chars", settings.raw["locality"].get("train_chars", 100_000))),
        "locality_requested_valid_chars": int(settings.raw["locality"].get("kn_valid_chars", settings.raw["locality"].get("valid_chars", 20_000))),
        "locality_target_unit": "Unicode character excluding whitespace as eligible target; whitespace retained inside context stream",
        "locality_train_valid_fraction": float(settings.raw["locality"].get("train_fraction", 0.8)),
        "speech_time_source": "FLEURS num_samples at 16 kHz",
    }
    curve_rows = [
        {
            "language": language,
            "lag_characters": lag,
            "lag_seconds": lag * speech["seconds_per_character"],
            "information_bits": info,
            "surprisal_bits": locality.surprisal[i],
            "source": "google/fleurs",
        }
        for i, (lag, info) in enumerate(zip(locality.lags, locality.information))
    ]
    coverage = {
        "language": language,
        "fleurs_config": lang_meta["fleurs_config"],
        "fleurs_status": "ok",
        "fleurs_train_files": ";".join(f["path"] for f in fleurs_meta["files"]),
        "fleurs_train_bytes": fleurs_meta["downloaded_bytes"],
        "fleurs_schema_columns": ";".join(fleurs_meta["schema_checks"][0]["columns"]) if fleurs_meta["schema_checks"] else "",
        "ud_status": ud_meta.get("status"),
        "ud_treebank": ud_meta.get("treebank", ""),
        "ud_train_files": ";".join(f["path"] for f in ud_meta.get("files", [])),
        "ud_train_bytes": sum(f["size_bytes"] for f in ud_meta.get("files", [])),
        "ud_schema_columns": ";".join(ud_meta["schema_checks"][0]["columns"]) if ud_meta.get("schema_checks") else "",
        "one_file_at_a_time": True,
        "post_language_temp_files": len(list(temp_root.iterdir())) if temp_root.exists() else 0,
    }
    return feature_row, curve_rows, coverage


def run(config_path: Path, mode: str, device: str = "cpu", overwrite: bool = False, languages: list[str] | None = None, max_languages: int | None = None):
    del device
    settings = load_settings(config_path, mode)
    out_root = settings.results_root
    out_root.mkdir(parents=True, exist_ok=True)
    temp_root = settings.temp_root
    if overwrite and out_root.exists():
        for name in ["language_features.csv", "locality_curves.csv", "source_coverage.csv", "run_metadata.json"]:
            (out_root / name).unlink(missing_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    _prepare_results_store(out_root)
    for child in temp_root.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
    sources = HuggingFaceSources()
    seed = int(settings.raw["sampling"].get("seed", 20260916))

    registry = {k: v for k, v in FLEURS_LANGUAGES.items() if languages is None or k in languages}
    if languages:
        missing = sorted(set(languages) - set(registry))
        if missing:
            raise ValueError(f"Unknown language(s): {missing}")
    if max_languages is not None:
        ordered_names = list(registry)[: int(max_languages)]
        registry = {k: registry[k] for k in ordered_names}

    # Metadata ordering only; this never downloads data. Smaller total train
    # Parquet is processed first so a disk-limited run accumulates results early.
    listing = sources.list_files(FLEURS_REPO, revision=FLEURS_REVISION)
    sizes = {
        cfg: sum(f.size for f in sources.fleurs_train_files(listing, cfg))
        for cfg in [m["fleurs_config"] for m in registry.values()]
    }
    ordered = sorted(registry.items(), key=lambda kv: (sizes.get(kv[1]["fleurs_config"], 10**30), kv[0]))

    if mode == "pilot":
        ordered = ordered[: min(3, len(ordered))]

    metadata = {
        "analysis_version": ANALYSIS_VERSION,
        "mode": mode,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fleurs_repo": FLEURS_REPO,
        "fleurs_revision": FLEURS_REVISION,
        "ud_repo": UD_REPO,
        "ud_revision": UD_REVISION,
        "language_count_requested": len(ordered),
        "languages": [name for name, _ in ordered],
        "language_order": "ascending total FLEURS train Parquet bytes; metadata-only sort",
        "storage_policy": "exactly one downloaded Parquet file at a time; each file deleted in finally-block after streaming batch processing; no persistent HF dataset cache",
        "fleurs_role": "paired speech + transcription + duration calibration and transcript locality",
        "ud_role": "independent annotated written-language structural/morphological predictors",
        "special_fleurs_all_config_not_used": True,
    }
    write_json(out_root / "run_metadata.json", metadata)

    feature_path = out_root / "language_features.csv"
    curve_path = out_root / "locality_curves.csv"
    coverage_path = out_root / "source_coverage.csv"

    for i, (language, meta) in enumerate(ordered, start=1):
        if not overwrite and mode == "full" and _completed(out_root, language):
            print(f"[{i}/{len(ordered)}] {language}: checkpoint exists; skipping.")
            continue
        largest_source = max([sizes.get(meta["fleurs_config"], 0)], default=0)
        free = free_bytes(temp_root)
        min_free = int(float(settings.raw["data"].get("minimum_free_gb", 8)) * (1024 ** 3))
        if largest_source > max(0, free - min_free):
            print(f"[{i}/{len(ordered)}] {language}: skipped before download; largest train shard {largest_source / 2**30:.2f} GiB does not fit current free space policy.")
            coverage = {"language": language, "fleurs_config": meta["fleurs_config"], "fleurs_status": "skipped_insufficient_disk_space", "fleurs_train_bytes": largest_source, "one_file_at_a_time": True}
            _append_csv(coverage_path, [coverage])
            continue
        try:
            print(f"\n[{i}/{len(ordered)}] Processing {language} ({meta['fleurs_config']})")
            feature, curves, coverage = _run_one(language, meta, settings, sources, stable_seed(language, base=seed))
            _append_csv(feature_path, [feature])
            _append_csv(curve_path, curves)
            _append_csv(coverage_path, [coverage])
            _write_checkpoint(out_root, language, {"fleurs_config": meta["fleurs_config"], "ud_treebank": coverage.get("ud_treebank", ""), "n_utterances": feature.get("fleurs_n_utterances"), "n_ud_sentences": feature.get("ud_n_sentences", 0)})
            print(f"[{i}/{len(ordered)}] {language}: complete; retained files=0")
        except InsufficientDataError as exc:
            _assert_one_file_or_none(temp_root)
            coverage = {
                "language": language,
                "fleurs_config": meta["fleurs_config"],
                "fleurs_status": "insufficient_data",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "one_file_at_a_time": True,
            }
            _append_csv(coverage_path, [coverage])
            print(f"[{i}/{len(ordered)}] {language}: INSUFFICIENT_DATA: {exc}")
        except Exception as exc:
            _assert_one_file_or_none(temp_root)
            coverage = {"language": language, "fleurs_config": meta["fleurs_config"], "fleurs_status": "failed", "error_type": type(exc).__name__, "error": str(exc), "one_file_at_a_time": True}
            _append_csv(coverage_path, [coverage])
            print(f"[{i}/{len(ordered)}] {language}: FAILED: {exc}")
            traceback.print_exc()

    features = _read_result_csv(feature_path)
    curves = _read_result_csv(curve_path)
    if "I_t_bits" in curves.columns and "information_bits" not in curves.columns:
        curves = curves.rename(columns={"I_t_bits": "information_bits"})
    if not features.empty:
        features = features.drop_duplicates(subset=["language"], keep="last")
        features.to_csv(feature_path, index=False)
    if not curves.empty:
        curves = curves.drop_duplicates(subset=["language", "lag_characters"], keep="last")
        curves.to_csv(curve_path, index=False)
    if not features.empty:
        _plot_curves(curves, out_root / "plots" / "locality_curves.png")
        _plot_decay(features, out_root / "plots" / "morphology_vs_locality.png")
        if mode == "full":
            try:
                fit_models(features, out_root / "regression", include_interactions=bool(settings.raw["regression"].get("include_interactions", True)))
            except Exception as exc:
                write_json(out_root / "regression_failed.json", {"error": str(exc), "type": type(exc).__name__})

    _assert_one_file_or_none(temp_root)
    print(f"\nFinished. Languages with feature rows: {len(features)}")
    print(f"Temporary directory retained files: {len(list(temp_root.iterdir())) if temp_root.exists() else 0}")
