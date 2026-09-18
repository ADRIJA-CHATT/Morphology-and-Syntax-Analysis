from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import math
import re

import numpy as np
import pyarrow.parquet as pq

from hf_sources import HuggingFaceSources, HFFileInfo, FLEURS_REPO, UD_REPO, FLEURS_REVISION, UD_REVISION
from normalization import normalize_text
from utils import reservoir_append, stable_seed


@dataclass(frozen=True)
class Utterance:
    language: str
    text: str
    duration_s: float
    source: str
    speaker_id: str | None = None


@dataclass(frozen=True)
class UDSentence:
    text: str
    tokens: tuple[str, ...]
    upos: tuple[str, ...]
    feats: tuple[str, ...]
    heads: tuple[int | None, ...]
    deprels: tuple[str, ...]


@dataclass(frozen=True)
class UDTreebankSpec:
    config: str
    train_files: tuple[HFFileInfo, ...]
    score: tuple[int, int, int]


def _normalise_batch_value(value: Any) -> Any:
    if hasattr(value, "as_py"):
        return value.as_py()
    return value


def validate_fleurs_schema(path: Path) -> dict[str, Any]:
    pf = pq.ParquetFile(path)
    cols = set(pf.schema_arrow.names)
    required = {"transcription", "num_samples"}
    missing = sorted(required - cols)
    if missing:
        raise ValueError(f"FLEURS schema mismatch in {path.name}: missing {missing}; got {sorted(cols)}")
    return {
        "columns": sorted(cols),
        "num_row_groups": int(pf.num_row_groups),
        "format": "Parquet",
        "required_columns_ok": True,
        "audio_sample_rate_hz": 16_000,
    }


def validate_ud_schema(path: Path) -> dict[str, Any]:
    pf = pq.ParquetFile(path)
    cols = set(pf.schema_arrow.names)
    required = {"text", "tokens", "upos", "feats", "head", "deprel"}
    missing = sorted(required - cols)
    if missing:
        raise ValueError(f"UD schema mismatch in {path.name}: missing {missing}; got {sorted(cols)}")
    return {
        "columns": sorted(cols),
        "num_row_groups": int(pf.num_row_groups),
        "format": "Parquet (UD CoNLL-U annotations)",
        "required_columns_ok": True,
    }


def _iter_fleurs_utterances(
    path: Path,
    language: str,
    min_chars: int,
    max_chars: int,
    min_duration_s: float,
    max_duration_s: float,
):
    """Yield every usable utterance from one Parquet shard exactly once.

    Sampling is intentionally performed by ``load_fleurs_samples`` across the
    complete set of shards.  We do not cap a shard before the global reservoir,
    because doing so would over-weight early shards.
    """
    schema_meta = validate_fleurs_schema(path)
    pf = pq.ParquetFile(path)
    cols = [c for c in ["transcription", "num_samples", "speaker_id"] if c in pf.schema_arrow.names]
    for batch in pf.iter_batches(batch_size=256, columns=cols, use_threads=True):
        for row in batch.to_pylist():
            text = normalize_text(str(row.get("transcription") or ""))
            if not text:
                continue
            n_chars = sum(not c.isspace() for c in text)
            duration_s = float(row.get("num_samples") or 0) / 16_000.0
            if n_chars < min_chars or n_chars > max_chars:
                continue
            if not (min_duration_s <= duration_s <= max_duration_s):
                continue
            yield Utterance(
                language=language,
                text=text,
                duration_s=duration_s,
                source=f"{FLEURS_REPO}:{path.name}",
                speaker_id=None if row.get("speaker_id") is None else str(row.get("speaker_id")),
            )


def _reservoir_utterances(
    path: Path,
    language: str,
    max_items: int,
    min_chars: int,
    max_chars: int,
    min_duration_s: float,
    max_duration_s: float,
    rng: np.random.Generator,
) -> tuple[list[Utterance], dict[str, Any]]:
    """Compatibility helper used by tests: exact reservoir sampling within one shard."""
    schema_meta = validate_fleurs_schema(path)
    reservoir: list[Utterance] = []
    accepted = 0
    for item in _iter_fleurs_utterances(
        path, language, min_chars, max_chars, min_duration_s, max_duration_s
    ):
        accepted = reservoir_append(reservoir, item, accepted, max_items, rng)
    return reservoir, {**schema_meta, "rows_accepted": accepted, "rows_returned": len(reservoir)}

def load_fleurs_samples(
    sources: HuggingFaceSources,
    language: str,
    config: str,
    work_dir: Path,
    max_utterances: int,
    min_chars: int,
    max_chars: int,
    min_duration_s: float,
    max_duration_s: float,
    seed: int,
    revision: str = FLEURS_REVISION,
) -> tuple[list[Utterance], dict[str, Any]]:
    files = sources.list_files(FLEURS_REPO, revision=revision)
    train_files = sources.fleurs_train_files(files, config)
    if not train_files:
        raise RuntimeError(f"No FLEURS train Parquet files found for config={config}")

    rng = np.random.default_rng(stable_seed(language, "fleurs", base=seed))
    reservoir: list[Utterance] = []
    total_seen = 0
    total_accepted = 0
    schemas = []
    downloaded_bytes = 0

    # True global reservoir sampling: every usable row across every train shard
    # participates exactly once.  Only the current Parquet shard exists on disk.
    for index, file_info in enumerate(train_files):
        local_path = work_dir / f"fleurs_{config}_{index:03d}.parquet"
        rows_accepted_file = 0
        try:
            sources.download_file(FLEURS_REPO, file_info.path, revision, local_path)
            downloaded_bytes += file_info.size
            schema_meta = validate_fleurs_schema(local_path)
            for item in _iter_fleurs_utterances(
                local_path, language, min_chars, max_chars, min_duration_s, max_duration_s
            ):
                rows_accepted_file += 1
                total_accepted += 1
                total_seen = reservoir_append(reservoir, item, total_seen, max_utterances, rng)
            schemas.append({
                "file": file_info.path,
                "size_bytes": file_info.size,
                **schema_meta,
                "rows_accepted": rows_accepted_file,
                "rows_returned_to_global_reservoir": min(rows_accepted_file, max_utterances),
            })
        finally:
            # This runs even on schema/read/processing/download failures.
            sources.delete_file(local_path)

    return reservoir, {
        "repo_id": FLEURS_REPO,
        "revision": revision,
        "config": config,
        "files": [{"path": f.path, "size_bytes": f.size} for f in train_files],
        "downloaded_bytes": downloaded_bytes,
        "schema_checks": schemas,
        "n_utterances": len(reservoir),
        "n_usable_rows_across_all_train_shards": total_accepted,
        "sample_rate_hz": 16_000,
        "sampling_method": "global_row_level_reservoir_across_all_train_shards",
    }

def _language_tokens_from_config(config: str) -> tuple[str, str]:
    prefix = config.split("_", 1)[0]
    return prefix, prefix


def discover_ud_treebank(
    sources: HuggingFaceSources,
    fleurs_lang_code: str,
    iso639_3: str,
    fleurs_config: str,
    revision: str = UD_REVISION,
) -> UDTreebankSpec | None:
    files = sources.list_files(UD_REPO, revision=revision)
    candidates: dict[str, list[HFFileInfo]] = {}
    for file_info in files:
        if not file_info.path.endswith(".parquet"):
            continue
        parts = file_info.path.split("/")
        config = parts[-2] if len(parts) >= 2 and parts[-1].startswith("train") else None
        if config is None:
            continue
        candidates.setdefault(config, []).append(file_info)

    scored: list[UDTreebankSpec] = []
    f2 = fleurs_lang_code.lower()
    f3 = iso639_3.lower()
    fc = fleurs_config.lower().split("_", 1)[0]
    for config, train_files in candidates.items():
        c0 = config.lower().split("_", 1)[0]
        score_exact3 = int(c0 == f3)
        score_exact2 = int(c0 == f2)
        score_fleurs = int(c0 == fc)
        if not (score_exact3 or score_exact2 or score_fleurs):
            continue
        # Prefer explicit ISO-3/ISO-2 matches, then largest train file as a
        # reproducible tie-break because some treebanks are tiny samples.
        score = (score_exact3, score_exact2, score_fleurs)
        scored.append(UDTreebankSpec(config=config, train_files=tuple(sorted(train_files, key=lambda x: x.path)), score=score))

    if not scored:
        return None
    scored.sort(key=lambda s: (s.score, sum(f.size for f in s.train_files)), reverse=True)
    return scored[0]


def _to_int_head(value: Any) -> int | None:
    value = _normalise_batch_value(value)
    if value is None:
        return None
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    return v if v >= 0 else None


def _as_tuple_str(value: Any) -> tuple[str, ...]:
    value = _normalise_batch_value(value)
    if value is None:
        return tuple()
    if isinstance(value, list):
        return tuple("_" if x is None else str(x) for x in value)
    return (str(value),)


def load_ud_sentences(
    sources: HuggingFaceSources,
    language: str,
    spec: UDTreebankSpec,
    work_dir: Path,
    max_sentences: int,
    max_chars: int,
    seed: int,
    revision: str = UD_REVISION,
) -> tuple[list[UDSentence], dict[str, Any]]:
    del seed  # deterministic first-N scan by fixed treebank order
    out: list[UDSentence] = []
    total_chars = 0
    schemas = []

    for index, file_info in enumerate(spec.train_files):
        local_path = work_dir / f"ud_{re.sub(r'[^A-Za-z0-9_.-]+', '_', spec.config)}_{index:03d}.parquet"
        try:
            sources.download_file(UD_REPO, file_info.path, revision, local_path)
            meta = validate_ud_schema(local_path)
            schemas.append({"file": file_info.path, "size_bytes": file_info.size, **meta})
            pf = pq.ParquetFile(local_path)
            cols = [c for c in ["text", "tokens", "upos", "feats", "head", "deprel"] if c in pf.schema_arrow.names]
            for batch in pf.iter_batches(batch_size=256, columns=cols, use_threads=True):
                for row in batch.to_pylist():
                    if len(out) >= max_sentences or total_chars >= max_chars:
                        break
                    tokens = _as_tuple_str(row.get("tokens"))
                    if not tokens:
                        continue
                    text = normalize_text(str(row.get("text") or " ".join(t for t in tokens if t != "_")))
                    if not text:
                        continue
                    if total_chars + len(text) > max_chars and out:
                        break
                    upos = _as_tuple_str(row.get("upos"))
                    feats = _as_tuple_str(row.get("feats"))
                    heads_raw = _normalise_batch_value(row.get("head"))
                    deprels = _as_tuple_str(row.get("deprel"))
                    heads = tuple(_to_int_head(v) for v in (heads_raw if isinstance(heads_raw, list) else []))
                    if len(heads) != len(tokens):
                        heads = tuple((h if h is not None else None) for h in heads) + (None,) * max(0, len(tokens) - len(heads))
                        heads = heads[:len(tokens)]
                    out.append(UDSentence(text=text, tokens=tokens, upos=upos, feats=feats, heads=heads, deprels=deprels))
                    total_chars += len(text)
                if len(out) >= max_sentences or total_chars >= max_chars:
                    break
        finally:
            sources.delete_file(local_path)
        if len(out) >= max_sentences or total_chars >= max_chars:
            break

    return out, {
        "repo_id": UD_REPO,
        "revision": revision,
        "treebank": spec.config,
        "files": [{"path": f.path, "size_bytes": f.size} for f in spec.train_files],
        "schema_checks": schemas,
        "n_sentences": len(out),
        "n_characters": total_chars,
        "language": language,
    }
