from __future__ import annotations

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_sources import HuggingFaceSources, FLEURS_REPO, FLEURS_REVISION, UD_REPO, UD_REVISION
from language_registry import FLEURS_LANGUAGES


def main():
    src = HuggingFaceSources()
    fleurs_files = src.list_files(FLEURS_REPO, FLEURS_REVISION)
    ud_files = src.list_files(UD_REPO, UD_REVISION)
    fleur_configs = {m["fleurs_config"] for m in FLEURS_LANGUAGES.values()}
    found_configs = {m["fleurs_config"] for m in FLEURS_LANGUAGES.values() if src.fleurs_train_files(fleurs_files, m["fleurs_config"])}
    summary = {
        "fleurs_repo": FLEURS_REPO,
        "fleurs_revision": FLEURS_REVISION,
        "fleurs_language_registry_count": len(FLEURS_LANGUAGES),
        "fleurs_train_configs_found": len(found_configs),
        "fleurs_missing_train_configs": sorted(fleur_configs - found_configs),
        "ud_repo": UD_REPO,
        "ud_revision": UD_REVISION,
        "ud_parquet_file_count": len([f for f in ud_files if f.path.endswith(".parquet")]),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["fleurs_missing_train_configs"]:
        raise SystemExit("FLEURS registry is missing one or more train configurations; inspect before a full run.")

if __name__ == "__main__":
    main()
