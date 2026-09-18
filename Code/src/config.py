from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml

@dataclass
class Settings:
    raw: dict[str, Any]
    mode: str

    @property
    def languages(self) -> dict[str, dict[str, str]]:
        return self.raw["languages"]

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    def _resolve_project_path(self, value: str | Path) -> Path:
        p = Path(value)
        return p if p.is_absolute() else self.project_root / p

    @property
    def data_root(self) -> Path:
        return self._resolve_project_path(self.raw["data"]["root"])

    @property
    def temp_root(self) -> Path:
        return self._resolve_project_path(self.raw["data"].get("temp", "data/hf_one_file_tmp"))

    @property
    def results_root(self) -> Path:
        return self._resolve_project_path(self.raw["data"]["results"])

    def sampling(self) -> dict[str, Any]:
        return self.raw["sampling"][self.mode]


def load_settings(path: Path, mode: str) -> Settings:
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if mode not in raw.get("sampling", {}):
        raise ValueError(f"No sampling configuration for mode={mode}")
    return Settings(raw=raw, mode=mode)
