from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import shutil
import numpy as np


def stable_seed(*parts: object, base: int = 0) -> int:
    raw = "|".join(map(str, parts))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return (int(digest[:12], 16) + base) % (2**32 - 1)


def reservoir_append(items: list[Any], item: Any, seen: int, max_items: int, rng: np.random.Generator) -> int:
    seen += 1
    if max_items <= 0:
        return seen
    if len(items) < max_items:
        items.append(item)
    else:
        j = int(rng.integers(0, seen))
        if j < max_items:
            items[j] = item
    return seen


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def free_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path).free)
