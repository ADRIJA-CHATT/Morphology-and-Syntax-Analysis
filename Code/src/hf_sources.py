from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import os
import time

import requests
from huggingface_hub import HfApi, hf_hub_url
from dotenv import load_dotenv


@dataclass(frozen=True)
class HFFileInfo:
    path: str
    size: int


class HuggingFaceSources:
    """Small, explicit HF Hub client used without persistent dataset caching.

    The pipeline deliberately downloads one Parquet file into a temporary
    directory, consumes it in record batches, and deletes it before moving on.
    """

    def __init__(self, token: str | None = None, timeout_connect_s: int = 30, timeout_read_s: int = 180):
        load_dotenv(Path(__file__).resolve().parents[1] / "additionals" / ".env")
        self.token = token or os.getenv("HF_TOKEN") or None
        self.api = HfApi(token=self.token)
        self.timeout = (timeout_connect_s, timeout_read_s)

    def repo_sha(self, repo_id: str, revision: str = "main") -> str:
        info = self.api.repo_info(repo_id=repo_id, repo_type="dataset", revision=revision)
        return getattr(info, "sha", revision)

    def list_files(self, repo_id: str, revision: str = "main") -> list[HFFileInfo]:
        out: list[HFFileInfo] = []
        for item in self.api.list_repo_tree(
            repo_id=repo_id,
            path_in_repo="",
            recursive=True,
            revision=revision,
            repo_type="dataset",
        ):
            path = getattr(item, "path", None)
            if not path:
                continue
            size = getattr(item, "size", None)
            if size is None:
                # Directories and a few metadata entries are not useful here.
                continue
            out.append(HFFileInfo(path=str(path), size=int(size)))
        return out

    @staticmethod
    def fleurs_train_files(files: Iterable[HFFileInfo], config: str) -> list[HFFileInfo]:
        prefixes = (f"{config}/train-", f"{config}/train/", f"parquet-data/{config}/train-", f"parquet-data/{config}/train/")
        selected = [f for f in files if f.path.endswith(".parquet") and (f.path.startswith(prefixes[0]) or f.path.startswith(prefixes[1]) or f.path.startswith(prefixes[2]) or f.path.startswith(prefixes[3]))]
        return sorted(selected, key=lambda x: x.path)

    @staticmethod
    def ud_train_files(files: Iterable[HFFileInfo], config: str) -> list[HFFileInfo]:
        prefixes = (f"{config}/train", f"parquet/{config}/train")
        selected = [f for f in files if f.path.endswith(".parquet") and f.path.startswith(prefixes)]
        return sorted(selected, key=lambda x: x.path)

    def download_file(self, repo_id: str, filename: str, revision: str, destination: Path, chunk_bytes: int = 8 << 20) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.unlink()
        partial = destination.with_suffix(destination.suffix + ".part")
        if partial.exists():
            partial.unlink()

        url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision, repo_type="dataset")
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        started = time.time()
        try:
            with requests.get(url, headers=headers, stream=True, timeout=self.timeout, allow_redirects=True) as response:
                response.raise_for_status()
                with partial.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=chunk_bytes):
                        if chunk:
                            fh.write(chunk)
            partial.replace(destination)
            elapsed = time.time() - started
            print(f"[download] {repo_id}:{filename} -> {destination} ({destination.stat().st_size:,} bytes, {elapsed:.1f}s)")
            return destination
        except Exception:
            try:
                partial.unlink(missing_ok=True)
            except OSError:
                pass
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    @staticmethod
    def delete_file(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            print(f"[storage] warning: could not delete {path}: {exc}")


FLEURS_REPO = "google/fleurs"
UD_REPO = "universal-dependencies/universal_dependencies"
FLEURS_REVISION = "main"
UD_REVISION = "2.18"
