from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / "additionals" / ".env")

# EXACT general Common Voice Scripted Speech 26.0 datasets requested.
# These are NOT accent/gender/region-specific subsets.
COMMON_VOICE_26: dict[str, dict[str, str]] = {
    "English": {"id": "cmqim2hn800ssnr07gvmpcnwu", "locale": "en", "title": "Common Voice Scripted Speech 26.0 - English"},
    "German": {"id": "cmqim3xpi00t6nr07k0myqtkr", "locale": "de", "title": "Common Voice Scripted Speech 26.0 - German"},
    "Dutch": {"id": "cmqinokkq00wwnr07hv5oax8l", "locale": "nl", "title": "Common Voice Scripted Speech 26.0 - Dutch"},
    "Swedish": {"id": "cmqintw0q00xwnr07mjjzpe61", "locale": "sv-SE", "title": "Common Voice Scripted Speech 26.0 - Swedish"},
    "French": {"id": "cmqim41b000tanr07q9btypkc", "locale": "fr", "title": "Common Voice Scripted Speech 26.0 - French"},
    "Spanish": {"id": "cmqim2spa00synr071fcp7av0", "locale": "es", "title": "Common Voice Scripted Speech 26.0 - Spanish"},
    "Italian": {"id": "cmqini14100vmnq07309ocknr", "locale": "it", "title": "Common Voice Scripted Speech 26.0 - Italian"},
    "Portuguese": {"id": "cmqinmnkf00w8nr07hkbbxgw7", "locale": "pt", "title": "Common Voice Scripted Speech 26.0 - Portuguese"},
    "Russian": {"id": "cmqinj9g500vsnr07qf4hmr3j", "locale": "ru", "title": "Common Voice Scripted Speech 26.0 - Russian"},
    "Polish": {"id": "cmqinmu2a00winq07gyrtri0q", "locale": "pl", "title": "Common Voice Scripted Speech 26.0 - Polish"},
    "Czech": {"id": "cmqinmdik00wanq07o6cqnhps", "locale": "cs", "title": "Common Voice Scripted Speech 26.0 - Czech"},
    "Turkish": {"id": "cmqinosfq00x4nr07gnk0rdf9", "locale": "tr", "title": "Common Voice Scripted Speech 26.0 - Turkish"},
    "Finnish": {"id": "cmqiodpav00yknq07kfcyu7c0", "locale": "fi", "title": "Common Voice Scripted Speech 26.0 - Finnish"},
    "Hungarian": {"id": "cmqinob6900wknr07s6fgcprx", "locale": "hu", "title": "Common Voice Scripted Speech 26.0 - Hungarian"},
    "Indonesian": {"id": "cmqinqwef00xonr07z4vbfovw", "locale": "id", "title": "Common Voice Scripted Speech 26.0 - Indonesian"},
    "Vietnamese": {"id": "cmqiof4sv00ywnq07m63v4pnp", "locale": "vi", "title": "Common Voice Scripted Speech 26.0 - Vietnamese"},
    "Hindi": {"id": "cmqiod71900zgnr07uiyw57br", "locale": "hi", "title": "Common Voice Scripted Speech 26.0 - Hindi"},
    "Mandarin Chinese": {"id": "cmqim47x700tunq074za20dq1", "locale": "zh-CN", "title": "Common Voice Scripted Speech 26.0 - Chinese (China)"},
    "Japanese": {"id": "cmqim4lxy00tunr07cjkcupeg", "locale": "ja", "title": "Common Voice Scripted Speech 26.0 - Japanese"},
    "Korean": {"id": "cmqi922c5001pnq07dmj0oypw", "locale": "ko", "title": "Common Voice Scripted Speech 26.0 - Korean"},
}


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if hasattr(obj, name):
        return getattr(obj, name)
    try:
        return obj.get(name, default)
    except Exception:
        return default


def preflight_common_voice(languages: list[str]) -> list[dict[str, str]]:
    """
    Verify the exact Common Voice v26 dataset identity for each requested
    language WITHOUT downloading anything. Returns one row per language with
    a stable shape so callers (e.g. check_common_voice_access.py) can render
    a simple report:

        {"language": ..., "status": "verified" | "needs_access" | "error",
         "dataset_id": ..., "dataset_name": ...}

    "needs_access" means the dataset identity is registered but MDC has not
    yet granted this API key access (the dataset-specific terms have not
    been accepted through the web interface). "error" covers anything else
    (network failure, unexpected metadata mismatch, etc.) and includes the
    exception text in "dataset_name" for debugging.
    """
    rows: list[dict[str, str]] = []
    for language in languages:
        spec = COMMON_VOICE_26.get(language)
        if spec is None:
            rows.append({
                "language": language,
                "status": "error",
                "dataset_id": "",
                "dataset_name": "No verified Common Voice v26 dataset registered for this language.",
            })
            continue
        try:
            get_common_voice_details(language)
            rows.append({
                "language": language,
                "status": "verified",
                "dataset_id": spec["id"],
                "dataset_name": spec["title"],
            })
        except PermissionError as exc:
            rows.append({
                "language": language,
                "status": "needs_access",
                "dataset_id": spec["id"],
                "dataset_name": (
                    f"Open https://mozilladatacollective.com/datasets/{spec['id']}, "
                    "read/accept the dataset-specific terms, then rerun."
                ),
            })
        except Exception as exc:
            msg = str(exc)
            is_access_issue = any(
                x in msg.lower() for x in ("terms", "conditions", "access denied", "forbidden", "403")
            )
            rows.append({
                "language": language,
                "status": "needs_access" if is_access_issue else "error",
                "dataset_id": spec["id"],
                "dataset_name": (
                    f"Open https://mozilladatacollective.com/datasets/{spec['id']}, "
                    "read/accept the dataset-specific terms, then rerun."
                ) if is_access_issue else msg,
            })
    return rows


def get_common_voice_details(language: str):
    if language not in COMMON_VOICE_26:
        raise KeyError(f"No verified Common Voice v26 dataset for {language}.")
    try:
        from datacollective import get_dataset_details
    except ImportError as exc:
        raise RuntimeError(
            "datacollective is not installed. Install requirements.txt in the active venv."
        ) from exc

    details = get_dataset_details(COMMON_VOICE_26[language]["id"])
    expected = COMMON_VOICE_26[language]
    actual_title = str(_attr(details, "name", ""))
    actual_locale = str(_attr(details, "locale", ""))
    actual_task = str(_attr(details, "task", "")).upper()

    if actual_title != expected["title"]:
        raise RuntimeError(
            f"Common Voice safety check failed for {language}: expected exact dataset "
            f"'{expected['title']}', got '{actual_title}'."
        )
    if actual_locale != expected["locale"]:
        raise RuntimeError(
            f"Common Voice safety check failed for {language}: expected locale "
            f"'{expected['locale']}', got '{actual_locale}'."
        )
    if actual_task != "ASR":
        raise RuntimeError(
            f"Common Voice safety check failed for {language}: expected ASR, got '{actual_task}'."
        )
    return details


def get_common_voice_size_bytes(language: str) -> int | None:
    """
    Archive size in bytes for one language's Common Voice v26 dataset, read
    from MDC metadata WITHOUT downloading the archive. Returns None if the
    size is unavailable for any reason (no access yet, offline, SDK/API
    change). Never raises: callers use this for ordering/reporting only, and
    a missing size must not abort a run.
    """
    try:
        details = get_common_voice_details(language)
    except Exception:
        return None
    size = _attr(details, "sizeBytes", None)
    try:
        size = int(size)
    except (TypeError, ValueError):
        return None
    return size if size > 0 else None


def order_common_voice_languages_by_size(
    languages: list[str], *, verbose: bool = True
) -> tuple[list[str], dict[str, int | None]]:
    """
    Order languages by Common Voice archive size, smallest first.

    Rationale: archive sizes span roughly three orders of magnitude (Korean
    is a few GB; English is ~88 GB). Processing smallest-first means most
    languages produce results early, and a failure partway through a long
    run costs the least remaining work. It also surfaces configuration or
    credential problems within minutes instead of after a multi-hour
    download.

    Languages whose size could not be determined are placed AFTER all
    known-size languages, preserving their relative config order among
    themselves, so an unknown size can never push a small language behind a
    large one. Returns (ordered_languages, sizes_by_language).
    """
    sizes: dict[str, int | None] = {}
    if verbose:
        print("[ORDER] Querying Common Voice archive sizes (metadata only, no downloads)...")
    for language in languages:
        sizes[language] = get_common_voice_size_bytes(language)

    known = [l for l in languages if sizes[l] is not None]
    unknown = [l for l in languages if sizes[l] is None]
    ordered = sorted(known, key=lambda l: sizes[l]) + unknown

    if verbose:
        for i, language in enumerate(ordered, 1):
            size = sizes[language]
            size_str = f"{size / (1024 ** 3):8.1f} GB" if size is not None else "   unknown"
            print(f"[ORDER] {i:2d}. {language:<18s} {size_str}")
        if unknown:
            print(
                f"[ORDER] {len(unknown)} language(s) had no size available and were placed "
                "last. This usually means dataset terms have not been accepted yet, or the "
                "metadata request failed; those languages will report a clear error when "
                "their turn comes."
            )
        total = sum(s for s in sizes.values() if s)
        if total:
            print(f"[ORDER] Total download across sized languages: {total / (1024 ** 3):.1f} GB")
    return ordered, sizes


def download_common_voice(language: str, out_dir: Path) -> Path:
    """
    Download exactly one requested general Common Voice v26 archive.
    MDC itself handles resumable .part/.checksum downloads.
    """
    details = get_common_voice_details(language)
    dataset_id = COMMON_VOICE_26[language]["id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datacollective import download_dataset
        archive = download_dataset(
            dataset_id,
            download_directory=str(out_dir),
            show_progress=True,
            overwrite_existing=False,
            enable_logging=False,
        )
        return Path(archive)
    except PermissionError as exc:
        raise RuntimeError(
            f"MDC access is not enabled for {language}. "
            f"Open https://mozilladatacollective.com/datasets/{dataset_id}, "
            "read/accept the dataset-specific terms, then rerun. "
            "No manual file download is required."
        ) from exc
    except Exception as exc:
        # MDC currently reports terms/access failures in a few SDK exception paths
        # with RuntimeError/HTTPError instead of PermissionError.
        msg = str(exc)
        if any(x in msg.lower() for x in (
            "terms", "conditions", "access denied", "forbidden", "403"
        )):
            raise RuntimeError(
                f"MDC access is not enabled for {language}. "
                f"Open https://mozilladatacollective.com/datasets/{dataset_id}, "
                "read/accept the dataset-specific terms, then rerun. "
                "No manual file download is required."
            ) from exc
        raise


def hf_download_exact(
    repo_id: str,
    filename: str,
    revision: str = "main",
    cache_dir: Optional[str] = None,
) -> Path:
    from huggingface_hub import hf_hub_download

    cache = cache_dir or os.getenv("HF_DATASETS_CACHE") or os.getenv("HF_HOME")
    try:
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                revision=revision,
                cache_dir=cache,
            )
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not download Hugging Face dataset file "
            f"'{repo_id}/{filename}' at revision '{revision}'."
        ) from exc