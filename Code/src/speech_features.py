from __future__ import annotations

import re
import unicodedata
from typing import Sequence

import numpy as np

_TOKEN_RE = re.compile(r"\S+", re.UNICODE)
# Broad orthographic vowel inventory. For abugidas and scripts without explicit
# vowel letters, the estimator falls back to one syllable per orthographic word.
VOWELS = set("aeiouyAEIOUYáéíóúýÁÉÍÓÚÝàèìòùÀÈÌÒÙäëïöüÿÄËÏÖÜŸâêîôûÂÊÎÔÛãõÃÕåÅæÆœŒøØāēīōūĀĒĪŌŪăĕĭŏŭĂĔĬŎŬąęįųĄĘĮŲėėĖįĮőŐűŰ")
CJK_RANGES = ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x3040, 0x30FF), (0xAC00, 0xD7AF))


def _in_cjk_script(text: str) -> bool:
    return any(any(lo <= ord(ch) <= hi for lo, hi in CJK_RANGES) for ch in text)


def estimate_syllables(text: str, language_code: str = "") -> int:
    words = _TOKEN_RE.findall(text)
    if not words:
        return 0
    if _in_cjk_script(text):
        return sum(sum(any(lo <= ord(ch) <= hi for lo, hi in CJK_RANGES[:4]) for ch in w) for w in words)
    total = 0
    for word in words:
        groups = 0
        prev_vowel = False
        for ch in word:
            if ch in VOWELS:
                if not prev_vowel:
                    groups += 1
                prev_vowel = True
            else:
                prev_vowel = False
        total += max(1, groups)
    return max(1, total)


def aggregate_speech_features(texts: Sequence[str], durations: Sequence[float], language_code: str = "") -> dict[str, float | str]:
    if len(texts) != len(durations) or not texts:
        raise ValueError("texts and durations must be equally sized and non-empty")
    words = [len(_TOKEN_RE.findall(t)) for t in texts]
    chars = [sum(not ch.isspace() for ch in t) for t in texts]
    syllables = [estimate_syllables(t, language_code) for t in texts]
    total_time = float(np.sum(durations))
    return {
        "word_rate_wps": float(np.sum(words) / total_time) if total_time > 0 else float("nan"),
        "syllable_rate_sps": float(np.sum(syllables) / total_time) if total_time > 0 else float("nan"),
        "speech_char_rate_cps": float(np.sum(chars) / total_time) if total_time > 0 else float("nan"),
        "seconds_per_character": float(total_time / max(1, np.sum(chars))),
        "mean_duration_s": float(np.mean(durations)),
        "median_duration_s": float(np.median(durations)),
        "estimated_syllable_total": float(np.sum(syllables)),
        "syllable_estimator": "unicode_orthographic_heuristic_v2; approximate, not gold phonological annotation",
    }
