from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    out: list[str] = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith(("L", "N", "M")):
            out.append(ch)
        elif ch.isspace():
            out.append(" ")
        else:
            out.append(" ")
    text = re.sub(r"\s+", " ", "".join(out)).strip()
    return text


def char_tokens(text: str) -> list[str]:
    text = normalize_text(text)
    return list(text)


def word_tokens(text: str) -> list[str]:
    text = normalize_text(text)
    return [w for w in text.split(" ") if w]


def sentence_stream(texts: list[str]) -> str:
    cleaned = []
    for text in texts:
        t = normalize_text(text)
        if len(t) >= 2:
            cleaned.append(t)
    return " ".join(cleaned)
