from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from entropy import entropy_rate_from_utterances
from normalization import normalize_text
from corpora import UDSentence

_TOKEN_RE = re.compile(r"\S+", re.UNICODE)
ESTIMATOR_VERSION = "koplenig_lz_longest_match_v6_ud_annotations"


def _words(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalize_text(text))


def _mean_lengths(words: list[str]) -> tuple[float, float]:
    if not words:
        return float("nan"), float("nan")
    x = np.fromiter((len(w) for w in words), dtype=np.float64)
    return float(x.mean()), float(np.median(x))


def _shuffle_word_order(utterances: Sequence[str], rng: np.random.Generator) -> list[str]:
    out = []
    for text in utterances:
        words = _words(text)
        if len(words) >= 2:
            rng.shuffle(words)
        out.append(" ".join(words))
    return out


def _mask_word_structure(utterances: Sequence[str], rng: np.random.Generator) -> list[str]:
    """Destroy within-word character regularities while preserving word types, lengths and order."""
    original_words = [_words(t) for t in utterances]
    alphabet = sorted({c for ws in original_words for w in ws for c in w})
    if not alphabet:
        return [" ".join(ws) for ws in original_words]
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for ws in original_words:
        for word in ws:
            if len(word) < 2 or word in mapping:
                continue
            replacement = "".join(rng.choice(alphabet, size=len(word)).tolist())
            for _ in range(100):
                if replacement not in used:
                    break
                replacement = "".join(rng.choice(alphabet, size=len(word)).tolist())
            mapping[word] = replacement
            used.add(replacement)
    return [" ".join(mapping.get(word, word) for word in ws) for ws in original_words]


def _entropy_triplet(utterance_texts: Sequence[str], seed: int) -> tuple[float, float, float]:
    original = [normalize_text(t) for t in utterance_texts if normalize_text(t)]
    if not original:
        return float("nan"), float("nan"), float("nan")
    h_original = entropy_rate_from_utterances(original, seed=seed)
    h_order = entropy_rate_from_utterances(_shuffle_word_order(original, np.random.default_rng(seed + 101)), seed=seed)
    h_structure = entropy_rate_from_utterances(_mask_word_structure(original, np.random.default_rng(seed + 202)), seed=seed)
    return h_original, h_order, h_structure


def _token_stats(utterance_texts: Sequence[str]) -> dict[str, float]:
    normalized = [normalize_text(t) for t in utterance_texts if normalize_text(t)]
    words = [w for t in normalized for w in _words(t)]
    mean_len, median_len = _mean_lengths(words)
    n_words = len(words)
    n_types = len(set(words))
    return {
        "n_words": float(n_words),
        "n_word_types": float(n_types),
        "mean_word_length_chars": mean_len,
        "median_word_length_chars": median_len,
        "lexical_ttr": (n_types / n_words) if n_words else float("nan"),
        "n_chars": float(sum(len(t) for t in normalized)),
        "mean_chars_per_utterance": float(np.mean([len(t) for t in normalized])) if normalized else float("nan"),
    }


def all_text_features(text: str, seed: int = 20260815, utterance_texts: Sequence[str] | None = None, entropy_order: int = 5) -> dict[str, float | str]:
    del entropy_order
    normalized_utts = [normalize_text(t) for t in (utterance_texts or [text]) if normalize_text(t)]
    stats = _token_stats(normalized_utts)
    h_original, h_order, h_structure = _entropy_triplet(normalized_utts, seed)
    d_order = h_order - h_original if np.isfinite(h_order) and np.isfinite(h_original) else float("nan")
    d_structure = h_structure - h_original if np.isfinite(h_structure) and np.isfinite(h_original) else float("nan")
    denom = d_order + d_structure if np.isfinite(d_order) and np.isfinite(d_structure) else float("nan")
    identified = np.isfinite(d_order) and np.isfinite(d_structure) and d_order >= 0 and d_structure >= 0 and denom > 0
    return {**stats,
        "H_original": h_original, "H_order": h_order, "H_structure": h_structure,
        "word_order_redundancy": d_order, "word_structure_redundancy": d_structure,
        "word_order_redundancy_raw": d_order, "word_structure_redundancy_raw": d_structure,
        "morphology_index": float(d_structure / denom) if identified else float("nan"),
        "morphology_index_status": "identified" if identified else "not_identified_nonnegative_redundancy_required",
        "morphology_estimator_version": ESTIMATOR_VERSION,
        "char_entropy_rate_bits": h_original,
        "morphology_corpus_characters": float(sum(len(t) for t in normalized_utts) + max(len(normalized_utts)-1, 0)),
        "morphology_character_definition": "normalized_utterances_joined_by_single_spaces; Koplenig-style entropy-rate corpus",
    }


def _feat_count(feats: str) -> int:
    if not feats or feats == "_":
        return 0
    return sum(1 for x in feats.split("|") if "=" in x)


def ud_annotated_features(sentences: Sequence[UDSentence], seed: int = 20260815) -> dict[str, float | str]:
    texts = [s.text for s in sentences if s.text]
    out = all_text_features(" ".join(texts), seed=seed, utterance_texts=texts)
    n_tokens = 0
    tokens_with_feats = 0
    feat_counts = []
    dependency_distances = []
    directions = []
    for sent in sentences:
        for i, tok in enumerate(sent.tokens):
            if tok == "_":
                continue
            n_tokens += 1
            feat = sent.feats[i] if i < len(sent.feats) else "_"
            k = _feat_count(feat)
            feat_counts.append(k)
            if k > 0:
                tokens_with_feats += 1
            head = sent.heads[i] if i < len(sent.heads) else None
            if head is not None and head > 0 and i + 1 != head:
                dependency_distances.append(abs((i + 1) - head))
                directions.append("head_before" if head < i + 1 else "head_after")

    if directions:
        p = np.mean([d == "head_before" for d in directions])
        q = 1 - p
        direction_entropy = float(-(p*np.log2(p) if p > 0 else 0) - (q*np.log2(q) if q > 0 else 0))
    else:
        direction_entropy = float("nan")
    out.update({
        "ud_n_sentences": float(len(sentences)),
        "ud_n_tokens": float(n_tokens),
        "ud_morph_feature_density": float(tokens_with_feats / n_tokens) if n_tokens else float("nan"),
        "ud_mean_feats_per_token": float(np.mean(feat_counts)) if feat_counts else float("nan"),
        "ud_dependency_direction_entropy_bits": direction_entropy,
        "ud_mean_dependency_distance": float(np.mean(dependency_distances)) if dependency_distances else float("nan"),
        "morphology_source": "Universal Dependencies 2.18 treebank",
    })
    return out
