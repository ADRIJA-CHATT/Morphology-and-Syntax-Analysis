from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _lpf_entropy(text: str) -> float:
    """
    Estimate entropy rate in bits/character using the non-parametric
    Lempel-Ziv longest-match estimator used by Koplenig et al. (2017):

        H = [ (1/N) sum_{i=2}^N l_i / log_2(i) ]^{-1},

    where l_i is one plus the longest substring beginning at position i that
    has already occurred in the preceding text.

    `pydivsufsort.longest_previous_factor` computes all longest previous
    factors efficiently from a suffix array/LCP representation.
    """
    from pydivsufsort import longest_previous_factor

    if len(text) < 3:
        return float("nan")

    # Map Unicode characters to compact integer symbols. This is lossless for
    # the character sequence and avoids dependence on byte encoding.
    alphabet: dict[str, int] = {}
    codes = np.empty(len(text), dtype=np.int32)
    next_code = 0
    for i, ch in enumerate(text):
        code = alphabet.get(ch)
        if code is None:
            code = next_code
            alphabet[ch] = code
            next_code += 1
        codes[i] = code

    lpf = np.asarray(longest_previous_factor(codes), dtype=np.int64)
    if len(lpf) != len(text):
        raise RuntimeError("LZ longest-previous-factor output has the wrong length.")

    # Paper notation is 1-indexed. For Python index j=i-1, log2(i)=log2(j+1).
    positions = np.arange(2, len(text) + 1, dtype=np.float64)
    match_lengths = lpf[1:] + 1.0
    denom = np.log2(positions)
    mean_description = float(np.sum(match_lengths / denom) / len(text))
    if not np.isfinite(mean_description) or mean_description <= 0:
        return float("nan")
    return float(1.0 / mean_description)


def entropy_rate_lz(text: str) -> float:
    return _lpf_entropy(str(text))


def entropy_rate_from_utterances(utterance_texts: Sequence[str], seed: int = 20260815) -> float:
    """
    Estimate one corpus entropy rate after deterministically randomizing
    utterance/"verse" order, analogous to Koplenig et al.'s randomization of
    verse order to reduce supra-verse dependence. Word-internal and word-order
    manipulations are performed separately by the morphology module.
    """
    texts = [str(t).strip() for t in utterance_texts if str(t).strip()]
    if not texts:
        return float("nan")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(texts))
    # A single space preserves the word-separator convention of the entropy
    # estimator; this separator is not used in the speech/time character count.
    corpus = " ".join(texts[i] for i in order)
    return entropy_rate_lz(corpus)


def conditional_entropy_ngram(text: str, order: int = 5) -> float:
    """
    Retained for backward compatibility with earlier pilot code.
    This is a finite-order plug-in conditional entropy, not the Koplenig LZ
    estimator used by the scientific morphology variables.
    """
    text = str(text)
    if len(text) <= order:
        return float("nan")
    ctx = {}
    counts = {}
    for i in range(order, len(text)):
        key = text[i - order:i]
        ctx[key] = ctx.get(key, 0) + 1
        counts[(key, text[i])] = counts.get((key, text[i]), 0) + 1
    n = sum(ctx.values())
    if n == 0:
        return float("nan")
    h = 0.0
    for (key, ch), c in counts.items():
        h -= (c / n) * math.log2(c / ctx[key])
    return float(h)
