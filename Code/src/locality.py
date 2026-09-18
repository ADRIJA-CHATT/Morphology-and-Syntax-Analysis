from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from utils import stable_seed


class InsufficientDataError(ValueError):
    """Raised when a language cannot support the requested locality estimate."""



def _char_count(text: str) -> int:
    return sum(not ch.isspace() for ch in str(text))


@dataclass
class LocalityResult:
    language: str
    lags: list[int]
    surprisal: list[float]
    information: list[float]
    seconds_per_character: float
    effective_train_chars: int = 0
    effective_valid_chars: int = 0
    eval_positions: int = 0


class CharacterKneserNey:
    """Interpolated Kneser-Ney character model of a single fixed order.

    Follows the standard "modified interpolated Kneser-Ney" recursion (Kneser
    & Ney 1995; Chen & Goodman 1999). The TOP order -- the order this
    instance was constructed with -- is smoothed from raw n-gram counts,
    discounted and interpolated with a lower-order distribution. Every order
    BELOW the top uses continuation ("type") counts instead of raw counts for
    both the discounted numerator and its normalizing context total:

        N_1+(* w)   = number of DISTINCT single-character left-extensions of
                      the string w that were observed (i.e. how many
                      different contexts w has been seen to continue),
        N_1+(* h *) = number of distinct (left-extension, next-character)
                      pairs observed for context h, i.e. sum_w N_1+(* h w).

    This is the defining property of Kneser-Ney smoothing: it is what avoids
    over-crediting a character sequence that is individually frequent but
    only ever occurs after one specific preceding context (the classic
    "San Francisco" problem -- "Francisco" is a common word but a poor
    standalone backoff candidate because it is not diverse in what precedes
    it). The interpolation weight's own diversity term, N_1+(context *)
    (how many distinct characters follow this context), is always computed
    from the raw n-gram table of that same order, at every level -- only the
    discounted count and its denominator switch to continuation counts below
    the top order. The order-1 (unigram) level is the continuation
    distribution itself.
    """

    def __init__(self, order: int, discount: float = 0.75):
        if order < 1:
            raise ValueError("order must be >= 1")
        if not (0.0 < discount < 1.0):
            raise ValueError("discount must lie in (0, 1)")
        self.order = int(order)
        self.discount = float(discount)
        self.counts: dict[int, Counter] = {}
        self.context_counts: dict[int, Counter] = {}
        self.unique_followers: dict[int, dict[tuple[str, ...], set[str]]] = {}
        # Continuation-count tables, used for every order BELOW the top order.
        # cont_counts[n][gram]   = N_1+(* gram), gram is a length-n tuple.
        # cont_context_totals[n][ctx] = N_1+(* ctx *), ctx is length n-1.
        self.cont_counts: dict[int, Counter] = {}
        self.cont_context_totals: dict[int, Counter] = {}
        self.vocab: set[str] = set()

    def fit(self, texts: Sequence[str]):
        texts = [str(t) for t in texts if str(t)]
        self.vocab = set("".join(texts))
        if not self.vocab:
            raise ValueError("No characters available for Kneser-Ney training.")

        # A first-order KN model still needs bigram type statistics for its
        # continuation probability. Higher-order models need all lower orders.
        max_n = max(2, self.order)
        for n in range(1, max_n + 1):
            counts = Counter()
            for text in texts:
                if len(text) < n:
                    continue
                for i in range(n - 1, len(text)):
                    counts[tuple(text[i - n + 1:i + 1])] += 1
            self.counts[n] = counts

        for n in range(2, max_n + 1):
            ctx_counts = Counter()
            followers = defaultdict(set)
            for ng, c in self.counts[n].items():
                ctx = ng[:-1]
                ctx_counts[ctx] += c
                followers[ctx].add(ng[-1])
            self.context_counts[n] = ctx_counts
            self.unique_followers[n] = dict(followers)

        # Continuation counts for every order below the top order. Each
        # DISTINCT (n+1)-gram key "p + gram" (p one extending character,
        # gram the length-n string) contributes exactly one count toward
        # N_1+(* gram) and one toward the context total N_1+(* gram[:-1] *).
        for n in range(1, max_n):
            cont = Counter()
            cont_ctx = Counter()
            for np1_gram in self.counts[n + 1]:
                gram = np1_gram[1:]
                cont[gram] += 1
                cont_ctx[gram[:-1]] += 1
            self.cont_counts[n] = cont
            self.cont_context_totals[n] = cont_ctx

        if not self.cont_counts.get(1):
            # Degenerate training stream (too little data to observe any
            # bigrams): fall back to raw unigram counts so probabilities
            # remain well-defined rather than uniformly floored.
            self.cont_counts[1] = Counter({(c,): self.counts[1][(c,)] for c in self.vocab})
        return self

    def _p_continuation(self, token: str) -> float:
        cont = self.cont_counts.get(1, Counter())
        total = sum(cont.values())
        if total <= 0:
            return 1e-12
        return max(cont.get((token,), 0) / total, 1e-12)

    def _prob_recursive(self, context: tuple[str, ...], token: str, order: int) -> float:
        if order <= 1:
            return self._p_continuation(token)
        if len(context) < order - 1:
            return self._prob_recursive(context, token, order - 1)

        ctx = context[-(order - 1):]
        is_top_order = (order == self.order)

        if is_top_order:
            c_h = self.context_counts.get(order, {}).get(ctx, 0)
            if c_h <= 0:
                return self._prob_recursive(ctx[1:], token, order - 1)
            c_hw = self.counts[order].get(ctx + (token,), 0)
        else:
            c_h = self.cont_context_totals.get(order, {}).get(ctx, 0)
            if c_h <= 0:
                return self._prob_recursive(ctx[1:], token, order - 1)
            c_hw = self.cont_counts.get(order, {}).get(ctx + (token,), 0)

        # The diversity term of the interpolation weight -- how many DISTINCT
        # characters follow this context -- always comes from the raw
        # n-gram table of this order, at every level (top or interpolated).
        followers = self.unique_followers.get(order, {}).get(ctx, set())
        discounted = max(c_hw - self.discount, 0.0) / c_h
        backoff_weight = self.discount * len(followers) / c_h
        return max(
            discounted + backoff_weight * self._prob_recursive(ctx[1:], token, order - 1),
            1e-12,
        )

    def prob(self, context: Sequence[str], token: str) -> float:
        return self._prob_recursive(tuple(context[-(self.order - 1):]), token, self.order)

    def cross_entropy_on_positions(
        self,
        heldout: Sequence[str],
        positions: Sequence[tuple[int, int]],
    ) -> float:
        losses = []
        for text_idx, pos in positions:
            text = heldout[text_idx]
            context = text[pos - self.order:pos] if self.order > 0 else ""
            token = text[pos]
            losses.append(-math.log2(self.prob(tuple(context), token)))
        if not losses:
            raise ValueError(f"No held-out targets for order {self.order}.")
        return float(np.mean(losses))


def _kn_split_utterances(texts: Sequence[str], seed: int, train_fraction: float = 0.8):
    usable = [str(t) for t in texts if _char_count(str(t)) >= 2]
    if len(usable) < 2:
        raise InsufficientDataError("Need at least two usable utterances for held-out locality estimation.")
    rng = np.random.default_rng(stable_seed("kn_split", base=int(seed)))
    perm = rng.permutation(len(usable))
    usable = [usable[int(i)] for i in perm]
    n_train = min(max(1, int(round(train_fraction * len(usable)))), len(usable) - 1)
    return usable[:n_train], usable[n_train:]


def _take_until(texts: Sequence[str], cap: int) -> list[str]:
    """Take whole utterances up to a character cap, skipping overflow items.

    Skipping an individual utterance that would cross the cap is preferable to
    stopping the scan: it avoids making the usable sample depend strongly on
    the random utterance order. When the requested cap exceeds available text,
    callers pass the available total and therefore retain the whole partition.
    """
    if cap <= 0:
        return []
    out, total = [], 0
    for t in texts:
        n = _char_count(t)
        if n <= 0:
            continue
        if total + n > cap:
            continue
        out.append(t)
        total += n
        if total >= cap:
            break
    return out


def _monotone_nonincreasing(values: Sequence[float]) -> np.ndarray:
    return np.minimum.accumulate(np.asarray(values, dtype=float))


def _common_eval_positions(heldout: Sequence[str], max_order: int, max_positions: int, seed: int):
    eligible = []
    for idx, text in enumerate(heldout):
        if len(text) <= max_order:
            continue
        # Every character, including spaces, is a possible target. This matches
        # the character-stream definition used by the locality estimator.
        for pos in range(max_order, len(text)):
            eligible.append((idx, pos))
    if not eligible:
        raise ValueError("Held-out utterances are too short for the requested locality order.")
    if len(eligible) > max_positions:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(eligible), size=max_positions, replace=False)
        chosen.sort()
        eligible = [eligible[int(i)] for i in chosen]
    return eligible


def choose_max_lag(n_characters: int, params: dict | None = None) -> tuple[int, str]:
    """Pre-specified data-availability rule applied before inspecting I_t."""
    params = params or {}
    standard = int(params.get("max_lag_chars", 20))
    threshold = int(params.get("low_resource_char_threshold", 100_000))
    low = int(params.get("low_resource_max_lag_chars", 10))
    if int(n_characters) < threshold:
        return max(1, min(low, standard)), "low_resource_fixed_lag"
    return max(1, standard), "standard_fixed_lag"


def estimate_information_locality_kn(
    language: str,
    utterance_texts: Sequence[str],
    seconds_per_character: float,
    max_lag: int = 20,
    params: dict | None = None,
    seed: int = 20260815,
) -> LocalityResult:
    """Primary locality estimator: held-out fixed-order character KN models.

    For each t, a separate order-t model is fit on training utterances and
    evaluated on the same held-out target positions. This follows the
    Hahn–Degen–Futrell held-out cross-entropy construction; the monotone
    correction is applied after estimating all S_t values.
    """
    params = dict(params or {})
    train_chars = int(params.get("kn_train_chars", params.get("train_chars", 500_000)))
    valid_chars = int(params.get("kn_valid_chars", params.get("valid_chars", 100_000)))
    eval_positions = int(params.get("kn_eval_positions", 100_000))
    discount = float(params.get("kn_discount", 0.75))
    train_fraction = float(params.get("train_fraction", 0.8))

    train, valid = _kn_split_utterances(utterance_texts, seed, train_fraction)
    train = _take_until(train, train_chars)
    valid = _take_until(valid, valid_chars)
    if not train or not valid:
        raise ValueError("Empty Kneser-Ney train/held-out partition.")

    max_order = int(max_lag) + 1
    positions = _common_eval_positions(
        valid,
        max_order=max_order,
        max_positions=eval_positions,
        seed=stable_seed(language, "kn_targets", base=seed),
    )

    S_raw = []
    for order in range(1, max_order + 1):
        model = CharacterKneserNey(order=order, discount=discount).fit(train)
        S_raw.append(model.cross_entropy_on_positions(valid, positions))

    S_corrected = _monotone_nonincreasing(S_raw)
    information = [
        float(S_corrected[t - 1] - S_corrected[t])
        for t in range(1, max_lag + 1)
    ]

    return LocalityResult(
        language=language,
        lags=list(range(1, max_lag + 1)),
        surprisal=S_corrected.tolist(),
        information=information,
        seconds_per_character=float(seconds_per_character),
    )




def _kn_split_utterances(texts: Sequence[str], seed: int, train_fraction: float = 0.8):
    usable = [str(t) for t in texts if _char_count(str(t)) >= 2]
    if len(usable) < 2:
        raise InsufficientDataError("Need at least two usable utterances for held-out locality estimation.")
    rng = np.random.default_rng(stable_seed("kn_split", base=int(seed)))
    perm = rng.permutation(len(usable))
    usable = [usable[int(i)] for i in perm]
    n_train = min(max(1, int(round(train_fraction * len(usable)))), len(usable) - 1)
    return usable[:n_train], usable[n_train:]


def _take_until(texts: Sequence[str], cap: int) -> list[str]:
    """Take whole utterances up to a character cap, skipping overflow items.

    Skipping an individual utterance that would cross the cap is preferable to
    stopping the scan: it avoids making the usable sample depend strongly on
    the random utterance order. When the requested cap exceeds available text,
    callers pass the available total and therefore retain the whole partition.
    """
    if cap <= 0:
        return []
    out, total = [], 0
    for t in texts:
        n = _char_count(t)
        if n <= 0:
            continue
        if total + n > cap:
            continue
        out.append(t)
        total += n
        if total >= cap:
            break
    return out


def _monotone_nonincreasing(values: Sequence[float]) -> np.ndarray:
    return np.minimum.accumulate(np.asarray(values, dtype=float))


def _common_eval_positions(heldout: Sequence[str], max_order: int, max_positions: int, seed: int):
    eligible = []
    for idx, text in enumerate(heldout):
        if len(text) <= max_order:
            continue
        for pos in range(max_order, len(text)):
            if text[pos].isspace():
                continue
            eligible.append((idx, pos))
    if not eligible:
        raise ValueError("Held-out utterances are too short for the requested locality order.")
    if len(eligible) > max_positions:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(eligible), size=max_positions, replace=False)
        chosen.sort()
        eligible = [eligible[int(i)] for i in chosen]
    return eligible


def choose_max_lag(n_characters: int, params: dict | None = None) -> tuple[int, str]:
    params = params or {}
    standard = int(params.get("max_lag_chars", 20))
    threshold = int(params.get("low_resource_char_threshold", 100_000))
    low = int(params.get("low_resource_max_lag_chars", 10))
    if int(n_characters) < threshold:
        return max(1, min(low, standard)), "low_resource_fixed_lag"
    return max(1, standard), "standard_fixed_lag"


def estimate_information_locality_kn(
    language: str,
    utterance_texts: Sequence[str],
    seconds_per_character: float,
    max_lag: int = 20,
    params: dict | None = None,
    seed: int = 20260815,
) -> LocalityResult:
    params = dict(params or {})
    requested_train_chars = int(params.get("kn_train_chars", params.get("train_chars", 100_000)))
    requested_valid_chars = int(params.get("kn_valid_chars", params.get("valid_chars", 20_000)))
    requested_eval_positions = int(params.get("kn_eval_positions", 25_000))
    discount = float(params.get("kn_discount", 0.75))
    train_fraction = float(params.get("train_fraction", 0.8))

    train, valid = _kn_split_utterances(utterance_texts, seed, train_fraction)
    available_train_chars = sum(_char_count(t) for t in train)
    available_valid_chars = sum(_char_count(t) for t in valid)

    # These are upper bounds, not minimum corpus requirements. FLEURS is a
    # modest per-language corpus, so every language should use all available
    # text when it falls below the requested KN caps. This keeps the estimator
    # valid without falsely rejecting languages simply because they are smaller.
    effective_train_chars = min(max(1, requested_train_chars), available_train_chars)
    effective_valid_chars = min(max(1, requested_valid_chars), available_valid_chars)
    train = _take_until(train, effective_train_chars)
    valid = _take_until(valid, effective_valid_chars)

    if not train or not valid:
        raise InsufficientDataError(
            f"Unable to construct non-empty KN train/held-out partitions "
            f"(available train={available_train_chars}, valid={available_valid_chars})."
        )

    max_order = int(max_lag) + 1
    positions = _common_eval_positions(
        valid,
        max_order=max_order,
        max_positions=requested_eval_positions,
        seed=stable_seed(language, "kn_targets", base=seed),
    )
    S_raw = []
    for order in range(1, max_order + 1):
        model = CharacterKneserNey(order=order, discount=discount).fit(train)
        S_raw.append(model.cross_entropy_on_positions(valid, positions))
    S_corrected = _monotone_nonincreasing(S_raw)
    information = [float(S_corrected[t - 1] - S_corrected[t]) for t in range(1, max_lag + 1)]
    return LocalityResult(
        language=language,
        lags=list(range(1, max_lag + 1)),
        surprisal=S_corrected.tolist(),
        information=information,
        seconds_per_character=float(seconds_per_character),
        effective_train_chars=sum(_char_count(t) for t in train),
        effective_valid_chars=sum(_char_count(t) for t in valid),
        eval_positions=len(positions),
    )


def estimate_information_locality(
    language: str,
    text: str | None = None,
    seconds_per_character: float = float("nan"),
    max_lag: int = 20,
    params: dict | None = None,
    device: str = "cpu",
    seed: int = 20260815,
    utterance_texts: Sequence[str] | None = None,
) -> LocalityResult:
    del device
    # Backward-compatible positional convenience: older callers may pass a
    # sequence as the second positional argument. Treat that as utterances,
    # not as one giant text string. Strings remain single-text inputs.
    if utterance_texts is None and text is not None and not isinstance(text, str):
        utterance_texts = [str(x) for x in text]
        text = None
    if utterance_texts is None:
        if text is None:
            raise ValueError("Provide either text or utterance_texts.")
        utterance_texts = [text]
    return estimate_information_locality_kn(language, utterance_texts, seconds_per_character,
                                            max_lag=max_lag, params=params, seed=seed)
