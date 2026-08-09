from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# -----------------------------
# Helpers
# -----------------------------
PUNCT_RE = re.compile(r"^\W+$", re.UNICODE)


def infer_language_from_filename(path: Path) -> str:
    """
    Assumes files are named like:
        English.annotated.tsv
        RussianCorpus.annotated.tsv
        fr.annotated.tsv

    We take the part before the first dot or underscore.
    """
    stem = path.stem
    first = re.split(r"[._-]", stem)[0]
    return first.strip()


def is_punctuation(x: str) -> bool:
    if x is None:
        return True
    s = str(x).strip()
    return s == "" or bool(PUNCT_RE.match(s))


def safe_lower(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def phonological_unit_count(phonemes: str, surface: str) -> int:
    """
    Counts phonological units using whitespace-delimited phoneme output.
    Falls back to surface character length if phonemes are missing.
    """
    p = safe_lower(phonemes)
    if p and p != "-":
        units = [u for u in p.split() if u]
        if units:
            return len(units)

    s = str(surface).strip()
    return max(1, len(s))


def shannon_entropy_from_counts(counts: Counter) -> float:
    """
    Shannon entropy in bits:
        H = - sum p(x) log2 p(x)
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array(list(counts.values()), dtype=float) / float(total)
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def zscore(series: pd.Series) -> pd.Series:
    mu = series.mean()
    sd = series.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return series * 0.0
    return (series - mu) / sd


# -----------------------------
# Load annotated corpora
# -----------------------------
def load_annotated_tsvs(input_dir: str | Path) -> pd.DataFrame:
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("*.tsv"))
    if not files:
        raise FileNotFoundError(f"No .tsv files found in {input_dir}")

    frames = []
    for fp in files:
        df = pd.read_csv(fp, sep="\t")
        df["language"] = infer_language_from_filename(fp)
        df["source_file"] = fp.name
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)

    required = {"surface", "lemma", "upos", "feats", "phonemes", "syllables", "start_sec", "end_sec"}
    missing = required - set(all_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "sentence_id" not in all_df.columns:
        all_df["sentence_id"] = 1

    return all_df


# -----------------------------
# Language-level predictors
# -----------------------------
def compute_language_predictors(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for lang, g in df.groupby("language"):
        g = g.copy()

        # Remove punctuation for most measures
        token_mask = ~g["surface"].astype(str).map(is_punctuation)
        tok = g.loc[token_mask].copy()

        if tok.empty:
            continue

        # Speech rate: syllables / duration
        duration = float(g["end_sec"].max() - g["start_sec"].min())
        duration = max(duration, 1e-9)
        total_syllables = float(pd.to_numeric(tok["syllables"], errors="coerce").fillna(0).sum())
        speech_rate = total_syllables / duration

        # Average word length:
        # use phonological unit count if phonemes exist; otherwise surface character length
        token_lengths = tok.apply(
            lambda r: phonological_unit_count(r["phonemes"], r["surface"]),
            axis=1
        )
        avg_word_length = float(np.mean(token_lengths))

        # Word-order entropy proxy:
        # POS trigram entropy over sentence-separated token sequences
        pos_trigram_counts = Counter()
        for sid, sent in tok.groupby("sentence_id"):
            pos_seq = [str(x) for x in sent.sort_values("token_id")["upos"].fillna("UNK").tolist()]
            if len(pos_seq) < 3:
                continue
            for i in range(len(pos_seq) - 2):
                trigram = tuple(pos_seq[i:i+3])
                pos_trigram_counts[trigram] += 1
        h_order = shannon_entropy_from_counts(pos_trigram_counts)

        # Morphological entropy:
        # entropy of morphological feature bundles
        feats = tok["feats"].fillna("NONE").astype(str)
        feats = feats.replace({"": "NONE", "-": "NONE"})
        feat_counts = Counter(feats.tolist())
        h_morph = shannon_entropy_from_counts(feat_counts)

        eps = 1e-12
        morph_richness_share = h_morph / (h_order + h_morph + eps)
        morph_richness_logratio = math.log((h_morph + eps) / (h_order + eps))

        rows.append(
            {
                "language": lang,
                "speech_rate": speech_rate,
                "avg_word_length": avg_word_length,
                "H_order": h_order,
                "H_morph": h_morph,
                "morph_richness_share": morph_richness_share,
                "morph_richness_logratio": morph_richness_logratio,
                "n_tokens": int(len(tok)),
                "duration_sec": duration,
            }
        )

    return pd.DataFrame(rows)


# -----------------------------
# Information locality decay I_t
# -----------------------------
def tokenize_sequence_for_it(df_lang: pd.DataFrame, use: str = "lemma") -> List[List[str]]:
    """
    Converts one language corpus into sentence-separated token sequences.
    use = "lemma" or "surface"
    """
    if use not in {"lemma", "surface"}:
        raise ValueError("use must be 'lemma' or 'surface'")

    seqs = []
    for sid, sent in df_lang.groupby("sentence_id"):
        sent = sent.sort_values("token_id")
        toks = []
        for _, r in sent.iterrows():
            surf = str(r["surface"])
            if is_punctuation(surf):
                continue
            token = safe_lower(r[use])
            if not token or token == "-":
                token = safe_lower(r["surface"])
            token = token.strip()
            if token:
                toks.append(token)
        if toks:
            seqs.append(toks)
    return seqs


def conditional_entropy_ngram(seqs: List[List[str]], order: int, alpha: float = 0.1) -> float:
    """
    Estimates S_t = H(W_i | W_{i-order}, ..., W_{i-1})
    using smoothed n-gram probabilities.

    Formula:
        p(w | c) = (count(c,w) + alpha) / (count(c) + alpha * |V|)
        S_t = - (1/N) sum log2 p(w_i | context)
    """
    if order < 1:
        raise ValueError("order must be >= 1")

    context_counts = Counter()
    context_token_counts = Counter()
    vocab = set()

    for sent in seqs:
        padded = ["<BOS>"] * order + sent + ["<EOS>"]
        vocab.update(sent)
        vocab.add("<EOS>")
        for i in range(order, len(padded)):
            ctx = tuple(padded[i - order:i])
            w = padded[i]
            context_counts[ctx] += 1
            context_token_counts[(ctx, w)] += 1

    V = max(1, len(vocab))
    total = sum(context_token_counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for (ctx, w), c in context_token_counts.items():
        p = (c + alpha) / (context_counts[ctx] + alpha * V)
        entropy -= c * math.log2(p)

    return entropy / total


def compute_it_curve(df_lang: pd.DataFrame, max_order: int = 6, use: str = "lemma") -> pd.DataFrame:
    """
    Returns a table with columns:
        language, context_order, S_t, I_t

    where I_t = S_t - S_{t+1}
    """
    lang = str(df_lang["language"].iloc[0])
    seqs = tokenize_sequence_for_it(df_lang, use=use)

    # Need S_1 ... S_(max_order+1) to compute I_1 ... I_max_order
    S = {}
    for t in range(1, max_order + 2):
        S[t] = conditional_entropy_ngram(seqs, order=t, alpha=0.1)

    rows = []
    for t in range(1, max_order + 1):
        rows.append(
            {
                "language": lang,
                "context_order": t,
                "S_t": S[t],
                "I_t": S[t] - S[t + 1],
            }
        )

    return pd.DataFrame(rows)


# -----------------------------
# Build regression dataset
# -----------------------------
def build_model_table(input_dir: str | Path, max_order: int = 6, use: str = "lemma") -> pd.DataFrame:
    df = load_annotated_tsvs(input_dir)
    pred = compute_language_predictors(df)

    curves = []
    for lang, g in df.groupby("language"):
        curves.append(compute_it_curve(g, max_order=max_order, use=use))
    it_df = pd.concat(curves, ignore_index=True)

    model_df = it_df.merge(pred, on="language", how="inner")

    # Standardize predictors for a cleaner linear model
    model_df["speech_rate_z"] = zscore(model_df["speech_rate"])
    model_df["avg_word_length_z"] = zscore(model_df["avg_word_length"])
    model_df["morph_richness_z"] = zscore(model_df["morph_richness_logratio"])
    model_df["context_order_z"] = zscore(model_df["context_order"])

    return model_df


# -----------------------------
# Fit linear model
# -----------------------------
def fit_linear_model(model_df: pd.DataFrame):
    """
    Linear model:
        I_t ~ context_order_z
              + speech_rate_z
              + avg_word_length_z
              + morph_richness_z
              + context_order_z:speech_rate_z
              + context_order_z:avg_word_length_z
              + context_order_z:morph_richness_z
    """
    formula = (
        "I_t ~ context_order_z "
        "+ speech_rate_z "
        "+ avg_word_length_z "
        "+ morph_richness_z "
        "+ context_order_z:speech_rate_z "
        "+ context_order_z:avg_word_length_z "
        "+ context_order_z:morph_richness_z"
    )

    model = smf.ols(formula=formula, data=model_df).fit(
        cov_type="cluster",
        cov_kwds={"groups": model_df["language"]},
    )
    return model


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a linear model for information locality decay from annotated corpora."
    )
    parser.add_argument("--input-dir", required=True, help="Folder containing *.annotated.tsv files")
    parser.add_argument("--max-order", type=int, default=6, help="Maximum context order for I_t")
    parser.add_argument(
        "--sequence",
        choices=["lemma", "surface"],
        default="lemma",
        help="Token sequence used for I_t estimation",
    )
    parser.add_argument(
        "--out-prefix",
        default="it_model",
        help="Prefix for output CSV files",
    )

    args = parser.parse_args()

    model_df = build_model_table(args.input_dir, max_order=args.max_order, use=args.sequence)
    model = fit_linear_model(model_df)

    # Save outputs
    model_df.to_csv(f"{args.out_prefix}_data.csv", index=False)

    with open(f"{args.out_prefix}_summary.txt", "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())

    print(model.summary())
    print(f"\nSaved: {args.out_prefix}_data.csv")
    print(f"Saved: {args.out_prefix}_summary.txt")