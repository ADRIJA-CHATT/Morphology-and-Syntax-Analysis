from __future__ import annotations

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, ElasticNetCV, RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _fit_quietly(fn):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
        try:
            from sklearn.exceptions import UndefinedMetricWarning
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        except ImportError:
            pass
        return fn()

NUMERIC = [
    "syllable_rate_sps", "word_rate_wps", "speech_char_rate_cps",
    "mean_word_length_chars", "morphology_index", "word_structure_redundancy",
    "word_order_redundancy", "char_entropy_rate_bits", "lexical_ttr",
    "mean_duration_s", "ud_morph_feature_density", "ud_mean_feats_per_token",
    "ud_dependency_direction_entropy_bits", "ud_mean_dependency_distance",
    "n_chars", "n_words", "n_utterances",
]
CAT = ["region_group", "script"]


def _add_interactions(df: pd.DataFrame, enabled: bool) -> pd.DataFrame:
    x = df.copy()
    if enabled:
        x["morphology_x_syllable_rate"] = x["morphology_index"] * x["syllable_rate_sps"]
        x["word_length_x_syllable_rate"] = x["mean_word_length_chars"] * x["syllable_rate_sps"]
        x["morphology_x_word_length"] = x["morphology_index"] * x["mean_word_length_chars"]
    return x


def _transformer(df: pd.DataFrame):
    numeric = [c for c in NUMERIC if c in df.columns]
    categorical = [c for c in CAT if c in df.columns]
    parts = [("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric)]
    if categorical:
        parts.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    return ColumnTransformer(parts, remainder="drop")


def _fit_one(df: pd.DataFrame, outcome: str, out_dir: Path, prefix: str, include_interactions: bool):
    work = df.loc[np.isfinite(df[outcome].astype(float))].copy()
    if len(work) < 8:
        raise RuntimeError(f"{prefix} regression needs at least 8 complete languages; found {len(work)}.")
    y_raw = work[outcome].astype(float).to_numpy()
    if np.any(y_raw <= 0):
        raise RuntimeError(f"{outcome} must be strictly positive before log transform.")
    y = np.log(y_raw)
    X = _add_interactions(work, include_interactions)
    transformer = _transformer(X)
    model = Pipeline([("features", transformer), ("ridge", RidgeCV(alphas=np.logspace(-4, 4, 80), cv=LeaveOneOut()))])
    pred = _fit_quietly(lambda: cross_val_predict(model, X, y, cv=LeaveOneOut()))
    _fit_quietly(lambda: model.fit(X, y))
    metrics = {
        "n_languages": int(len(work)), "outcome": outcome,
        "cv_r2_log_outcome": float(r2_score(y, pred)),
        "cv_mae_log_outcome": float(mean_absolute_error(y, pred)),
        "cv_rmse_log_outcome": float(np.sqrt(mean_squared_error(y, pred))),
        "selected_alpha": float(model.named_steps["ridge"].alpha_),
    }
    pd.DataFrame([metrics]).to_csv(out_dir / f"regression_{prefix}_metrics.csv", index=False)
    pd.DataFrame({
        "language": work["language"].values,
        "observed_outcome": y_raw,
        "predicted_outcome": np.exp(pred),
        "observed_log_outcome": y,
        "predicted_log_outcome": pred,
        "cv_residual": y - pred,
    }).to_csv(out_dir / f"regression_{prefix}_predictions.csv", index=False)
    try:
        names = model.named_steps["features"].get_feature_names_out()
        coef = model.named_steps["ridge"].coef_
        coef_df = pd.DataFrame({"feature": names, "coefficient_log_outcome": coef})
        coef_df["abs_coefficient"] = np.abs(coef_df["coefficient_log_outcome"])
        coef_df.sort_values("abs_coefficient", ascending=False).to_csv(out_dir / f"regression_{prefix}_coefficients.csv", index=False)
    except Exception:
        pass
    return metrics, X


def _sensitivity(df: pd.DataFrame, outcome: str, out_dir: Path, prefix: str, include_interactions: bool):
    work = df.loc[np.isfinite(df[outcome].astype(float))].copy()
    if len(work) < 8:
        return
    y_raw = work[outcome].astype(float).to_numpy()
    if np.any(y_raw <= 0):
        return
    y = np.log(y_raw)
    X = _add_interactions(work, include_interactions)
    transformer = _transformer(X)
    rows = []
    for name, estimator in {
        "bayesian_ridge": BayesianRidge(),
        "elastic_net": ElasticNetCV(l1_ratio=0.5, alphas=50, cv=3, max_iter=20_000),
    }.items():
        pipe = Pipeline([("features", transformer), ("model", estimator)])
        try:
            pred = _fit_quietly(lambda: cross_val_predict(pipe, X, y, cv=LeaveOneOut()))
            _fit_quietly(lambda: pipe.fit(X, y))
            rows.append({"model": name, "outcome": outcome, "n_languages": len(work), "cv_r2_log_outcome": r2_score(y, pred), "cv_mae_log_outcome": mean_absolute_error(y, pred)})
        except Exception as exc:
            rows.append({"model": name, "outcome": outcome, "n_languages": len(work), "cv_r2_log_outcome": np.nan, "cv_mae_log_outcome": np.nan, "error": str(exc)})
    pd.DataFrame(rows).to_csv(out_dir / f"regression_{prefix}_sensitivity_metrics.csv", index=False)


def fit_models(df: pd.DataFrame, out_dir: Path, include_interactions: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    char_metrics, _ = _fit_one(df, "locality_lambda_per_char", out_dir, "character", include_interactions)
    _sensitivity(df, "locality_lambda_per_char", out_dir, "character", include_interactions)
    # Do not include speech_char_rate_cps here: lambda_s = lambda_char * char/s,
    # so using char/s to predict lambda_s creates an algebraically coupled model.
    time_numeric_backup = NUMERIC[:]
    try:
        NUMERIC[:] = [c for c in NUMERIC if c != "speech_char_rate_cps"]
        time_metrics, _ = _fit_one(df, "locality_lambda_per_s", out_dir, "time_domain", include_interactions)
        _sensitivity(df, "locality_lambda_per_s", out_dir, "time_domain", include_interactions)
    finally:
        NUMERIC[:] = time_numeric_backup
    pd.DataFrame({"language": df["language"]}).to_csv(out_dir / "feature_matrix_languages.csv", index=False)
    (out_dir / "regression_model_definition.json").write_text(json.dumps({
        "primary_outcome": "log(locality_lambda_per_char)",
        "model": "RidgeCV with leave-one-language-out cross-validation",
        "sensitivity": ["BayesianRidge", "ElasticNetCV(l1_ratio=0.5)"],
        "secondary_time_outcome": "log(locality_lambda_per_s)",
        "time_domain_excludes_speech_char_rate_cps": True,
        "reason": "lambda_per_s = lambda_per_char * speech_char_rate_cps; including the latter as a predictor would introduce algebraic coupling.",
    }, indent=2), encoding="utf-8")
    return char_metrics, time_metrics
