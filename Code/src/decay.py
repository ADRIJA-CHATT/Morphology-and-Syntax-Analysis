from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def _r2(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if len(y) < 2:
        return float("nan")
    sst = np.sum((y - y.mean()) ** 2)
    if sst <= 0:
        return float("nan")
    return float(1.0 - np.sum((y - pred) ** 2) / sst)


def _positive_prefix(lags, information, positive_floor):
    """Return the contiguous positive-information prefix starting at lag 1.

    The decay is only identified from the observed positive-information portion
    of the locality curve. Once I_t falls to the floor or below, later points
    are not treated as observed positive decay data and are excluded from the
    exponential fit. This avoids creating an apparently excellent fit by
    fitting an artificial tail of zeros produced by the monotonic S_t correction.
    """
    x = np.asarray(lags, dtype=float)
    y = np.asarray(information, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    order = np.argsort(x)
    x, y = x[order], y[order]

    # The theoretical locality curve starts at t=1. If the first available
    # lag is not approximately one, we do not silently re-anchor it.
    if not np.isclose(x[0], 1.0):
        return np.array([], dtype=float), np.array([], dtype=float)

    keep_x = []
    keep_y = []
    expected_lag = 1.0
    for lag, val in zip(x, y):
        if not np.isclose(lag, expected_lag):
            break
        if not np.isfinite(val) or val <= positive_floor:
            break
        keep_x.append(float(lag))
        keep_y.append(float(val))
        expected_lag += 1.0

    return np.asarray(keep_x, dtype=float), np.asarray(keep_y, dtype=float)


def _fit_exponential_positive_prefix(lags, information, seconds_per_character):
    x = np.asarray(lags, dtype=float)
    y = np.asarray(information, dtype=float)
    if len(y) < 4 or np.any(~np.isfinite(y)) or np.any(y <= 0):
        return np.nan, np.nan, np.nan, np.nan

    tau = x * float(seconds_per_character)

    def residual(theta):
        log_a, log_lam = theta
        a = np.exp(log_a)
        lam = np.exp(log_lam)
        return a * np.exp(-lam * tau) - y

    # Use the first positive point as the initial amplitude and a characteristic
    # time of one observed lag as the initial scale.
    a0 = max(float(y[0]), 1e-8)
    lam0 = 1.0 / max(float(seconds_per_character) * max(float(x[0]), 1.0), 1e-8)
    res = least_squares(
        residual,
        x0=np.array([np.log(a0), np.log(lam0)]),
        bounds=([-30.0, -20.0], [30.0, 20.0]),
        max_nfev=20000,
    )
    log_a, log_lam = map(float, res.x)
    a = float(np.exp(log_a))
    lam_sec = float(np.exp(log_lam))
    pred = a * np.exp(-lam_sec * tau)
    return lam_sec, a, _r2(y, pred), float(np.sqrt(np.mean((y - pred) ** 2)))


def _hilberg_fit(lags, information):
    x = np.asarray(lags, dtype=float)
    y = np.asarray(information, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = x[mask], y[mask]
    if len(y) < 4:
        return np.nan, np.nan, np.nan, False

    def basis(alpha):
        return np.maximum(x ** (-alpha) - (x + 1.0) ** (-alpha), 1e-300)

    def residual(theta):
        alpha, log_a = theta
        return log_a + np.log(basis(alpha)) - np.log(y)

    try:
        res = least_squares(
            residual,
            x0=np.array([0.5, np.log(max(y[0], 1e-12))]),
            bounds=([0.001, -50.0], [5.0, 50.0]),
            max_nfev=20000,
        )
        alpha, log_a = map(float, res.x)
        amp = float(np.exp(log_a))
        pred = amp * basis(alpha)
        at_bound = bool(abs(alpha - 0.001) < 1e-6 or abs(alpha - 5.0) < 1e-6)
        return alpha, amp, _r2(y, pred), at_bound
    except Exception:
        return np.nan, np.nan, np.nan, False


def fit_decay(
    lags,
    information,
    seconds_per_character: float,
    positive_floor: float = 1e-5,
    min_decay_points: int = 4,
):
    """Fit locality decay only on the observed contiguous positive prefix.

    A decay parameter is identified only when at least `min_decay_points`
    consecutive positive I_t values occur starting at t=1. Non-positive or
    monotonic-correction-zero tail values are not treated as additional decay
    observations.
    """
    x_pref, y_pref = _positive_prefix(lags, information, positive_floor)
    n_positive = int(len(y_pref))

    base = {
        "locality_lambda_per_char": np.nan,
        "locality_lambda_per_s": np.nan,
        "locality_quality_status": "insufficient_positive_information",
        "locality_lambda_r2": np.nan,
        "locality_fit_rmse_bits": np.nan,
        "hilberg_alpha": np.nan,
        "hilberg_r2": np.nan,
        "hilberg_alpha_at_bound": False,
        "locality_amplitude": np.nan,
        "n_decay_points": n_positive,
        "n_positive_information_points": n_positive,
        "decay_start_lag": np.nan if n_positive == 0 else float(x_pref[0]),
        "decay_end_lag": np.nan if n_positive == 0 else float(x_pref[-1]),
        "decay_start_seconds": np.nan if n_positive == 0 else float(x_pref[0] * seconds_per_character),
        "decay_end_seconds": np.nan if n_positive == 0 else float(x_pref[-1] * seconds_per_character),
        "decay_fit_method": "insufficient_positive_information",
    }

    if n_positive < int(min_decay_points):
        if n_positive > 0:
            base["decay_fit_method"] = "insufficient_positive_prefix_points"
        return base

    lam_sec, amp, r2, rmse = _fit_exponential_positive_prefix(
        x_pref, y_pref, seconds_per_character
    )
    alpha, _, h_r2, at_bound = _hilberg_fit(x_pref, y_pref)
    lam_char = lam_sec * float(seconds_per_character) if np.isfinite(lam_sec) else np.nan

    base.update({
        "locality_lambda_per_char": lam_char,
        "locality_lambda_per_s": lam_sec,
        "locality_quality_status": "identified" if np.isfinite(lam_sec) and lam_sec > 0 else "unidentified",
        "locality_lambda_r2": r2,
        "locality_fit_rmse_bits": rmse,
        "hilberg_alpha": alpha,
        "hilberg_r2": h_r2,
        "hilberg_alpha_at_bound": at_bound,
        "locality_amplitude": amp,
        "decay_fit_method": "nonlinear_exponential_least_squares_positive_prefix",
    })
    return base


def bootstrap_lambda(
    lags,
    information,
    seconds_per_character: float,
    reps: int = 300,
    seed: int = 20260815,
    positive_floor: float = 1e-5,
    min_decay_points: int = 4,
):
    """Residual bootstrap using exactly the same positive prefix as fit_decay."""
    x, y = _positive_prefix(lags, information, positive_floor)
    if len(y) < int(min_decay_points):
        return {
            "lambda_q025": np.nan,
            "lambda_q975": np.nan,
            "lambda_bootstrap_n": 0,
            "lambda_bootstrap_method": "not_identified",
        }

    lam_sec, amp, _, _ = _fit_exponential_positive_prefix(x, y, seconds_per_character)
    if not np.isfinite(lam_sec):
        return {
            "lambda_q025": np.nan,
            "lambda_q975": np.nan,
            "lambda_bootstrap_n": 0,
            "lambda_bootstrap_method": "not_identified",
        }

    tau = x * float(seconds_per_character)
    fitted = amp * np.exp(-lam_sec * tau)
    residuals = y - fitted
    rng = np.random.default_rng(seed)
    lambdas = []

    for _ in range(int(reps)):
        e = rng.choice(residuals, size=len(residuals), replace=True)
        yb = fitted + e
        # Do not manufacture positive information by clipping an entire
        # bootstrap replicate. Retain only replicates whose complete original
        # prefix remains above the positive floor.
        if np.any(yb <= positive_floor):
            continue
        try:
            bb, _, _, _ = _fit_exponential_positive_prefix(x, yb, seconds_per_character)
            if np.isfinite(bb) and bb > 0:
                lambdas.append(bb)
        except Exception:
            continue

    if len(lambdas) < max(50, int(reps) // 10):
        return {
            "lambda_q025": np.nan,
            "lambda_q975": np.nan,
            "lambda_bootstrap_n": int(len(lambdas)),
            "lambda_bootstrap_method": "parametric_residual_positive_prefix",
        }

    q025, q975 = np.quantile(lambdas, [0.025, 0.975])
    return {
        "lambda_q025": float(q025),
        "lambda_q975": float(q975),
        "lambda_bootstrap_n": int(len(lambdas)),
        "lambda_bootstrap_method": "parametric_residual_positive_prefix",
    }


def memory_cost(lags, information):
    x = np.asarray(lags, dtype=float)
    y = np.asarray(information, dtype=float)
    y = np.where(np.isfinite(y) & (y > 0), y, 0.0)
    return float(np.sum(x * y))
