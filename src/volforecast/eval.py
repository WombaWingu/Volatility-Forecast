"""
Volatility forecast evaluation: loss functions (QLIKE, MSE on variance, log),
Diebold–Mariano test, prediction intervals, regime-conditioned metrics.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def loss_mae(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean absolute error on volatility."""
    return float(np.nanmean(np.abs(yhat - y)))


def loss_rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    """Root mean squared error on volatility."""
    return float(np.sqrt(np.nanmean((yhat - y) ** 2)))


def loss_qlike(y: np.ndarray, yhat: np.ndarray) -> float:
    """
    Quasi-likelihood (QLIKE) for volatility: (y²/σ²) - log(y²/σ²) - 1.
    Common in volatility forecasting; scale-invariant.
    Skips observations where y² or σ² is non-positive to avoid log(0).
    """
    y2 = np.asarray(y, dtype=float) ** 2
    s2 = np.asarray(yhat, dtype=float) ** 2
    eps = 1e-12
    s2 = np.maximum(s2, eps)
    # Only use observations where both variances are positive
    mask = (y2 >= eps) & (s2 >= eps)
    if not np.any(mask):
        return np.nan
    ratio = np.where(mask, y2 / s2, np.nan)
    q = ratio - np.log(ratio) - 1
    return float(np.nanmean(q))


def loss_mse_variance(y: np.ndarray, yhat: np.ndarray) -> float:
    """MSE on variance (forecast σ² vs realized σ²)."""
    y2 = np.asarray(y, dtype=float) ** 2
    s2 = np.asarray(yhat, dtype=float) ** 2
    return float(np.nanmean((s2 - y2) ** 2))


def loss_log_vol(y: np.ndarray, yhat: np.ndarray) -> float:
    """Log error on volatility: mean((log(σ_hat) - log(σ))²)."""
    y = np.maximum(np.asarray(y, dtype=float), 1e-12)
    yhat = np.maximum(np.asarray(yhat, dtype=float), 1e-12)
    return float(np.nanmean((np.log(yhat) - np.log(y)) ** 2))


def volatility_metrics_full(
    y: np.ndarray,
    yhat: np.ndarray,
) -> dict[str, float]:
    """MAE, RMSE, Correlation, QLIKE, MSE_var, Log_vol."""
    err = yhat - y
    y2 = np.maximum(y ** 2, 1e-12)
    s2 = np.maximum(yhat ** 2, 1e-12)
    corr = np.nan
    if np.nanstd(yhat) > 0 and np.nanstd(y) > 0:
        corr = float(np.corrcoef(y, yhat)[0, 1])
    return {
        "MAE": loss_mae(y, yhat),
        "RMSE": loss_rmse(y, yhat),
        "Correlation": corr,
        "QLIKE": loss_qlike(y, yhat),
        "MSE_var": loss_mse_variance(y, yhat),
        "Log_vol": loss_log_vol(y, yhat),
    }


def diebold_mariano(
    e1: np.ndarray,
    e2: np.ndarray,
    loss: str = "squared",
    h: int = 1,
) -> tuple[float, float]:
    """
    Diebold–Mariano test: H0 = equal predictive accuracy.
    e1, e2: forecast errors (y - yhat) for model 1 and 2.
    loss: 'squared' (d = e1² - e2²) or 'absolute' (d = |e1| - |e2|).
    h: forecast horizon for Harvey correction.
    Returns (dm_stat, p_value two-sided).
    """
    if loss == "squared":
        d = (e1 ** 2) - (e2 ** 2)
    else:
        d = np.abs(e1) - np.abs(e2)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return np.nan, np.nan
    d_bar = np.mean(d)
    # HAC variance: Newey-West style with lag ~ min(n-1, h)
    gamma_0 = np.var(d, ddof=0)
    if gamma_0 <= 0:
        return np.nan, np.nan
    # Bartlett kernel, max_lag = h
    max_lag = min(n - 1, h)
    for k in range(1, max_lag + 1):
        gamma_k = np.mean((d[:-k] - d_bar) * (d[k:] - d_bar))
        gamma_0 += 2 * (1 - k / (max_lag + 1)) * gamma_k
    var_d = gamma_0 / n
    if var_d <= 0:
        return np.nan, np.nan
    dm = d_bar / np.sqrt(var_d)
    from scipy import stats
    p = 2 * (1 - stats.norm.cdf(abs(dm)))
    return float(dm), float(p)


def prediction_interval_bootstrap(
    y: np.ndarray,
    yhat: np.ndarray,
    residuals: np.ndarray | None = None,
    n_bootstrap: int = 500,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap prediction intervals: resample (y - yhat), add to yhat.
    Returns (lower, upper) arrays of length len(y).
    """
    if residuals is None:
        residuals = y - yhat
    res = residuals[np.isfinite(residuals)]
    n = len(y)
    lower = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    rng = np.random.default_rng(42)
    for i in range(n):
        if not np.isfinite(yhat[i]):
            continue
        boot = yhat[i] + rng.choice(res, size=n_bootstrap, replace=True)
        boot = np.maximum(boot, 1e-8)
        lower[i] = np.quantile(boot, alpha / 2)
        upper[i] = np.quantile(boot, 1 - alpha / 2)
    return lower, upper


def metrics_by_regime(
    y: np.ndarray,
    yhat: np.ndarray,
    regime: np.ndarray | pd.Series,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Compute MAE/RMSE/QLIKE by regime (e.g. low/medium/high vol)."""
    if metrics is None:
        metrics = ["MAE", "RMSE", "QLIKE"]
    regime = np.asarray(regime)
    results = []
    for r in np.unique(regime):
        mask = (regime == r) & np.isfinite(y) & np.isfinite(yhat)
        if mask.sum() < 10:
            continue
        yy, yyh = y[mask], yhat[mask]
        row = {"regime": r}
        if "MAE" in metrics:
            row["MAE"] = loss_mae(yy, yyh)
        if "RMSE" in metrics:
            row["RMSE"] = loss_rmse(yy, yyh)
        if "QLIKE" in metrics:
            row["QLIKE"] = loss_qlike(yy, yyh)
        results.append(row)
    if not results:
        return pd.DataFrame(columns=["regime"] + metrics).set_index("regime")
    return pd.DataFrame(results).set_index("regime")


def residual_diagnostics(
    residuals: np.ndarray,
    max_lag: int = 20,
) -> dict[str, float | np.ndarray]:
    """Autocorrelation of errors and Ljung–Box style check."""
    res = residuals[np.isfinite(residuals)]
    if len(res) < max_lag + 5:
        return {"acf": np.array([]), "lb_pvalue": np.nan}
    acf = np.array([np.corrcoef(res[:-k], res[k:])[0, 1] for k in range(1, max_lag + 1)])
    # Ljung–Box approx: Q = n(n+2) sum(ρ²/(n-k))
    n = len(res)
    q = n * (n + 2) * np.sum(acf ** 2 / (n - np.arange(1, max_lag + 1)))
    from scipy import stats
    lb_pvalue = 1 - stats.chi2.cdf(q, max_lag)
    return {"acf": acf, "lb_pvalue": float(lb_pvalue)}
