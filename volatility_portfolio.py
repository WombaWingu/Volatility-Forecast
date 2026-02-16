"""
Portfolio construction: risk parity, minimum variance, volatility-controlled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import volatility_multicov as vmc


def risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    """
    Risk parity: weights proportional to inverse of marginal risk.
    Solves for w such that w_i * (Sigma w)_i is equal across assets.
    Simple iterative scheme.
    """
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(50):
        vol = np.sqrt(w @ cov @ w)
        if vol < 1e-12:
            break
        marginal = cov @ w / vol
        inv_marg = 1.0 / np.maximum(marginal, 1e-10)
        w = inv_marg / inv_marg.sum()
    return w / w.sum()


def min_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Minimum variance portfolio: w = Sigma^{-1} 1 / (1' Sigma^{-1} 1)."""
    try:
        inv = np.linalg.inv(cov + 1e-8 * np.eye(cov.shape[0]))
        ones = np.ones(cov.shape[0])
        w = inv @ ones / (ones @ inv @ ones)
        return np.maximum(w, 0) / np.maximum(w, 0).sum()
    except Exception:
        return np.ones(cov.shape[0]) / cov.shape[0]


def vol_target_portfolio_weights(
    cov: np.ndarray, target_vol_ann: float = 0.10
) -> np.ndarray:
    """
    Scale min-var or risk-parity to target volatility.
    """
    w = min_variance_weights(cov)
    vol = np.sqrt(w @ cov @ w)
    if vol < 1e-12:
        return w
    scale = target_vol_ann / np.sqrt(252) / vol
    return np.clip(w * scale, 0, 3.0 / cov.shape[0])


def portfolio_backtest(
    returns: pd.DataFrame,
    weight_func,
    cov_func,
    window: int = 63,
) -> pd.Series:
    """
    Backtest portfolio: rolling cov -> weights -> portfolio returns.
    weight_func: (cov) -> weights
    cov_func: (returns, window) -> cov_by_date dict
    Uses weights from end of day t-1 for return on day t.
    """
    cov_series = cov_func(returns, window)
    dates = sorted(cov_series.keys())
    if len(dates) < 2:
        return pd.Series(dtype=float)
    port_ret = []
    for i in range(1, len(dates)):
        d_prev, d_curr = dates[i - 1], dates[i]
        cov = cov_series.get(d_prev)
        if cov is None:
            continue
        w = weight_func(cov)
        if d_curr not in returns.index:
            continue
        r = returns.loc[d_curr].fillna(0).values
        if len(r) != len(w):
            continue
        port_ret.append((d_curr, float(w @ r)))
    if not port_ret:
        return pd.Series(dtype=float)
    return pd.Series({d: v for d, v in port_ret})


def _rolling_cov_series(ret: pd.DataFrame, win: int) -> dict:
    cs, _ = vmc.rolling_covariance(ret, win)
    return cs


def risk_parity_backtest(returns: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Risk parity backtest using rolling covariance."""
    return portfolio_backtest(returns, risk_parity_weights, _rolling_cov_series, window)


def min_variance_backtest(returns: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Minimum variance backtest."""
    return portfolio_backtest(returns, min_variance_weights, _rolling_cov_series, window)
