"""
Multi-asset covariance forecasting: rolling, shrinkage, EWMA correlation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_covariance(
    returns: pd.DataFrame, window: int = 63
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rolling covariance and correlation matrices.
    Returns (cov_series dict of cov matrices by date, corr_series).
    """
    idx = returns.dropna(how="all").index
    ret = returns.reindex(idx).fillna(0)
    n_days = len(ret)
    cov_list = []
    for t in range(window, n_days):
        x = ret.iloc[t - window : t]
        cov = x.cov().values
        cov_list.append((idx[t], cov))
    dates = [c[0] for c in cov_list]
    covs = np.array([c[1] for c in cov_list])
    n_assets = ret.shape[1]
    cov_series = {}
    for i, d in enumerate(dates):
        cov_series[d] = covs[i]
    corr_series = {}
    for d in dates:
        c = cov_series[d]
        std = np.sqrt(np.diag(c))
        std = np.where(std > 1e-10, std, 1.0)
        corr = c / np.outer(std, std)
        corr_series[d] = corr
    return cov_series, corr_series


def shrinkage_covariance(
    returns: pd.DataFrame, window: int = 63, shrinkage: float = 0.1
) -> dict:
    """
    Shrinkage covariance: S_shrink = (1-delta)*S_sample + delta*F (target).
    Target F = diag(S_sample) for constant-correlation style.
    Returns cov by date.
    """
    cov_series, _ = rolling_covariance(returns, window)
    out = {}
    for d, S in cov_series.items():
        n = S.shape[0]
        F = np.diag(np.diag(S))
        S_shrink = (1 - shrinkage) * S + shrinkage * F
        out[d] = S_shrink
    return out


def ewma_covariance(
    returns: pd.DataFrame, lam: float = 0.94
) -> tuple[pd.DataFrame, dict]:
    """
    EWMA covariance: sigma^2_t = lam*sigma^2_{t-1} + (1-lam)*r_t r_t'.
    Returns (cov_ewma DataFrame of annualized vols per asset, cov_by_date dict).
    """
    ret = returns.dropna(how="all").fillna(0)
    n_assets = ret.shape[1]
    n_days = len(ret)
    cov = ret.iloc[:20].cov().values
    cov_list = []
    for t in range(20, n_days):
        r = ret.iloc[t - 1 : t].values.reshape(-1, 1)
        cov = lam * cov + (1 - lam) * (r @ r.T)
        cov_list.append((ret.index[t], cov.copy()))
    cov_by_date = {c[0]: c[1] for c in cov_list}
    vols = pd.DataFrame(
        index=[c[0] for c in cov_list],
        columns=ret.columns,
    )
    for i, (d, cov) in enumerate(cov_list):
        vols.loc[d] = np.sqrt(np.diag(cov) * 252)
    return vols, cov_by_date


def ewma_correlation_with_univariate_vols(
    returns: pd.DataFrame,
    vol_forecasts: dict[str, pd.Series],
    lam: float = 0.94,
) -> pd.DataFrame:
    """
    Build cov matrix: D_t @ R_t @ D_t where D = diag(vol_forecast), R = EWMA correlation.
    vol_forecasts: {ticker: annualized vol series}
    """
    ret = returns.dropna(how="all").fillna(0)
    tickers = list(ret.columns)
    n = len(ret)
    corr = np.eye(len(tickers))
    cov_list = []
    for t in range(50, n):
        r = ret.iloc[t - 1].values.reshape(-1, 1)
        outer = r @ r.T
        std = np.sqrt(np.diag(outer) + 1e-12)
        std = np.where(std > 1e-10, std, 1.0)
        rho = outer / np.outer(std, std)
        corr = lam * corr + (1 - lam) * rho
        vols = np.array([vol_forecasts.get(tk, pd.Series([0.15])).reindex(ret.index).ffill().iloc[t] if tk in vol_forecasts else 0.15 for tk in tickers])
        vols = np.maximum(vols / np.sqrt(252), 1e-8)
        D = np.diag(vols)
        cov_t = D @ corr @ D
        cov_list.append((ret.index[t], cov_t))
    return pd.DataFrame({d: c.flatten() for d, c in cov_list}).T


def forecast_covariance(
    returns: pd.DataFrame,
    method: str = "rolling",
    window: int = 63,
    lam: float = 0.94,
    shrinkage: float = 0.1,
) -> dict:
    """
    Forecast covariance matrix for next period.
    method: 'rolling' | 'ewma' | 'shrinkage'
    """
    if method == "rolling":
        cov_series, _ = rolling_covariance(returns, window)
        if cov_series:
            last_date = max(cov_series.keys())
            return {last_date: cov_series[last_date]}
    if method == "ewma":
        _, cov_by_date = ewma_covariance(returns, lam)
        if cov_by_date:
            last_date = max(cov_by_date.keys())
            return {last_date: cov_by_date[last_date]}
    if method == "shrinkage":
        cov_series = shrinkage_covariance(returns, window, shrinkage)
        if cov_series:
            last_date = max(cov_series.keys())
            return {last_date: cov_series[last_date]}
    return {}
