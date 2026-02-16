"""
Volatility data and estimators: range-based (Parkinson, Garman–Klass, Rogers–Satchell),
realized vol variants, and forward targets for multiple horizons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def parkinson_vol(high: pd.Series, low: pd.Series) -> pd.Series:
    """Parkinson volatility from high-low range: σ = sqrt(ln(H/L)² / (4*ln(2)))."""
    log_hl = np.log(high / low.replace(0, np.nan))
    var_p = (log_hl ** 2) / (4 * np.log(2))
    return np.sqrt(var_p)


def garman_klass_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Garman–Klass volatility (OHLC): more efficient than close-to-close."""
    log_hl = np.log(high / low.replace(0, np.nan))
    log_co = np.log(close / open_.replace(0, np.nan))
    # σ² = 0.5*(ln(H/L))² - (2*ln(2)-1)/2 * (ln(C/O))²
    var_gk = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) / 2 * (log_co ** 2)
    var_gk = var_gk.clip(lower=1e-12)
    return np.sqrt(var_gk)


def rogers_satchell_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """Rogers–Satchell volatility (OHLC, allows drift)."""
    log_hc = np.log(high / close.replace(0, np.nan))
    log_ho = np.log(high / open_.replace(0, np.nan))
    log_lc = np.log(low / close.replace(0, np.nan))
    log_lo = np.log(low / open_.replace(0, np.nan))
    rs = log_hc * log_ho + log_lc * log_lo
    rs = rs.clip(lower=1e-12)
    return np.sqrt(rs)


def realized_vol_close(returns: pd.Series, window: int) -> pd.Series:
    """Realized volatility from close-to-close returns: sqrt(sum of squared returns)."""
    return np.sqrt((returns ** 2).rolling(window).sum())


def realized_var_close(returns: pd.Series, window: int) -> pd.Series:
    """Realized variance (sum of squared returns) over window."""
    return (returns ** 2).rolling(window).sum()


def range_based_vol_rolling(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 5,
    method: str = "gk",
) -> pd.Series:
    """
    Rolling range-based volatility over `window` days.
    method: 'parkinson' | 'gk' | 'rs'. Returns daily vol (per day).
    """
    if method == "parkinson":
        daily = parkinson_vol(high, low)
    elif method == "gk":
        daily = garman_klass_vol(open_, high, low, close)
    elif method == "rs":
        daily = rogers_satchell_vol(open_, high, low, close)
    else:
        raise ValueError("method must be 'parkinson', 'gk', or 'rs'")
    return np.sqrt((daily ** 2).rolling(window).mean())


def forward_realized_vol(
    returns: pd.Series,
    horizon_days: int,
    trading_days: int = 252,
) -> pd.Series:
    """
    Forward realized vol over next `horizon_days`: sqrt(sum of squared returns)
    then annualized with sqrt(trading_days / horizon_days).
    """
    fwd_sq = (returns ** 2).rolling(horizon_days).sum().shift(-horizon_days)
    fwd_vol = np.sqrt(fwd_sq)
    ann = np.sqrt(trading_days / horizon_days)
    return (fwd_vol * ann).rename(f"rv_fwd_{horizon_days}d")


def har_rv_features(
    rv_daily: pd.Series,
    rv_weekly: pd.Series | None = None,
    rv_monthly: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Build HAR-style features: daily RV, 5d (weekly) avg RV, 22d (monthly) avg RV.
    If rv_weekly/rv_monthly not provided, they are computed from rv_daily as
    rolling means of past 5 and 22 days of rv_daily (variance then take mean and sqrt for vol).
    Standard HAR uses realized VARIANCE; we use vol and work in vol space (HAR-RV in vol).
    """
    if rv_weekly is None:
        # Past 5-day mean of squared daily vol -> then sqrt
        rv_weekly = np.sqrt((rv_daily ** 2).rolling(5).mean())
    if rv_monthly is None:
        rv_monthly = np.sqrt((rv_daily ** 2).rolling(22).mean())
    return pd.DataFrame(
        {
            "rv_d": rv_daily,
            "rv_w": rv_weekly,
            "rv_m": rv_monthly,
        },
        index=rv_daily.index,
    )
