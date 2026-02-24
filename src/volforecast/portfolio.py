"""
Portfolio construction: risk parity, minimum variance, volatility-controlled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import multicov as vmc


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


def compute_target_shares(
    equity: float,
    close: float,
    sigma: float,
    target_vol_ann: float = 0.10,
    cap: float = 1.0,
    floor: float = 0.05,
    integer_only: bool = True,
) -> dict:
    """
    Compute target shares for volatility targeting.
    
    Args:
        equity: Account equity in dollars
        close: Current close price
        sigma: Forecast annualized volatility
        target_vol_ann: Target annualized volatility (default 0.10 = 10%)
        cap: Maximum exposure multiplier (default 1.0 = no leverage)
        floor: Minimum volatility floor to prevent extreme leverage (default 0.05)
        integer_only: If True, round shares to integer
    
    Returns:
        dict with keys: shares, exposure_multiplier, dollar_exposure, forecast_vol_ann
    """
    exposure_mult = min(cap, target_vol_ann / max(sigma, floor))
    desired_exposure = equity * exposure_mult
    desired_shares = desired_exposure / close
    
    if integer_only:
        desired_shares = int(round(desired_shares))
        # Recalculate exposure based on integer shares
        desired_exposure = desired_shares * close
        exposure_mult = desired_exposure / equity if equity > 0 else 0.0
    
    return {
        "shares": desired_shares,
        "exposure_multiplier": exposure_mult,
        "dollar_exposure": desired_exposure,
        "forecast_vol_ann": sigma,
        "target_vol_ann": target_vol_ann,
        "cap": cap,
        "floor": floor,
    }


def generate_tomorrow_positions(
    signals_df: pd.DataFrame,
    model_used: str,
    equity: float,
    target_vol_ann: float = 0.10,
    cap: float = 1.0,
    floor: float = 0.05,
    integer_only: bool = True,
) -> pd.DataFrame:
    """
    Generate tomorrow's positions from latest signals.
    
    Args:
        signals_df: DataFrame with Date index and columns: Close, Simple, and model forecasts
        model_used: Model name to use (e.g., 'ridge', 'ewma', 'garch')
        equity: Account equity in dollars
        target_vol_ann: Target annualized volatility
        cap: Maximum exposure multiplier
        floor: Minimum volatility floor
        integer_only: Round shares to integer
    
    Returns:
        DataFrame with columns: signal_date, trade_date, ticker, model_used, forecast_vol_ann,
        target_vol_ann, cap, floor, close, desired_shares_int, dollar_exposure, exposure_multiplier
    """
    from datetime import date, timedelta
    
    # Get latest signal
    last_signal = signals_df.dropna(subset=[model_used]).iloc[-1]
    signal_date = pd.Timestamp(last_signal.name) if hasattr(last_signal, 'name') else signals_df.index[-1]
    
    if isinstance(signal_date, pd.Timestamp):
        trade_date = signal_date + timedelta(days=1)
    else:
        trade_date = pd.Timestamp(signal_date) + timedelta(days=1)
    
    sigma = float(last_signal[model_used])
    close_price = float(last_signal["Close"])
    
    result = compute_target_shares(
        equity=equity,
        close=close_price,
        sigma=sigma,
        target_vol_ann=target_vol_ann,
        cap=cap,
        floor=floor,
        integer_only=integer_only,
    )
    
    return pd.DataFrame([{
        "signal_date": signal_date,
        "trade_date": trade_date,
        "ticker": "UNKNOWN",  # Will be filled by caller
        "model_used": model_used,
        "forecast_vol_ann": result["forecast_vol_ann"],
        "target_vol_ann": result["target_vol_ann"],
        "cap": result["cap"],
        "floor": result["floor"],
        "close": close_price,
        "desired_shares_int": result["shares"],
        "dollar_exposure": result["dollar_exposure"],
        "exposure_multiplier": result["exposure_multiplier"],
    }])
