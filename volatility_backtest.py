"""
Volatility targeting backtest and risk metrics: VaR, ES, Kupiec test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def volatility_targeting_backtest(
    returns: pd.Series,
    vol_forecast: pd.Series,
    target_vol_ann: float = 0.10,
    trading_days: int = 252,
    max_leverage: float = 3.0,
    min_leverage: float = 0.0,
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> pd.DataFrame:
    """
    Scale exposure to target vol using forecast: weight_t = target_vol / vol_forecast_t.
    Optionally apply transaction costs (bps) and slippage, and leverage constraints.
    Returns DataFrame with columns: scaled_returns, leverage, realized_vol_ann, cum_return, drawdown,
    turnover, costs, net_returns.
    """
    vol_forecast = vol_forecast.reindex(returns.index).ffill().bfill()
    vol_forecast = vol_forecast.clip(lower=1e-8)
    target_daily = target_vol_ann / np.sqrt(trading_days)
    leverage = target_daily / (vol_forecast / np.sqrt(trading_days))
    leverage = leverage.clip(upper=max_leverage, lower=min_leverage)
    lev_shift = leverage.shift(1)
    scaled_returns = returns * lev_shift
    scaled_returns = scaled_returns.dropna()
    turnover = lev_shift.diff().abs().reindex(scaled_returns.index).fillna(0)
    costs = (turnover * cost_bps / 1e4) + (scaled_returns.abs() * slippage_bps / 1e4)
    net_returns = scaled_returns - costs.reindex(scaled_returns.index).fillna(0)
    cum = (1 + net_returns).cumprod()
    dd = cum / cum.cummax() - 1
    rv = np.sqrt((scaled_returns ** 2).rolling(22).sum() / 22) * np.sqrt(trading_days)
    df = pd.DataFrame({
        "scaled_returns": scaled_returns,
        "net_returns": net_returns,
        "leverage": leverage.reindex(scaled_returns.index).ffill(),
        "realized_vol_ann": rv,
        "cum_return": cum,
        "drawdown": dd,
        "turnover": turnover,
        "costs": costs,
    })
    return df


def backtest_summary(bt: pd.DataFrame, risk_free: float = 0.0) -> dict[str, float]:
    """Sharpe, max drawdown, mean return, turnover, realized vol error."""
    ret = bt["net_returns"].dropna() if "net_returns" in bt.columns else bt["scaled_returns"].dropna()
    if len(ret) == 0:
        return {"sharpe": np.nan, "max_dd": np.nan, "mean_ret": np.nan, "turnover": np.nan}
    ann_factor = np.sqrt(252)
    sharpe = (ret.mean() - risk_free / 252) / (ret.std() + 1e-12) * ann_factor
    max_dd = bt["drawdown"].min()
    lev = bt["leverage"].dropna()
    turnover = lev.diff().abs().mean() if len(lev) > 1 else 0.0
    rv_ann = ret.std() * ann_factor if len(ret) > 21 else np.nan
    out = {
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "mean_ret": float(ret.mean() * 252),
        "turnover": float(turnover),
        "realized_vol_ann": float(rv_ann) if not np.isnan(rv_ann) else np.nan,
    }
    return out


def mean_variance_utility(
    returns: pd.Series, risk_aversion: float = 2.0, ann_factor: float = 252.0
) -> float:
    """Mean-variance utility: E[r] - (gamma/2) * Var(r), annualized."""
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    mu = r.mean() * ann_factor
    var = r.var() * ann_factor
    return float(mu - (risk_aversion / 2) * var)


def vol_targeting_economic_summary(
    bt: pd.DataFrame,
    target_vol_ann: float = 0.10,
    risk_free: float = 0.0,
    cost_bps: float = 0.0,
    risk_aversion: float = 2.0,
) -> dict[str, float]:
    """
    Full economic value summary: Sharpe, max DD, turnover, realized vol error vs target,
    mean-variance utility.
    """
    base = backtest_summary(bt, risk_free)
    ret = bt["net_returns"].dropna() if "net_returns" in bt.columns else bt["scaled_returns"].dropna()
    base["utility"] = mean_variance_utility(ret, risk_aversion)
    rv = base.get("realized_vol_ann")
    if rv is not None and not np.isnan(rv):
        base["vol_error"] = abs(rv - target_vol_ann)
    return base


def parametric_var(returns: pd.Series, sigma: pd.Series, alpha: float = 0.01) -> pd.Series:
    """Parametric VaR: -sigma * z_alpha (one-day, so sigma is daily)."""
    from scipy import stats
    z = stats.norm.ppf(alpha)
    return (-sigma * z).reindex(returns.index).ffill()


def parametric_es(returns: pd.Series, sigma: pd.Series, alpha: float = 0.01) -> pd.Series:
    """Parametric ES (CVaR): sigma * phi(z_alpha) / alpha, daily."""
    from scipy import stats
    z = stats.norm.ppf(alpha)
    es = sigma * (stats.norm.pdf(z) / alpha)
    return es.reindex(returns.index).ffill()


def kupiec_test(
    returns: pd.Series,
    var_series: pd.Series,
    alpha: float = 0.01,
) -> tuple[float, float]:
    """
    Kupiec (1995) POF test: proportion of failures (exceedances) vs expected.
    H0: true exceedance rate = alpha. Returns (LR_stat, p_value).
    """
    common = returns.dropna().index.intersection(var_series.dropna().index)
    if len(common) == 0:
        return np.nan, np.nan
    r = returns.loc[common]
    v = var_series.loc[common]
    n = len(r)
    x = (r < -np.abs(v)).sum()
    p = alpha
    if x == 0:
        lik_alt = (1 - p) ** n
    else:
        lik_alt = (x / n) ** x * ((n - x) / n) ** (n - x)
    lik_null = (1 - p) ** (n - x) * p ** x
    lr = -2 * (np.log(lik_null + 1e-12) - np.log(lik_alt + 1e-12))
    from scipy import stats
    pval = 1 - stats.chi2.cdf(lr, 1)
    return float(lr), float(pval)
