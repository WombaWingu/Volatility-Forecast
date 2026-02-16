"""
Full predictive return distributions: model standardized residuals
(Normal vs Student-t vs skew-t), generate density forecasts, PIT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats



def standardized_residuals(
    returns: pd.Series, sigma: pd.Series
) -> pd.Series:
    """Standardized residuals: z_t = r_t / sigma_t (daily scale)."""
    common = returns.dropna().index.intersection(sigma.dropna().index)
    z = returns.loc[common] / sigma.loc[common].clip(lower=1e-10)
    return z


def fit_residual_distribution(
    z: np.ndarray, dist: str = "normal"
) -> dict:
    """
    Fit distribution to standardized residuals.
    dist: 'normal' | 'studentt' | 'skewt'
    Returns fitted params dict.
    """
    z = z[np.isfinite(z) & (np.abs(z) < 20)]
    if len(z) < 30:
        return {}
    if dist == "normal":
        return {"dist": "normal", "mu": float(np.mean(z)), "sigma": float(np.std(z) + 1e-8)}
    if dist == "studentt":
        df, loc, scale = stats.t.fit(z)
        return {"dist": "studentt", "df": float(df), "loc": float(loc), "scale": float(scale)}
    if dist == "skewt":
        try:
            from scipy.stats import skewnorm
            a, loc, scale = skewnorm.fit(z)
            return {"dist": "skewt", "a": float(a), "loc": float(loc), "scale": float(scale)}
        except Exception:
            return fit_residual_distribution(z, "studentt")
    return {}


def quantile_forecast(
    sigma: np.ndarray, params: dict, alpha: float
) -> np.ndarray:
    """
    VaR from volatility forecast + residual distribution.
    VaR_alpha = -sigma * quantile(alpha) of residual dist.
    """
    s = np.asarray(sigma, dtype=float).ravel()
    n = len(s)
    out = np.full(n, np.nan)
    dist = params.get("dist", "normal")
    if dist == "normal":
        z_alpha = stats.norm.ppf(alpha)
        out = -s * z_alpha
    elif dist == "studentt":
        df = params.get("df", 5.0)
        loc = params.get("loc", 0.0)
        scale = params.get("scale", 1.0)
        z_alpha = stats.t.ppf(alpha, df=df, loc=loc, scale=scale)
        out = -s * z_alpha
    elif dist == "skewt":
        try:
            from scipy.stats import skewnorm
            a = params.get("a", 0.0)
            loc = params.get("loc", 0.0)
            scale = params.get("scale", 1.0)
            z_alpha = skewnorm.ppf(alpha, a=a, loc=loc, scale=scale)
            out = -s * z_alpha
        except Exception:
            z_alpha = stats.norm.ppf(alpha)
            out = -s * z_alpha
    return out


def es_forecast(
    sigma: np.ndarray, params: dict, alpha: float
) -> np.ndarray:
    """
    ES (Expected Shortfall) from volatility + residual distribution.
    ES_alpha = sigma * E[-Z | Z <= z_alpha].
    """
    s = np.asarray(sigma, dtype=float).ravel()
    n = len(s)
    out = np.full(n, np.nan)
    dist = params.get("dist", "normal")
    if dist == "normal":
        z_alpha = stats.norm.ppf(alpha)
        es_z = -stats.norm.pdf(z_alpha) / alpha
        out = s * es_z
    elif dist == "studentt":
        df = params.get("df", 5.0)
        loc = params.get("loc", 0.0)
        scale = params.get("scale", 1.0)
        z_alpha = stats.t.ppf(alpha, df=df, loc=loc, scale=scale)
        # ES for t: closed form
        from scipy.special import gammaln
        def t_es(alpha_val, df_val):
            x = stats.t.ppf(alpha_val, df_val)
            c = (df_val + x**2) / (df_val - 1)
            return -stats.t.pdf(x, df_val) / alpha_val * c * scale + loc
        try:
            es_z = t_es(alpha, df)
            out = s * (-es_z)
        except Exception:
            es_z = -stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
            out = s * es_z
    elif dist == "skewt":
        es_z = -stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
        out = s * es_z
    return out


def pit_values(
    returns: np.ndarray, sigma: np.ndarray, params: dict
) -> np.ndarray:
    """
    Probability Integral Transform: F(r_t | sigma_t, params).
    If model is well calibrated, PIT ~ Uniform(0,1).
    """
    r = np.asarray(returns, dtype=float).ravel()
    s = np.asarray(sigma, dtype=float).ravel()
    n = min(len(r), len(s))
    r, s = r[:n], s[:n]
    pit = np.full(n, np.nan)
    dist = params.get("dist", "normal")
    z = r / np.maximum(s, 1e-10)
    if dist == "normal":
        pit = stats.norm.cdf(z)
    elif dist == "studentt":
        df = params.get("df", 5.0)
        loc = params.get("loc", 0.0)
        scale = params.get("scale", 1.0)
        pit = stats.t.cdf(z, df=df, loc=loc, scale=scale)
    elif dist == "skewt":
        try:
            from scipy.stats import skewnorm
            a = params.get("a", 0.0)
            loc = params.get("loc", 0.0)
            scale = params.get("scale", 1.0)
            pit = skewnorm.cdf(z, a=a, loc=loc, scale=scale)
        except Exception:
            pit = stats.norm.cdf(z)
    return pit


def density_forecast_pipeline(
    returns: pd.Series,
    vol_forecast: pd.Series,
    dist: str = "studentt",
) -> tuple[pd.Series, pd.Series, dict, np.ndarray]:
    """
    Full density forecast: fit residuals, produce VaR/ES, PIT.
    vol_forecast: annualized; we convert to daily for standardization.
    Returns (var_series, es_series, params, pit_array).
    """
    sigma_daily = vol_forecast.reindex(returns.index).ffill() / np.sqrt(252)
    sigma_daily = sigma_daily.clip(lower=1e-10)
    z = standardized_residuals(returns, sigma_daily)
    params = fit_residual_distribution(z.values, dist)
    if not params:
        params = {"dist": "normal", "mu": 0, "sigma": 1}
    var_arr = quantile_forecast(sigma_daily.values, params, 0.01)
    es_arr = es_forecast(sigma_daily.values, params, 0.01)
    pit_arr = pit_values(returns.reindex(sigma_daily.index).ffill().values, sigma_daily.values, params)
    var_series = pd.Series(var_arr, index=returns.index).reindex(returns.index)
    es_series = pd.Series(es_arr, index=returns.index).reindex(returns.index)
    return var_series, es_series, params, pit_arr
