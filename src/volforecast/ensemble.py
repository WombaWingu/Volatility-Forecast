"""
Ensemble models: GAS (score-driven), XGBoost/LightGBM, rank-weighted average, stacking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import models as vm


def gas_volatility_rolling(
    returns: pd.Series,
    window: int = 252,
    step: int = 2,
) -> pd.Series:
    """
    Simple score-driven (GAS-style) volatility: f_t = omega + alpha*score + beta*f_{t-1}.
    Score = r^2_{t-1} - f_{t-1} (for variance). Parameters from EWMA-like initialization.
    """
    r = returns.dropna()
    n = len(r)
    f = np.full(n, np.nan)
    f[window - 1] = (r.iloc[:window] ** 2).mean()
    omega, alpha, beta = 0.01, 0.08, 0.90
    for t in range(window, n):
        score = r.iloc[t - 1] ** 2 - f[t - 1]
        f[t] = omega + alpha * score + beta * f[t - 1]
        f[t] = max(f[t], 1e-12)
    sigma = np.sqrt(f)
    ann = vm.annualize_vol(pd.Series(sigma, index=r.index), 252, 1)
    return ann.rename("gas_pred")


def xgb_volatility_rolling(
    data: pd.DataFrame,
    feat_cols: list[str],
    target_col: str,
    window: int = 1260,
    step: int = 2,
) -> pd.Series:
    """
    XGBoost regressor on volatility features (time-series split).
    """
    try:
        import xgboost as xgb  # type: ignore[import-untyped]
    except ImportError:
        return pd.Series(dtype=float)
    out = pd.Series(index=data.index, dtype=float)
    for t in range(window, len(data), step):
        train = data.iloc[t - window : t]
        x_train = train[feat_cols].dropna(how="all")
        y_train = train[target_col]
        mask = x_train.notna().all(axis=1) & y_train.notna()
        x_train = x_train[mask]
        y_train = y_train[mask]
        if len(x_train) < 200:
            continue
        model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        model.fit(x_train, y_train)
        x_test = data.iloc[[t]][feat_cols]
        if x_test.notna().all().all():
            out.iloc[t] = max(1e-8, float(model.predict(x_test)[0]))
    return out.ffill().rename("xgb_pred")


def rank_weighted_ensemble(
    forecasts: dict[str, pd.Series],
    realized: pd.Series,
    window: int = 252,
) -> pd.Series:
    """
    Rank-weighted ensemble: weight models by inverse rank of recent RMSE.
    """
    common = realized.dropna().index
    for k, v in forecasts.items():
        common = common.intersection(v.dropna().index)
    if len(common) < window + 10:
        return pd.Series(dtype=float)
    rv = realized.loc[common]
    out = pd.Series(index=common, dtype=float)
    names = list(forecasts.keys())
    for t in range(window, len(common)):
        idx = common[t]
        rv_t = rv.iloc[t - window : t].values
        rmses = []
        for name in names:
            f = forecasts[name].loc[common].iloc[t - window : t].values
            mask = np.isfinite(rv_t) & np.isfinite(f)
            if mask.sum() < 50:
                rmses.append(np.inf)
            else:
                rmses.append(np.sqrt(np.mean((rv_t[mask] - f[mask]) ** 2)))
        ranks = np.argsort(np.argsort(rmses))
        inv_rank = 1.0 / (ranks + 1)
        w = inv_rank / inv_rank.sum()
        pred = sum(w[i] * forecasts[names[i]].loc[idx] for i in range(len(names)) if pd.notna(forecasts[names[i]].loc[idx]))
        out.loc[idx] = pred
    return out.ffill().rename("ensemble")


def stacking_ensemble(
    forecasts: dict[str, pd.Series],
    realized: pd.Series,
    window: int = 252,
    step: int = 21,
) -> pd.Series:
    """
    Meta-model stacking: Ridge on model forecasts, walk-forward.
    """
    from sklearn.linear_model import Ridge

    common = realized.dropna().index
    for k, v in forecasts.items():
        common = common.intersection(v.dropna().index)
    if len(common) < window + step:
        return pd.Series(dtype=float)
    X = np.column_stack([forecasts[n].loc[common].values for n in forecasts])
    y = realized.loc[common].values
    out = np.full(len(common), np.nan)
    for t in range(window, len(common) - step, step):
        x_train = X[t - window : t]
        y_train = y[t - window : t]
        mask = np.isfinite(x_train).all(axis=1) & np.isfinite(y_train)
        if mask.sum() < 100:
            continue
        x_train = x_train[mask]
        y_train = y_train[mask]
        meta = Ridge(alpha=1.0).fit(x_train, y_train)
        for s in range(t, min(t + step, len(common))):
            x_test = X[s : s + 1]
            if np.isfinite(x_test).all():
                out[s] = max(1e-8, meta.predict(x_test)[0])
    return pd.Series(out, index=common).ffill().rename("stacking")
