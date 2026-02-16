"""
Stock Volatility Research — core models and metrics.
EWMA, GARCH(1,1)/EGARCH/GJR, Ridge, HAR-RV, baselines (naïve, rolling mean).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from arch import arch_model
except ImportError:
    arch_model = None

import volatility_data as vd


def realized_vol(returns: pd.Series, window: int = 5) -> pd.Series:
    """Rolling realized volatility: sqrt(sum of squared returns) over window."""
    return np.sqrt((returns ** 2).rolling(window).sum())


def annualize_vol(daily_vol: pd.Series, trading_days: int = 252, horizon_days: int = 2) -> pd.Series:
    """Annualize daily vol: daily_vol * sqrt(trading_days / horizon_days).
    Use same horizon_days as your realized vol (e.g. 2) so EWMA/GARCH match realized scale."""
    return daily_vol * np.sqrt(trading_days / horizon_days)


def naive_vol_forecast(
    realized_vol_ann: pd.Series,
    horizon_days: int = 1,
) -> pd.Series:
    """Baseline: tomorrow's vol = today's vol (no horizon scaling if same-day realized)."""
    return realized_vol_ann.shift(horizon_days).rename("naive_vol")


def rolling_mean_vol_forecast(
    realized_vol_ann: pd.Series,
    window: int = 22,
    horizon_days: int = 1,
) -> pd.Series:
    """Baseline: forecast = rolling mean of past realized vol."""
    roll = realized_vol_ann.rolling(window).mean().shift(horizon_days)
    return roll.rename("roll_mean_vol")


def har_rv_rolling_forecast(
    rv_daily_ann: pd.Series,
    window: int = 252,
    step: int = 1,
    horizon_days: int = 1,
) -> pd.Series:
    """
    HAR-RV: regress next realized vol on daily, weekly (5d), monthly (22d) lagged RV.
    Rolling OLS; returns annualized forecast aligned with index.
    """
    rv_w = np.sqrt((rv_daily_ann ** 2).rolling(5).mean())
    rv_m = np.sqrt((rv_daily_ann ** 2).rolling(22).mean())
    out = pd.Series(index=rv_daily_ann.index, dtype=float)
    # Need at least 22 + window for monthly feature
    for t in range(window + 22, len(rv_daily_ann), step):
        train_start = t - window
        train_end = t
        # Target: rv at s; features: rv_d, rv_w, rv_m at s-1
        y_train = rv_daily_ann.iloc[train_start + 1 : train_end].values
        X_d = rv_daily_ann.iloc[train_start : train_end - 1].values
        X_w = rv_w.iloc[train_start : train_end - 1].values
        X_m = rv_m.iloc[train_start : train_end - 1].values
        X = np.column_stack([X_d, X_w, X_m])
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y_train)
        if mask.sum() < 50:
            continue
        X, y_train = X[mask], y_train[mask]
        try:
            beta = np.linalg.lstsq(X, y_train, rcond=None)[0]
            x_t = np.array([
                rv_daily_ann.iloc[t - 1],
                rv_w.iloc[t - 1],
                rv_m.iloc[t - 1],
            ])
            if not np.isfinite(x_t).all():
                continue
            pred = np.dot(beta, x_t)
            out.iloc[t] = max(1e-8, float(pred))
        except Exception:
            continue
    return out.ffill().rename("har_rv_pred")


def ewma_volatility(returns: pd.Series, lam: float = 0.94, burn_in: int = 20) -> pd.Series:
    """
    EWMA variance then volatility (annualized for 2-day horizon).
    Uses burn_in observations to initialize variance.
    """
    ewma_var = pd.Series(index=returns.index, dtype=float)
    ewma_var.iloc[burn_in - 1] = returns.iloc[:burn_in].var()
    for i in range(burn_in, len(returns)):
        ewma_var.iloc[i] = lam * ewma_var.iloc[i - 1] + (1 - lam) * (returns.iloc[i - 1] ** 2)
    return annualize_vol(np.sqrt(ewma_var))


def _garch_rolling_forecast_impl(
    returns: pd.Series,
    window: int,
    step: int,
    scale_pct: bool,
    vol_model: str = "GARCH",
    dist: str = "normal",
    **arch_kw: Any,
) -> pd.Series:
    """Shared impl for GARCH/EGARCH/GJR with optional Student-t."""
    if arch_model is None:
        raise ImportError("arch package required: pip install arch")
    scale = 100.0 if scale_pct else 1.0
    out = pd.Series(index=returns.index, dtype=float)
    # Suppress GARCH optimizer convergence warnings (arch may print via showwarning)
    _showwarning = warnings.showwarning
    def _noop_showwarning(*args, **kwargs):
        pass
    try:
        warnings.showwarning = _noop_showwarning
        for t in range(window, len(returns), step):
            train = returns.iloc[t - window : t].dropna()
            if len(train) < window * 0.9:
                continue
            try:
                am = arch_model(
                    train * scale,
                    mean="Zero",
                    vol=vol_model,
                    p=1,
                    q=1,
                    dist=dist,
                    **arch_kw,
                )
                res = am.fit(disp="off")
                f = res.forecast(horizon=1, reindex=False)
                var_next = f.variance.values[-1, 0]
                sigma_next = np.sqrt(var_next) / scale
                out.iloc[t] = annualize_vol(pd.Series([sigma_next])).iloc[0]
            except Exception:
                continue
    finally:
        warnings.showwarning = _showwarning
    return out


def garch_rolling_forecast(
    returns: pd.Series,
    window: int = 1260,
    step: int = 2,
    scale_pct: bool = True,
) -> pd.Series:
    """
    Rolling GARCH(1,1) one-step volatility forecast (annualized for 2-day).
    Returns Series aligned with returns index; NaN before first forecast.
    """
    return _garch_rolling_forecast_impl(
        returns, window, step, scale_pct, vol_model="GARCH", dist="normal"
    ).rename("garch_pred")


def gjr_garch_rolling_forecast(
    returns: pd.Series,
    window: int = 1260,
    step: int = 2,
    scale_pct: bool = True,
) -> pd.Series:
    """GJR-GARCH (asymmetric leverage): negative shocks increase vol more."""
    return _garch_rolling_forecast_impl(
        returns, window, step, scale_pct, vol_model="GARCH", dist="normal", o=1
    ).rename("gjr_pred")


def egarch_rolling_forecast(
    returns: pd.Series,
    window: int = 1260,
    step: int = 2,
    scale_pct: bool = True,
) -> pd.Series:
    """EGARCH (exponential GARCH, asymmetric in log-variance)."""
    return _garch_rolling_forecast_impl(
        returns, window, step, scale_pct, vol_model="EGARCH", dist="normal", p=1, o=1, q=1
    ).rename("egarch_pred")


def garch_studentt_rolling_forecast(
    returns: pd.Series,
    window: int = 1260,
    step: int = 2,
    scale_pct: bool = True,
) -> pd.Series:
    """GARCH(1,1) with Student-t innovations (heavy tails)."""
    return _garch_rolling_forecast_impl(
        returns, window, step, scale_pct, vol_model="GARCH", dist="t"
    ).rename("garch_t_pred")


def volatility_metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    """MAE, RMSE, and correlation between realized and predicted vol."""
    err = yhat - y
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "Correlation": float(np.corrcoef(y, yhat)[0, 1]) if np.std(yhat) > 0 else np.nan,
    }


def volatility_regime(vol_series: pd.Series, q_low: float = 0.33, q_high: float = 0.67) -> pd.Series:
    """
    Simple regime label: low / medium / high by quantiles.
    Returns a Series of strings "low", "medium", "high".
    """
    q1 = vol_series.quantile(q_low)
    q2 = vol_series.quantile(q_high)
    regime = pd.Series(index=vol_series.index, dtype=object)
    regime.loc[vol_series <= q1] = "low"
    regime.loc[(vol_series > q1) & (vol_series <= q2)] = "medium"
    regime.loc[vol_series > q2] = "high"
    return regime


def build_volatility_features(
    data: pd.DataFrame,
    returns: pd.Series,
    include_range_vol: bool = True,
    include_har_style: bool = True,
    include_higher_moments: bool = True,
) -> list[str]:
    """
    Add volatility features for Ridge: lagged vols (d/w/m), range-based vol,
    rolling kurtosis/skew, drawdown. All added in place; returns feature list.
    """
    # Close-to-close vol at different windows
    data["rolling_vol_5"] = vd.realized_vol_close(returns, 5)
    data["rolling_vol_10"] = vd.realized_vol_close(returns, 10)
    data["rolling_vol_20"] = vd.realized_vol_close(returns, 20)
    data["rolling_vol_63"] = vd.realized_vol_close(returns, 63)  # ~3 months
    data["rolling_abs_5"] = returns.abs().rolling(5).mean()
    data["rolling_abs_10"] = returns.abs().rolling(10).mean()
    data["return_std_5"] = returns.rolling(5).std()
    data["return_std_20"] = returns.rolling(20).std()

    feat_cols = [
        "rolling_vol_5", "rolling_vol_10", "rolling_vol_20", "rolling_vol_63",
        "rolling_abs_5", "rolling_abs_10", "return_std_5", "return_std_20",
    ]

    if include_range_vol and "High" in data.columns and "Low" in data.columns:
        data["range_vol_gk"] = vd.range_based_vol_rolling(
            data["Open"], data["High"], data["Low"], data["Close"], window=5, method="gk"
        )
        feat_cols.append("range_vol_gk")

    if include_har_style:
        rv_d = vd.realized_vol_close(returns, 1)
        data["rv_d"] = rv_d
        data["rv_w"] = np.sqrt((rv_d ** 2).rolling(5).mean())
        data["rv_m"] = np.sqrt((rv_d ** 2).rolling(22).mean())
        feat_cols.extend(["rv_d", "rv_w", "rv_m"])

    if include_higher_moments:
        data["rolling_skew_20"] = returns.rolling(20).skew()
        data["rolling_kurt_20"] = returns.rolling(20).kurt()
        feat_cols.extend(["rolling_skew_20", "rolling_kurt_20"])

    if "Close" in data.columns:
        data["drawdown"] = (data["Close"] / data["Close"].cummax() - 1).fillna(0)
        data["drawdown_5"] = data["drawdown"].rolling(5).min()
        feat_cols.extend(["drawdown", "drawdown_5"])

    return feat_cols


def ridge_rolling_forecast(
    data: pd.DataFrame,
    feat_cols: list[str],
    target_col: str,
    window: int = 1260,
    step: int = 2,
    alpha: float = 1.0,
    min_train: int = 200,
) -> pd.Series:
    """
    Rolling Ridge (with StandardScaler) forecast of forward volatility.
    Returns Series aligned with data index.
    """
    ridge_pred = pd.Series(index=data.index, dtype=float)
    for t in range(window, len(data), step):
        train_slice = slice(t - window, t)
        test_idx = t
        x_train = data.iloc[train_slice][feat_cols].copy()
        y_train = data.iloc[train_slice][target_col].copy()
        x_test = data.iloc[[test_idx]][feat_cols].copy()
        train_ok = x_train.notna().all(axis=1) & y_train.notna()
        x_train = x_train.loc[train_ok]
        y_train = y_train.loc[train_ok]
        if len(x_train) < min_train:
            continue
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])
        pipe.fit(x_train, y_train)
        ridge_pred.iloc[t] = pipe.predict(x_test).item()
    return ridge_pred.rename("ridge_pred")


def ridge_rolling_forecast_tuned(
    data: pd.DataFrame,
    feat_cols: list[str],
    target_col: str,
    window: int = 1260,
    step: int = 2,
    alphas: list[float] | None = None,
    n_cv: int = 5,
    min_train: int = 200,
) -> pd.Series:
    """
    Ridge with walk-forward alpha tuning: purge gap so no leakage.
    Uses last n_cv folds of training window to pick alpha (min RMSE).
    """
    if alphas is None:
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    ridge_pred = pd.Series(index=data.index, dtype=float)
    for t in range(window, len(data), step):
        train_slice = slice(t - window, t)
        x_all = data.iloc[train_slice][feat_cols]
        y_all = data.iloc[train_slice][target_col]
        train_ok = x_all.notna().all(axis=1) & y_all.notna()
        x_all = x_all.loc[train_ok]
        y_all = y_all.loc[train_ok]
        if len(x_all) < min_train or len(x_all) < n_cv * 20:
            continue
        # Walk-forward CV: fold k uses [0 : -n_cv*len_k] train, last chunk test
        n = len(x_all)
        fold_size = max(20, n // (n_cv + 1))
        best_alpha = alphas[0]
        best_rmse = np.inf
        for alpha in alphas:
            rmses = []
            for k in range(n_cv):
                test_start = n - (k + 1) * fold_size
                test_end = n - k * fold_size
                if test_start < fold_size or test_end <= test_start:
                    continue
                x_train = x_all.iloc[:test_start]
                y_train = y_all.iloc[:test_start]
                x_test = x_all.iloc[test_start:test_end]
                y_test = y_all.iloc[test_start:test_end]
                if len(x_train) < 50:
                    continue
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=alpha)),
                ])
                pipe.fit(x_train, y_train)
                pred = pipe.predict(x_test)
                rmses.append(np.sqrt(np.mean((pred - y_test.values) ** 2)))
            if rmses and np.mean(rmses) < best_rmse:
                best_rmse = np.mean(rmses)
                best_alpha = alpha
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=best_alpha)),
        ])
        pipe.fit(x_all, y_all)
        x_t = data.iloc[[t]][feat_cols]
        if x_t.notna().all().all():
            ridge_pred.iloc[t] = max(1e-8, pipe.predict(x_t).item())
    return ridge_pred.ffill().rename("ridge_pred")
