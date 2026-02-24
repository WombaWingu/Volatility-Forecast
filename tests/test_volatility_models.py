"""
Unit tests for volatility_models.
Run from project root: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from volforecast import models as vm


@pytest.fixture
def simple_returns():
    """Short return series for tests (no network required)."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.standard_normal(100) * 0.01, index=pd.date_range("2020-01-01", periods=100, freq="B"))


def test_realized_vol_shape(simple_returns):
    out = vm.realized_vol(simple_returns, window=5)
    assert out.shape == simple_returns.shape
    assert out.iloc[:4].isna().all()
    assert out.iloc[4:].notna().all()


def test_realized_vol_value():
    # Two returns 0.01 and 0.01 -> squared sum = 2e-4, sqrt = 0.01414...
    r = pd.Series([0.01, 0.01, 0.0, 0.0, 0.0])
    out = vm.realized_vol(r, window=2)
    assert out.iloc[1] == pytest.approx(np.sqrt(2) * 0.01, rel=1e-5)


def test_annualize_vol():
    daily = pd.Series([0.01])
    out = vm.annualize_vol(daily, trading_days=252, horizon_days=2)
    assert out.iloc[0] == pytest.approx(0.01 * np.sqrt(252 / 2), rel=1e-5)


def test_ewma_volatility_shape(simple_returns):
    out = vm.ewma_volatility(simple_returns, lam=0.94, burn_in=20)
    assert out.shape == simple_returns.shape
    assert out.iloc[19:].notna().all()


def test_volatility_metrics_perfect():
    y = np.array([0.2, 0.3, 0.25])
    yhat = y.copy()
    m = vm.volatility_metrics(y, yhat)
    assert m["MAE"] == 0
    assert m["RMSE"] == 0
    assert m["Correlation"] == 1.0


def test_volatility_metrics_anti_corr():
    y = np.array([1.0, 2.0, 3.0])
    yhat = np.array([3.0, 2.0, 1.0])
    m = vm.volatility_metrics(y, yhat)
    assert m["Correlation"] == -1.0


def test_build_volatility_features(simple_returns):
    data = pd.DataFrame({"Close": 100 * (1 + simple_returns).cumprod()}, index=simple_returns.index)
    data["Simple"] = simple_returns
    feat_cols = vm.build_volatility_features(
        data, simple_returns,
        include_range_vol=False,
        include_har_style=True,
        include_higher_moments=True,
    )
    assert len(feat_cols) >= 6
    for c in feat_cols:
        assert c in data.columns


def test_ridge_rolling_forecast_small_window(simple_returns):
    """Ridge with small window so test runs quickly."""
    data = pd.DataFrame(index=simple_returns.index)
    data["roll_vol_fwd_ann"] = np.sqrt((simple_returns ** 2).rolling(2).sum()).shift(-2) * np.sqrt(252 / 2)
    feat_cols = vm.build_volatility_features(
        data, simple_returns,
        include_range_vol=False,
        include_har_style=True,
        include_higher_moments=False,
    )
    out = vm.ridge_rolling_forecast(
        data, feat_cols, "roll_vol_fwd_ann", window=30, step=5, min_train=15
    )
    assert len(out) == len(data)
    assert out.index.equals(data.index)
    # At least some non-NaN predictions after the first window
    assert out.iloc[30:].notna().sum() >= 1
