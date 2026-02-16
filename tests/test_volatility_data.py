"""Tests for volatility_data (range-based vol, realized vol)."""

import numpy as np
import pandas as pd
import pytest

import volatility_data as vd


def test_parkinson_vol():
    high = pd.Series([105, 102, 103])
    low = pd.Series([100, 99, 101])
    out = vd.parkinson_vol(high, low)
    assert out.iloc[0] == pytest.approx(np.sqrt((np.log(105/100)**2) / (4 * np.log(2))), rel=1e-5)


def test_realized_vol_close():
    r = pd.Series([0.01, -0.01, 0.02], index=pd.date_range("2020-01-01", periods=3, freq="B"))
    out = vd.realized_vol_close(r, window=2)
    assert out.iloc[1] == pytest.approx(np.sqrt(0.01**2 + 0.01**2), rel=1e-5)


def test_forward_realized_vol():
    r = pd.Series([0.01] * 5, index=pd.date_range("2020-01-01", periods=5, freq="B"))
    out = vd.forward_realized_vol(r, horizon_days=2, trading_days=252)
    assert out.notna().sum() >= 1
