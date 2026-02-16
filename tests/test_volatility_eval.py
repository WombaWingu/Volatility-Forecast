"""Tests for volatility_eval (losses, DM test, metrics)."""

import numpy as np
import pytest

import volatility_eval as ve


def test_loss_mae_rmse():
    y = np.array([0.2, 0.3, 0.25])
    yhat = np.array([0.22, 0.28, 0.24])  # errors 0.02, 0.02, 0.01
    assert ve.loss_mae(y, yhat) == pytest.approx(0.02 / 3 + 0.02 / 3 + 0.01 / 3, rel=1e-5)
    assert ve.loss_rmse(y, yhat) == pytest.approx(np.sqrt((0.02**2 + 0.02**2 + 0.01**2) / 3), rel=1e-5)


def test_loss_qlike_perfect():
    y = np.array([0.2, 0.3])
    # QLIKE: (y²/σ²) - log(y²/σ²) - 1; when yhat=y, ratio=1 so term = 0
    assert ve.loss_qlike(y, y) == pytest.approx(0.0, abs=1e-6)


def test_volatility_metrics_full():
    y = np.array([0.2, 0.3, 0.25])
    yhat = y.copy()
    m = ve.volatility_metrics_full(y, yhat)
    assert m["MAE"] == 0
    assert m["RMSE"] == 0
    assert m["Correlation"] == 1.0
    assert "QLIKE" in m
    assert "MSE_var" in m


def test_diebold_mariano_same_errors():
    e = np.random.RandomState(42).randn(100)
    dm_stat, p_val = ve.diebold_mariano(e, e, loss="squared", h=1)
    # Same errors => d=0, var can be 0 => p_val may be nan; or high p
    assert np.isnan(p_val) or p_val > 0.5
