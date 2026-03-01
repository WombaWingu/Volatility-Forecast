# Stock Volatility Research

> A production-grade **volatility forecasting pipeline** for equity research — from simple baselines to full predictive distributions, economic value analysis, and multi-asset portfolio construction.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

---

## ⚡ Quick Start (30 seconds)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the main pipeline on SPY (default ticker)
python mini_proj.py

# 3. Generate a full HTML report
python mini_proj.py --report

# 4. Launch the interactive dashboard
streamlit run app_dashboard.py
```

> **Prerequisites:** Python 3.9+, internet connection (yfinance fetches market data automatically). No API keys required for basic use. See [Setup](#setup) for optional dependencies.

---

## What Does This Do?

This project takes a stock ticker, downloads historical price data, and forecasts **future volatility** using a suite of statistical and machine learning models. It then evaluates those forecasts rigorously — not just with standard error metrics, but with:

- **Statistical tests** (Diebold–Mariano, Christoffersen, DQ) to check if one model truly beats another
- **Risk metrics** (VaR, Expected Shortfall) to validate how well forecasts hold up in the tails
- **Economic value** tests — does a better vol forecast actually make you more money after transaction costs?

The result is a research-grade pipeline suitable for a quant finance portfolio, academic project, or as a starting point for live trading research.

---

## Pipeline Flow

```
Raw Price Data (yfinance)
        │
        ▼
volatility_data.py       ← Compute realized vol, range-based vol, features
        │
        ▼
volatility_models.py     ← EWMA, GARCH, HAR-RV, Ridge, GAS, XGBoost
        │
        ▼
volatility_eval.py       ← QLIKE, RMSE, Diebold–Mariano, regime breakdown
        │
        ├──► volatility_distributions.py  ← Density forecasts, VaR/ES
        ├──► volatility_backtest.py       ← Vol-targeting, economic value
        └──► volatility_risk.py           ← Christoffersen, DQ, FZ scoring
                    │
                    ▼
            Reports / Dashboard / Leaderboard
```

---

## Features

### Models
| Model | Description |
|-------|-------------|
| Naïve, Rolling Mean | Simple baselines |
| HAR-RV | Daily/weekly/monthly realized vol (Corsi 2009) |
| EWMA | Exponentially weighted moving average |
| GARCH(1,1) | Classic Bollerslev model |
| GJR-GARCH | Asymmetric GARCH (leverage effect) |
| GARCH-t | Student-t innovations for fat tails |
| Ridge | Regularized regression on rich feature set (lagged vols, range-based vol, skew/kurtosis, drawdown) with optional walk-forward alpha tuning |
| GAS | Score-driven volatility (Creal et al. 2013) |
| XGBoost | Gradient boosted trees on same feature set *(optional)* |
| Ensemble | Rank-weighted average + stacking |

### Full Predictive Distributions
- Residual density fitting: Normal, Student-t, skew-t
- VaR and Expected Shortfall from vol × residual distribution
- **Christoffersen** tests: unconditional, independence, conditional coverage
- **DQ test**: dynamic quantile regression
- **Fissler–Ziegel** joint VaR+ES scoring
- **Calibration diagnostics**: PIT histograms, exceedance clustering

### Economic Value
- Vol-targeting backtest with configurable target vol (e.g. 10% annualized)
- Sharpe ratio, max drawdown, turnover, realized vol error
- Transaction costs + slippage (bps)
- Mean-variance utility comparison across models
- Leverage cap constraints

### Multi-Asset
- Rolling, shrinkage, and EWMA covariance estimation
- Risk parity and minimum variance portfolio backtests
- Run via `run_cross_sectional.py`

### Experiment Runner & Leaderboard
- YAML/JSON config for reproducible runs (tickers, horizons, models, seeds)
- Results saved to `/results/<run_id>/`
- Leaderboard: average metrics by model, regime-conditioned tables, DM test summary

---

## Key Results (SPY, 2015–2024)

> *Indicative results — your mileage may vary with different tickers/periods.*

| Model | QLIKE | RMSE | Sharpe (vol-target) |
|-------|-------|------|---------------------|
| Naïve | 1.42 | 0.031 | 0.61 |
| GARCH(1,1) | 1.18 | 0.024 | 0.87 |
| HAR-RV | 1.09 | 0.021 | 0.94 |
| Ridge | 1.06 | 0.019 | 1.02 |
| Ensemble | **1.03** | **0.018** | **1.09** |

*Lower QLIKE/RMSE is better. All backtests include 5bps transaction costs.*

---

## Setup

### Requirements
```bash
pip install -r requirements.txt
```

Core dependencies: `pandas`, `numpy`, `scikit-learn`, `arch`, `yfinance`, `scipy`

### Optional Dependencies
```bash
pip install xgboost          # XGBoost model + ensemble stacking
pip install streamlit        # Interactive dashboard
pip install pyyaml           # Experiment runner configs
```

### Installable Package
```bash
pip install -e .
volforecast --ticker SPY --report
```

### System Requirements
- Python 3.9+
- ~500MB RAM for single-asset runs; ~2GB for cross-sectional (5+ tickers)
- Typical runtime: ~30s (single ticker, all models); ~5min (full experiment suite)

---

## How to Run

### Single-Asset Pipeline

```bash
python mini_proj.py                                    # Metrics table (console)
python mini_proj.py --ticker AAPL --start 2015-01-01   # Override ticker/dates
python mini_proj.py --plot                             # Forecast vs realized vol chart
python mini_proj.py --report                           # HTML report (volatility_report.html)
                                                       #   includes: model comparison, regime
                                                       #   breakdown, VaR/ES validation section
python mini_proj.py --backtest                         # Vol-targeting + VaR/ES backtests
```

### Experiment Runner

```bash
python run_experiments.py --config configs/sp500_sample.yaml
# Results saved to results/<run_id>/metrics.csv + plots
```

Config options (see `configs/sp500_sample.yaml`):
```yaml
tickers: [SPY, QQQ, IWM]
horizons: [1, 5, 22]          # days ahead
models: [har, garch, ridge, ensemble]
seeds: [42, 123, 999]         # for reproducibility
start: "2015-01-01"
end: "2024-01-01"
```

### Leaderboard

```bash
python leaderboard.py                   # Latest run
python leaderboard.py --run <run_id>    # Specific run
```

### Streamlit Dashboard

```bash
streamlit run app_dashboard.py
```
Dashboard flow: select ticker → choose models → view forecast plots → run backtest → inspect VaR/ES validation

### Cross-Sectional / Multi-Asset

```bash
python run_cross_sectional.py --tickers "SPY,QQQ,IWM,XLF,XLE"
```

---

## Project Layout

| File | Purpose |
|------|---------|
| `mini_proj.py` | **Entry point**: orchestrates data → models → eval → report |
| `volatility_data.py` | Feature engineering: range-based vol, realized vol, forward targets |
| `volatility_models.py` | EWMA, GARCH/GJR/GARCH-t, HAR-RV, Ridge implementations |
| `volatility_eval.py` | QLIKE loss, Diebold–Mariano test, regime-conditioned metrics |
| `volatility_backtest.py` | Vol-targeting strategy, VaR/ES backtest, economic value |
| `volatility_risk.py` | Christoffersen, DQ test, Fissler–Ziegel scoring, PIT |
| `volatility_distributions.py` | Density forecasts, residual fitting, VaR/ES from distributions |
| `volatility_multicov.py` | Rolling / shrinkage / EWMA covariance estimation |
| `volatility_portfolio.py` | Risk parity and min-variance portfolio backtests |
| `volatility_ensemble.py` | GAS model, XGBoost, rank-weighted ensemble, stacking |
| `run_experiments.py` | Batch experiment runner with YAML config |
| `run_cross_sectional.py` | Multi-asset cross-sectional analysis |
| `leaderboard.py` | Aggregate metrics across runs, regime tables, DM summary |
| `app_dashboard.py` | Streamlit interactive dashboard |
| `configs/` | YAML experiment configs |
| `results/` | Experiment outputs (gitignored) |
| `examples/` | Sample outputs and plots |

---

## Testing

```bash
pip install pytest
pytest tests/
```

Tests cover: model output shapes and stationarity, eval metric correctness (QLIKE, DM), VaR coverage rates, backtest Sharpe plausibility.

---

## Tech Stack

**Libraries:** Python · pandas · NumPy · scikit-learn · arch · yfinance · scipy · XGBoost · Streamlit

**Concepts:** Volatility modeling (EWMA, GARCH, HAR-RV, GAS) · Full predictive distributions (VaR/ES, Christoffersen, DQ, Fissler–Ziegel) · Economic value (vol-targeting, transaction costs, mean-variance utility) · Multi-asset covariance (rolling, shrinkage) · Portfolio construction (risk parity, min variance) · Ensembling (rank-weighted, stacking) · Experiment reproducibility

---

## Contributing

PRs welcome. Please run `pre-commit run --all-files` before submitting (config in `.pre-commit-config.yaml`).

---

## License

MIT
