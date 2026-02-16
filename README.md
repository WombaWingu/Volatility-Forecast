# Stock Volatility Research

A **volatility forecasting** pipeline for equity research: baselines, **HAR-RV**, **GARCH** family (GJR, EGARCH, Student-t), **GAS**, **Ridge**, **XGBoost**, **ensemble**; evaluation (QLIKE, Diebold–Mariano, regime breakdown); **full predictive distributions** (VaR/ES, Christoffersen, DQ, Fissler–Ziegel); **economic value** (vol-targeting with transaction costs, mean-variance utility); **multi-asset** covariance and portfolio construction (risk parity, min variance); **experiment runner** with YAML config; **Streamlit dashboard**. Suitable as a portfolio or research project.

## Features

### Models
- **Baselines**: Naïve, rolling-mean
- **HAR-RV** (daily/weekly/monthly realized vol)
- **EWMA**, **GARCH(1,1)**, **GJR-GARCH**, **GARCH-t** (Student-t)
- **Ridge** with rich features (lagged vols, range-based vol, skew/kurtosis, drawdown) and optional walk-forward alpha tuning
- **GAS** (score-driven volatility)
- **XGBoost** on feature set (optional)
- **Ensemble**: rank-weighted average, stacking

### Full Predictive Distributions
- Density forecasts: model standardized residuals (Normal, Student-t, skew-t)
- VaR/ES from vol + residual distribution
- **Christoffersen** (unconditional, independence, conditional coverage)
- **DQ test** (dynamic quantile)
- **ES backtest**, **Fissler–Ziegel** joint VaR+ES scoring
- **Calibration diagnostics**: PIT histograms, exceedance clustering

### Economic Value
- Vol-targeting backtest: Sharpe, max drawdown, turnover, realized vol error vs target
- Transaction costs + slippage (bps)
- Mean-variance utility
- Leverage constraints (cap exposure)

### Multi-Asset
- Rolling / shrinkage / EWMA covariance
- Risk parity, minimum variance portfolio backtests

### Experiment Runner
- YAML/JSON config (tickers, horizons, models, seeds)
- Save to `/results/` with unique run id
- **Leaderboard** script: average metrics by model, regime-conditioned tables, DM summary

### Dashboard & Reports
- **Streamlit** dashboard: ticker → models → plots → backtest → VaR/ES validation
- Enhanced HTML report with VaR/ES validation section
- `/examples/` with sample outputs

## Setup

```bash
pip install -r requirements.txt
```

Optional: XGBoost (ensemble), Streamlit (dashboard), PyYAML (experiment configs).

For installable package:

```bash
pip install -e .
# then: volforecast --ticker SPY --report
```

## How to run

### Single-asset pipeline (`mini_proj.py`)

```bash
python mini_proj.py                                    # Metrics table
python mini_proj.py --ticker SPY --start 2015-01-01    # Override ticker/dates
python mini_proj.py --plot                             # Plot forecasts vs realized
python mini_proj.py --report                           # Generate volatility_report.html
python mini_proj.py --backtest                         # Vol-targeting + VaR/ES validation
```

### Experiment runner

```bash
python run_experiments.py --config configs/sp500_sample.yaml
# Results in results/<run_id>/
```

### Leaderboard

```bash
python leaderboard.py                                  # Latest run
python leaderboard.py --run <run_id>                   # Specific run
```

### Streamlit dashboard

```bash
streamlit run app_dashboard.py
```

### Cross-sectional

```bash
python run_cross_sectional.py --tickers "SPY,QQQ,IWM,XLF,XLE"
```

## Project layout

| File | Purpose |
|------|---------|
| `mini_proj.py` | Main pipeline: load data, run models, eval, report |
| `volatility_models.py` | EWMA, GARCH/GJR/GARCH-t, HAR-RV, Ridge |
| `volatility_data.py` | Range-based vol, realized vol, forward targets |
| `volatility_eval.py` | QLIKE, Diebold–Mariano, regime metrics |
| `volatility_backtest.py` | Vol-targeting, VaR/ES, economic value |
| `volatility_risk.py` | Christoffersen, DQ, FZ scoring, PIT |
| `volatility_distributions.py` | Density forecasts, VaR/ES from residuals |
| `volatility_multicov.py` | Rolling/shrinkage/EWMA covariance |
| `volatility_portfolio.py` | Risk parity, min variance backtests |
| `volatility_ensemble.py` | GAS, XGBoost, rank-weighted ensemble |
| `run_experiments.py` | Experiment runner (YAML config) |
| `leaderboard.py` | Leaderboard script |
| `app_dashboard.py` | Streamlit dashboard |
| `configs/` | YAML experiment configs |
| `results/` | Experiment outputs |
| `examples/` | Sample outputs |

## Resume / portfolio

**Tech:** Python, pandas, NumPy, scikit-learn, arch, yfinance, scipy, XGBoost, Streamlit.

**Concepts:** Volatility modeling (EWMA, GARCH, HAR-RV, GAS), full predictive distributions (VaR/ES, Christoffersen, DQ, Fissler–Ziegel), economic value (vol-targeting, transaction costs, mean-variance utility), multi-asset covariance (rolling, shrinkage), portfolio construction (risk parity, min variance), ensembling (rank-weighted, stacking), experiment reproducibility.
