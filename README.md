# Stock Volatility Research

A **volatility forecasting** pipeline for equity research: baselines, **HAR-RV**, **GARCH** family (GJR, EGARCH, Student-t), **GAS**, **Ridge**, **XGBoost**, **ensemble**; evaluation (QLIKE, Diebold–Mariano, regime breakdown); **full predictive distributions** (VaR/ES, Christoffersen, DQ, Fissler–Ziegel); **economic value** (vol-targeting with transaction costs, mean-variance utility); **multi-asset** covariance and portfolio construction (risk parity, min variance); **experiment runner** with YAML config; **Streamlit dashboard**. Suitable as a portfolio or research project.

**Project Snapshot (PDF):** [`docs/Project_Report.pdf`](docs/Project_Report.pdf)

## Quickstart

```bash
pip install -r requirements.txt
python volforecast_cli.py daily --ticker NVDA --export --model ridge
```

Outputs are written under `artifacts/` (signals + metrics + reports) and are ignored by git.

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
- Save to `artifacts/experiments/` with unique run id
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

### Unified CLI (`volforecast_cli.py`)

```bash
# Daily forecast for one ticker (signals + optional tomorrow positions)
python volforecast_cli.py daily --ticker NVDA --export --model ridge

# Cross-sectional comparison across many tickers
python volforecast_cli.py cross-sectional --tickers "SPY,QQQ,IWM" --export artifacts/cross_sectional/results.csv

# Experiments from YAML config
python volforecast_cli.py experiments --config configs/sp500_sample.yaml

# Generate tomorrow's position from latest signals
python volforecast_cli.py tomorrow-position --ticker NVDA
```

### Single-asset pipeline (`mini_proj.py`)

```bash
python scripts/mini_proj.py                                    # Metrics table
python scripts/mini_proj.py --ticker SPY --start 2015-01-01    # Override ticker/dates
python scripts/mini_proj.py --plot                             # Plot forecasts vs realized
python scripts/mini_proj.py --report                           # Generate volatility_report.html
python scripts/mini_proj.py --backtest                         # Vol-targeting + VaR/ES validation
```

### Experiment runner

```bash
python scripts/run_experiments.py --config configs/sp500_sample.yaml
# Outputs in artifacts/experiments/<run_id>/
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

### Cross-sectional and S&P 500 top stocks

Run cross-sectional comparison on a custom list or the full S&P 500, then get the top N stocks by volatility forecast accuracy (ridge MAE):

```bash
# Custom tickers
python volforecast_cli.py cross-sectional --tickers "SPY,QQQ,IWM,XLF,XLE" --export artifacts/cross_sectional/cross_section_results.csv

# Fetch current S&P 500 list from Wikipedia, then run cross-sectional
python scripts/fetch_sp500_tickers.py --format csv --out data/sp500_tickers.txt
python volforecast_cli.py cross-sectional --tickers "$(cat data/sp500_tickers.txt)" --export artifacts/cross_sectional/cross_section_results.csv

# Top 10 stocks to invest (best forecast accuracy)
python scripts/top_stocks_from_cross_section.py artifacts/cross_sectional/cross_section_results.csv -n 10 --out artifacts/cross_sectional/top10_stocks.csv
```

Or use the script directly: `python scripts/run_cross_sectional.py --tickers "SPY,QQQ,IWM" --export path.csv`

### Daily workflow (GitHub Actions)

The `.github/workflows/daily.yml` workflow runs on a schedule (Mon–Fri after market close) and on manual dispatch. It:

1. **test** — Runs the test suite.
2. **daily** — Runs the daily forecast for NVDA and uploads artifacts.
3. **cross_sectional** — Fetches the S&P 500 list from Wikipedia (first 200 tickers for speed), runs cross-sectional on them, computes the **top 10 stocks** by ridge MAE (best volatility forecast accuracy), and uploads `artifacts/cross_sectional/` (including `top10_stocks.csv`).

## Project layout

| File | Purpose |
|------|---------|
| `volforecast_cli.py` | Unified CLI: daily, cross-sectional, experiments, tomorrow-position |
| `src/volforecast/` | Package modules (data/models/eval/backtest/risk/etc.) |
| `scripts/mini_proj.py` | Main pipeline runner (single ticker) |
| `scripts/run_experiments.py` | Experiment runner (YAML config) |
| `leaderboard.py` | Leaderboard script |
| `app_dashboard.py` | Streamlit dashboard |
| `scripts/run_cross_sectional.py` | Cross-sectional model comparison across tickers |
| `scripts/fetch_sp500_tickers.py` | Fetch S&P 500 ticker list from Wikipedia |
| `scripts/top_stocks_from_cross_section.py` | Top N stocks from cross-sectional results (by ridge MAE) |
| `configs/` | YAML experiment configs |
| `artifacts/` | Daily/cross-sectional outputs (signals, metrics, top10) |
| `.github/workflows/daily.yml` | CI: tests, daily NVDA, cross-sectional (200 stocks) + top 10 |
| `examples/` | Sample outputs |


