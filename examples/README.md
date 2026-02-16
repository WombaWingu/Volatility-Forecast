# Volatility Forecast — Example Outputs

This folder contains sample outputs from the volatility forecasting pipeline.

## Contents

- `sample_report.html` — Example HTML research report (run `python mini_proj.py --ticker SPY --report` to generate)
- `sample_metrics.csv` — Example headline metrics table
- `sample_forecasts.csv` — Example forecast vs realized data (last 504 rows)

## Key Plots to Include

1. **Forecast vs realized** — Compare model forecasts (EWMA, GARCH, Ridge, HAR-RV) vs forward realized vol
2. **Regime performance** — MAE/RMSE/QLIKE by low/medium/high volatility regime
3. **VaR exceedances** — Hit sequence / exceedance clustering over time

## Generating Examples

```bash
python mini_proj.py --ticker SPY --start 2020-01-01 --report --export --backtest
# Copies volatility_metrics.csv, volatility_forecasts.csv, volatility_report.html to examples/
```

```bash
python run_experiments.py --config configs/sp500_sample.yaml
# Results in results/<run_id>/
```
