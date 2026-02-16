# Quick Start Guide

## Current Status

The refactoring is **mostly complete** for phases 0-5. The package structure is set up, but imports still work from the root directory modules for backward compatibility.

## Running the System

### Option 1: Using the CLI (Recommended)

From the project root:

```bash
# Daily forecast + generate tomorrow positions
python volforecast_cli.py daily --ticker NVDA --export --model ridge --equity 20000

# Generate tomorrow position from latest signals
python volforecast_cli.py tomorrow-position --ticker NVDA --model ridge --equity 20000

# Cross-sectional comparison
python volforecast_cli.py cross-sectional --tickers "SPY,QQQ,IWM" --export cross_section.csv

# Run experiments
python volforecast_cli.py experiments --config configs/nvda.yaml
```

**Note**: If you install the package with `pip install -e .`, you can use `python -m volforecast.cli` instead.

### Option 2: Using Original Scripts (Still Works)

```bash
# From scripts directory or project root
python scripts/mini_proj.py --ticker NVDA --export --report
python scripts/run_experiments.py --config configs/nvda.yaml
```

## Output Locations

All outputs now go to `artifacts/` directory:

- **Signals**: `artifacts/signals/<DATE>/volatility_signals.csv`
- **Tomorrow Positions**: `artifacts/signals/<DATE>/tomorrow_positions.csv`
- **Metrics**: `artifacts/metrics/<DATE>/<TICKER>_volatility_metrics.csv`
- **Reports**: `artifacts/reports/<DATE>/volatility_report_<TICKER>.html`
- **Logs**: `artifacts/logs/<DATE>/run_<TIME>.log`
- **Experiments**: `artifacts/experiments/<RUN_ID>/` and master `artifacts/experiments/leaderboard.csv`

## Experiments & Leaderboard

- Run experiments (saves to artifacts, updates master leaderboard):
  ```bash
  python volforecast_cli.py experiments --config configs/sp500_sample.yaml
  ```
- View leaderboard (latest run or master list):
  ```bash
  python leaderboard.py
  python leaderboard.py --master
  ```

## Dashboard

- Streamlit dashboard (run pipeline or load latest from artifacts):
  ```bash
  streamlit run app_dashboard.py
  ```
  In the sidebar: check **Use latest artifacts** then click **Load from artifacts** to view the latest run without re-running.

## GitHub Actions (Daily)

- Workflow `.github/workflows/daily.yml` runs Mon–Fri at 22:00 UTC: runs tests, daily forecast (NVDA), cross-sectional; uploads artifacts.

## Configuration

Example config files are in `configs/`:

- `configs/nvda.yaml` - Single ticker config
- `configs/universe_etfs.yaml` - Multi-ticker config

## Next Steps

1. **Test the CLI**: Run `python volforecast_cli.py daily --ticker NVDA --export`
2. **Check outputs**: Verify files appear in `artifacts/signals/<TODAY>/`
3. **Generate positions**: Run `python volforecast_cli.py tomorrow-position --ticker NVDA`
4. **Review logs**: Check `artifacts/logs/<TODAY>/` for run logs

## Troubleshooting

### Import Errors

If you get import errors, make sure you're running from the project root:

```bash
cd "c:\Users\P\Desktop\CS Practice\PRJ\Volatility Forecast"
python volforecast_cli.py daily --ticker NVDA
```

### Module Not Found

The CLI uses a standalone script (`volforecast_cli.py`) that handles imports automatically. If you prefer to use the package structure:

1. Install the package: `pip install -e .`
2. Then use: `python -m volforecast.cli daily --ticker NVDA`

The standalone script (`volforecast_cli.py`) works without installation and is recommended for now.
