# Refactoring Status

## ✅ Completed Phases

### Phase 0: Git Setup
- ✅ Updated `.gitignore` to exclude artifacts/, *.csv, *.html
- ✅ Verified structure

### Phase 1: Artifacts Organization
- ✅ Created `artifacts/` folder structure:
  - `artifacts/cache/` - Data cache
  - `artifacts/signals/` - Daily signals (date-organized)
  - `artifacts/reports/` - HTML reports (date-organized)
  - `artifacts/metrics/` - Metrics CSVs (date-organized)
  - `artifacts/experiments/` - Experiment results
  - `artifacts/cross_sectional/` - Cross-sectional results
  - `artifacts/logs/` - Log files (date-organized)
- ✅ Created `volatility_paths.py` for centralized path management
- ✅ Updated `mini_proj.py` to export to artifacts directories

### Phase 2: Package Structure (Partial)
- ✅ Created `src/volforecast/` package structure
- ✅ Copied modules to package:
  - `data.py`, `models.py`, `eval.py`, `backtest.py`, `risk.py`, `distributions.py`, `paths.py`
  - `portfolio.py`, `ensemble.py`, `multicov.py`
- ✅ Created `__init__.py`
- ✅ Created `pyproject.toml` for package setup
- ✅ Moved scripts to `scripts/` directory
- ⚠️ **TODO**: Update all imports to use package structure (currently modules still in root)

### Phase 3: CLI Entrypoint (Partial)
- ✅ Created `src/volforecast/cli.py` with commands:
  - `daily` - Run daily forecast + generate signals + tomorrow positions
  - `cross-sectional` - Cross-sectional comparison
  - `experiments` - Run experiments from config
  - `tomorrow-position` - Generate tomorrow's position from latest signals
- ⚠️ **TODO**: Fix imports to work with package structure

### Phase 4: Tomorrow Position
- ✅ Added `compute_target_shares()` to `portfolio.py`
- ✅ Added `generate_tomorrow_positions()` to `portfolio.py`
- ✅ Integrated into CLI `daily` command
- ✅ Outputs to `artifacts/signals/<DATE>/tomorrow_positions.csv`

### Phase 5: Config Files
- ✅ Created `configs/` directory
- ✅ Created example configs:
  - `configs/nvda.yaml` - Single ticker config
  - `configs/universe_etfs.yaml` - Multi-ticker config

### Phase 6: Logging
- ✅ Created `src/volforecast/logging_utils.py`
- ✅ Logging writes to `artifacts/logs/<DATE>/run_<TIME>.log`
- ⚠️ **TODO**: Integrate logging into CLI commands

## 🚧 Remaining Work

### Phase 2 Completion
- [ ] Update all module imports to use `from volforecast import ...`
- [ ] Update scripts to import from package
- [ ] Test that imports work correctly

### Phase 3 Completion
- [ ] Fix CLI imports to work with package structure
- [ ] Test all CLI commands
- [ ] Add `dashboard` command

### Phase 6 Completion
- [ ] Integrate logging into all CLI commands
- [ ] Add run ID generation
- [ ] Log all key decisions and outputs

### Phase 7: Systematic Experiments ✅
- [x] Update `run_experiments.py` to use `artifacts/experiments/<RUN_ID>/`
- [x] Master leaderboard: `artifacts/experiments/leaderboard.csv` updated on each run
- [x] Store results in `artifacts/experiments/<RUN_ID>/`

### Phase 8: Scheduling Automation ✅
- [x] GitHub Actions workflow: `.github/workflows/daily.yml`
- [x] Schedule: Mon–Fri 22:00 UTC (5 PM ET)
- [x] Runs tests, daily forecast (NVDA), cross-sectional; uploads artifacts

### Phase 9: Dashboard Integration ✅
- [x] `app_dashboard.py`: option "Use latest artifacts" + "Load from artifacts"
- [x] `leaderboard.py`: reads from `artifacts/experiments/`, supports `--master` for leaderboard.csv
- [x] Dashboard shows latest run without re-running pipeline

## Usage Examples

### Daily Forecast + Tomorrow Position
```bash
python -m volforecast daily --ticker NVDA --export --model ridge --equity 20000
```

### Generate Tomorrow Position from Latest Signals
```bash
python -m volforecast tomorrow-position --ticker NVDA --model ridge --equity 20000
```

### Cross-Sectional Comparison
```bash
python -m volforecast cross-sectional --tickers "SPY,QQQ,IWM" --export cross_section.csv
```

### Run Experiments
```bash
python -m volforecast experiments --config configs/nvda.yaml
```

## File Structure

```
volatility-forecast/
├── src/
│   └── volforecast/
│       ├── __init__.py
│       ├── cli.py
│       ├── data.py
│       ├── models.py
│       ├── eval.py
│       ├── backtest.py
│       ├── risk.py
│       ├── distributions.py
│       ├── portfolio.py
│       ├── paths.py
│       ├── logging_utils.py
│       └── ...
├── scripts/
│   ├── mini_proj.py
│   ├── run_cross_sectional.py
│   ├── run_experiments.py
│   └── TMRW_POSITION.py
├── configs/
│   ├── nvda.yaml
│   └── universe_etfs.yaml
├── artifacts/
│   ├── cache/
│   ├── signals/<DATE>/
│   ├── reports/<DATE>/
│   ├── metrics/<DATE>/
│   ├── experiments/<RUN_ID>/
│   └── logs/<DATE>/
├── tests/
├── pyproject.toml
└── .gitignore
```

## Next Steps

1. **Fix imports**: Update all scripts and modules to use package imports
2. **Test CLI**: Verify all commands work end-to-end
3. **Add logging**: Integrate logging into all operations
4. **Complete experiments**: Make experiment system use artifacts structure
5. **Dashboard**: Update dashboard to read from artifacts
6. **Automation**: Set up GitHub Actions for daily runs
