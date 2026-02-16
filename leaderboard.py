"""
Leaderboard script: average metrics by horizon, regime-conditioned tables, DM test summary.
Reads from artifacts/experiments/ (and master leaderboard.csv).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    import volatility_paths as vpaths
    RESULTS_DIR = vpaths.EXPERIMENTS_DIR
    LEADERBOARD_CSV = vpaths.EXPERIMENTS_DIR / "leaderboard.csv"
except ImportError:
    RESULTS_DIR = Path(__file__).resolve().parent / "results"
    LEADERBOARD_CSV = Path(__file__).resolve().parent / "artifacts" / "experiments" / "leaderboard.csv"


def load_run(run_dir: Path) -> dict:
    """Load experiment run from results directory."""
    metrics_files = list(run_dir.glob("*_metrics.csv"))
    if not metrics_files:
        return {}
    all_metrics = []
    for f in metrics_files:
        df = pd.read_csv(f, index_col=[0, 1] if "Model" in pd.read_csv(f, nrows=0).columns else 0)
        all_metrics.append(df)
    full = pd.concat(all_metrics, axis=0)
    regime_files = list(run_dir.glob("by_regime.csv"))
    by_regime = pd.read_csv(regime_files[0], index_col=0) if regime_files else None
    return {"metrics": full, "by_regime": by_regime, "run_dir": run_dir}


def leaderboard_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Average metrics by model across tickers."""
    if "ticker" in metrics.columns and "Model" in metrics.columns:
        avg = metrics.groupby("Model").agg("mean").round(6)
    elif isinstance(metrics.index, pd.MultiIndex):
        avg = metrics.groupby(level="Model").mean().round(6)
    else:
        avg = metrics.mean().to_frame().T.round(6)
    return avg


def regime_table(by_regime: pd.DataFrame | None) -> pd.DataFrame | None:
    """Regime-conditioned performance table."""
    if by_regime is None or len(by_regime) == 0:
        return None
    return by_regime.round(6)


def dm_summary(metrics: pd.DataFrame) -> pd.DataFrame | None:
    """DM test summary if available in metrics."""
    dm_cols = [c for c in metrics.columns if "DM" in c or "dm" in c]
    if not dm_cols:
        return None
    return metrics[dm_cols].agg(["mean", "std"]).round(4).T


def print_leaderboard(run_dir: Path | None = None) -> None:
    """Print leaderboard for latest run or specified run."""
    if run_dir is None:
        # Prefer run dirs (exclude leaderboard.csv and other files)
        runs = sorted(
            [p for p in RESULTS_DIR.iterdir() if p.is_dir() and (p / "leaderboard_metrics.csv").exists()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not runs:
            print("No results found. Run: python volforecast_cli.py experiments --config configs/sp500_sample.yaml")
            return
        run_dir = runs[0]
    data = load_run(run_dir)
    if not data:
        print(f"No valid metrics in {run_dir}")
        return
    metrics = data["metrics"]
    print(f"\n=== Leaderboard: {run_dir.name} ===\n")
    print("Average metrics by model:")
    print(leaderboard_table(metrics))
    if data.get("by_regime") is not None:
        print("\nBy regime:")
        print(regime_table(data["by_regime"]))
    dm = dm_summary(metrics)
    if dm is not None:
        print("\nDM test summary:")
        print(dm)


def main() -> None:
    parser = argparse.ArgumentParser(description="Volatility forecast leaderboard")
    parser.add_argument("--run", default=None, help="Results run directory (default: latest)")
    parser.add_argument("--master", action="store_true", help="Show master leaderboard.csv (all runs)")
    args = parser.parse_args()
    if args.master and LEADERBOARD_CSV.exists():
        print("\n=== Master leaderboard (all runs) ===\n")
        print(pd.read_csv(LEADERBOARD_CSV).to_string())
        return
    run_dir = Path(args.run) if args.run else None
    if run_dir and not run_dir.is_absolute():
        run_dir = RESULTS_DIR / run_dir
    print_leaderboard(run_dir)


if __name__ == "__main__":
    main()
