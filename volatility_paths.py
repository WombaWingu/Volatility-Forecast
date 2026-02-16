"""
Centralized path management for artifacts and outputs.
All outputs go to artifacts/ with date-organized subdirectories.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path


# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CACHE_DIR = ARTIFACTS_DIR / "cache"
SIGNALS_DIR = ARTIFACTS_DIR / "signals"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"
CROSS_SECTIONAL_DIR = ARTIFACTS_DIR / "cross_sectional"
LOGS_DIR = ARTIFACTS_DIR / "logs"


def get_date_dir(base_dir: Path, date_str: str | None = None) -> Path:
    """Get date-organized subdirectory. Creates if needed."""
    if date_str is None:
        date_str = date.today().isoformat()
    date_path = base_dir / date_str
    date_path.mkdir(parents=True, exist_ok=True)
    return date_path


def get_signals_dir(date_str: str | None = None) -> Path:
    """Get signals directory for a given date."""
    return get_date_dir(SIGNALS_DIR, date_str)


def get_reports_dir(date_str: str | None = None) -> Path:
    """Get reports directory for a given date."""
    return get_date_dir(REPORTS_DIR, date_str)


def get_metrics_dir(date_str: str | None = None) -> Path:
    """Get metrics directory for a given date."""
    return get_date_dir(METRICS_DIR, date_str)


def get_logs_dir(date_str: str | None = None) -> Path:
    """Get logs directory for a given date."""
    return get_date_dir(LOGS_DIR, date_str)


def get_experiment_dir(run_id: str) -> Path:
    """Get experiment directory for a run_id."""
    exp_dir = EXPERIMENTS_DIR / run_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def get_cache_path(ticker: str, start: str, end: str) -> Path:
    """Get cache file path for ticker data."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = f"{ticker}_{start}_{end}.csv"
    return CACHE_DIR / key


def get_latest_signals_file(ticker: str | None = None) -> Path | None:
    """Get the most recent volatility_signals.csv file."""
    if not SIGNALS_DIR.exists():
        return None
    
    # Find all date subdirectories
    date_dirs = [d for d in SIGNALS_DIR.iterdir() if d.is_dir()]
    if not date_dirs:
        return None
    
    # Sort by date (newest first)
    date_dirs.sort(reverse=True)
    
    # Check each date directory for signals file
    for date_dir in date_dirs:
        signals_file = date_dir / "volatility_signals.csv"
        if signals_file.exists():
            return signals_file
    
    return None
