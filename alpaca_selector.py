"""
ETF universe signal selector.

For each ETF in the universe:
  1. Reads volatility_signals_<TICKER>.csv   — for recent closes (momentum)
  2. Reads tomorrow_positions_<TICKER>.csv  — for forecast_vol_ann + desired shares

Scores each ETF with predicted Sharpe = annualized 20-day momentum / forecast_vol_ann.
Picks the single ETF with the highest score and writes two files:

    artifacts/signals/<DATE>/selected_position.csv   — winning ETF + position details
    artifacts/signals/<DATE>/sharpe_scores.csv        — all ETFs ranked for inspection

Usage:
    python alpaca_selector.py [--date 2026-04-10] [--window 20]
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

import volatility_paths as vpaths  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ETF_UNIVERSE = ["SPY", "QQQ", "IWM", "XLF", "XLE", "GLD", "TLT", "VXX"]

SELECTED_FIELDNAMES = [
    "ticker",
    "signal_date",
    "trade_date",
    "desired_shares_int",
    "dollar_exposure",
    "forecast_vol_ann",
    "trailing_return_ann",
    "predicted_sharpe",
    "close",
    "exposure_multiplier",
]

SCORES_FIELDNAMES = [
    "rank",
    "ticker",
    "predicted_sharpe",
    "trailing_return_ann",
    "forecast_vol_ann",
    "desired_shares_int",
    "dollar_exposure",
    "close",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data readers
# ---------------------------------------------------------------------------
def load_positions(ticker: str, date_str: str) -> dict | None:
    """Read tomorrow_positions_<TICKER>.csv; return first row as dict or None."""
    path = vpaths.SIGNALS_DIR / date_str / f"tomorrow_positions_{ticker}.csv"
    if not path.exists():
        log.warning("%s: positions file not found (%s)", ticker, path)
        return None
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            return row
    log.warning("%s: positions file is empty", ticker)
    return None


def load_closes(ticker: str, date_str: str) -> pd.Series | None:
    """
    Read volatility_signals_<TICKER>.csv and return the Close series,
    sorted oldest-first.
    """
    path = vpaths.SIGNALS_DIR / date_str / f"volatility_signals_{ticker}.csv"
    if not path.exists():
        log.warning("%s: signals file not found (%s)", ticker, path)
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if "Close" not in df.columns:
        log.warning("%s: no Close column in signals file", ticker)
        return None
    return df["Close"].dropna().sort_index()


# ---------------------------------------------------------------------------
# Sharpe scoring
# ---------------------------------------------------------------------------
def annualized_momentum(closes: pd.Series, window: int) -> float | None:
    """
    Trailing *window*-day return annualized to 252 trading days.

    Returns None if there are not enough observations.
    """
    if len(closes) < window + 1:
        return None
    r = float(closes.iloc[-1] / closes.iloc[-(window + 1)] - 1)
    return (1.0 + r) ** (252.0 / window) - 1.0


def score_etf(ticker: str, date_str: str, window: int) -> dict | None:
    """
    Compute predicted Sharpe for one ETF.

    Returns a score dict or None if any required data is missing.
    """
    positions = load_positions(ticker, date_str)
    if positions is None:
        return None

    forecast_vol = float(positions.get("forecast_vol_ann", 0) or 0)
    if forecast_vol <= 0:
        log.warning("%s: zero or missing forecast_vol_ann — skipping", ticker)
        return None

    closes = load_closes(ticker, date_str)
    if closes is None:
        return None

    trailing_ret = annualized_momentum(closes, window)
    if trailing_ret is None:
        log.warning(
            "%s: not enough price history for %d-day momentum (have %d rows) — skipping",
            ticker, window, len(closes),
        )
        return None

    # Predicted Sharpe: annualized momentum / forecast vol (risk-free ≈ 0 for ranking)
    predicted_sharpe = trailing_ret / max(forecast_vol, 1e-6)

    return {
        "ticker": ticker,
        "signal_date": positions.get("signal_date", ""),
        "trade_date": positions.get("trade_date", ""),
        "desired_shares_int": int(positions.get("desired_shares_int", 0)),
        "dollar_exposure": float(positions.get("dollar_exposure", 0)),
        "forecast_vol_ann": forecast_vol,
        "trailing_return_ann": trailing_ret,
        "predicted_sharpe": predicted_sharpe,
        "close": float(positions.get("close", 0)),
        "exposure_multiplier": float(positions.get("exposure_multiplier", 0)),
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_selected(date_str: str, best: dict) -> Path:
    path = vpaths.get_signals_dir(date_str) / "selected_position.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SELECTED_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(best)
    return path


def write_scores(date_str: str, scores: list[dict]) -> Path:
    path = vpaths.get_signals_dir(date_str) / "sharpe_scores.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SCORES_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for rank, row in enumerate(scores, start=1):
            writer.writerow({"rank": rank, **row})
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rank ETF universe by predicted Sharpe and select one position."
    )
    parser.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Signal date (default: today).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Trailing momentum window in trading days (default: 20).",
    )
    args = parser.parse_args()

    date_str = args.date or date.today().isoformat()
    log.info("=== ETF selector | date=%s  universe=%s  window=%d ===",
             date_str, ETF_UNIVERSE, args.window)

    # Score every ETF in the universe
    scores = []
    for ticker in ETF_UNIVERSE:
        result = score_etf(ticker, date_str, args.window)
        if result is not None:
            scores.append(result)
        else:
            log.warning("%s: could not be scored — excluded from selection", ticker)

    if not scores:
        log.error("No ETF could be scored. Cannot select a position.")
        return 1

    # Rank highest predicted Sharpe first
    scores.sort(key=lambda x: x["predicted_sharpe"], reverse=True)

    log.info("ETF rankings by predicted Sharpe:")
    for rank, s in enumerate(scores, start=1):
        log.info(
            "  #%d %-5s  sharpe=%.4f  mom=%.4f  vol=%.4f  shares=%d  $%.0f",
            rank, s["ticker"], s["predicted_sharpe"],
            s["trailing_return_ann"], s["forecast_vol_ann"],
            s["desired_shares_int"], s["dollar_exposure"],
        )

    best = scores[0]
    log.info("SELECTED: %s (predicted Sharpe %.4f)", best["ticker"], best["predicted_sharpe"])

    # Write outputs
    sel_path = write_selected(date_str, best)
    log.info("Selected position written → %s", sel_path)

    scores_path = write_scores(date_str, scores)
    log.info("Full rankings written → %s", scores_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
