"""
Alpaca paper trading executor — single-ETF rotation strategy.

Reads selected_position.csv (written by alpaca_selector.py), then:
  1. Closes any open position that is NOT the selected ETF.
  2. Buys / adjusts the selected ETF to its target share count.

Each order (including no-ops) is appended to:
    artifacts/signals/<DATE>/orders.csv

Required environment variables:
    ALPACA_API_KEY    — Alpaca API key ID
    ALPACA_SECRET_KEY — Alpaca secret key

Usage:
    python alpaca_executor.py [--date 2026-04-10] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path

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
# Hard cap: never let dollar_exposure from the signal exceed this.
MAX_DOLLAR_EXPOSURE = 100_000.0

LOG_FIELDNAMES = [
    "timestamp",
    "ticker",
    "action",           # "open" | "close" | "adjust" | "no_change"
    "shares",
    "side",             # "buy" | "sell"
    "order_id",
    "status",
    "dollar_exposure",
    "forecast_vol_ann",
    "predicted_sharpe",
    "error",
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
# Signal reader
# ---------------------------------------------------------------------------
def read_selected_position(date_str: str) -> dict | None:
    """Read selected_position.csv written by alpaca_selector.py."""
    path = vpaths.SIGNALS_DIR / date_str / "selected_position.csv"
    if not path.exists():
        log.error("selected_position.csv not found: %s", path)
        log.error("Run alpaca_selector.py first.")
        return None
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            return row
    log.error("selected_position.csv is empty: %s", path)
    return None


# ---------------------------------------------------------------------------
# Order log
# ---------------------------------------------------------------------------
def append_order_log(date_str: str, record: dict) -> None:
    """Append one row to artifacts/signals/<date_str>/orders.csv."""
    orders_path = vpaths.get_signals_dir(date_str) / "orders.csv"
    write_header = not orders_path.exists()
    with open(orders_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(record)
    log.info("Order logged → %s", orders_path)


def _blank_record(ticker: str, action: str, status: str, error: str = "") -> dict:
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "ticker": ticker,
        "action": action,
        "shares": 0,
        "side": "",
        "order_id": "",
        "status": status,
        "dollar_exposure": 0,
        "forecast_vol_ann": 0,
        "predicted_sharpe": 0,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Order helpers
# ---------------------------------------------------------------------------
def _submit_market_order(client, ticker: str, qty: int, side, dry_run: bool) -> dict:
    """
    Submit a market order.  Returns a partial log record (no timestamp/ticker).
    Errors are caught and surfaced in the record — never raised.
    """
    from alpaca.trading.enums import TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    record = {"shares": qty, "side": side.value, "order_id": "", "status": "", "error": ""}

    if dry_run:
        log.info("[DRY RUN] Would %s %d share(s) of %s.", side.value, qty, ticker)
        record["status"] = "dry_run"
        return record

    try:
        req = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=side,
            # DAY orders placed after market hours queue for next open —
            # ideal for signals generated after the 4 PM ET close.
            time_in_force=TimeInForce.DAY,
        )
        order = client.submit_order(req)
        record["order_id"] = str(order.id)
        record["status"] = str(order.status)
        log.info(
            "Order submitted — %s %d %s | id=%s status=%s",
            side.value, qty, ticker, order.id, order.status,
        )
    except Exception as exc:
        record["status"] = "error"
        record["error"] = str(exc)
        log.error("Order submission failed for %s %d %s: %s", side.value, qty, ticker, exc)

    return record


# ---------------------------------------------------------------------------
# Core execution
# ---------------------------------------------------------------------------
def execute_rotation(selected: dict, dry_run: bool = False) -> list[dict]:
    """
    Reconcile Alpaca paper account with the selected position.

    Steps:
      1. Close every open position that is NOT the selected ticker.
      2. Buy / adjust the selected ticker to its target share count.

    Returns a list of log-record dicts (one per order or no-op).
    """
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide

    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        raise EnvironmentError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set as environment variables."
        )

    selected_ticker = selected["ticker"].upper()
    desired_shares = int(selected["desired_shares_int"])
    dollar_exposure = float(selected.get("dollar_exposure", 0))
    forecast_vol = float(selected.get("forecast_vol_ann", 0))
    predicted_sharpe = float(selected.get("predicted_sharpe", 0))

    # Safety guard
    if dollar_exposure > MAX_DOLLAR_EXPOSURE:
        log.warning(
            "Signal dollar_exposure $%.2f exceeds cap $%.2f — setting desired_shares=0.",
            dollar_exposure, MAX_DOLLAR_EXPOSURE,
        )
        desired_shares = 0

    client = TradingClient(api_key, secret_key, paper=True)

    # Fetch all open positions (returns [] when account is flat)
    try:
        open_positions = client.get_all_positions()
    except Exception as exc:
        raise RuntimeError(f"Could not fetch open positions from Alpaca: {exc}") from exc

    current_by_ticker: dict[str, int] = {}
    for pos in open_positions:
        current_by_ticker[pos.symbol.upper()] = int(float(pos.qty))

    log.info("Open positions: %s", dict(current_by_ticker) or "none")
    log.info(
        "Target: %s  desired_shares=%d  $%.0f  vol=%.4f  sharpe=%.4f",
        selected_ticker, desired_shares, dollar_exposure, forecast_vol, predicted_sharpe,
    )

    records: list[dict] = []
    now = datetime.utcnow().isoformat()

    # --- Step 1: close any position NOT in the selected ticker ---
    for ticker, current_shares in current_by_ticker.items():
        if ticker == selected_ticker:
            continue
        if current_shares <= 0:
            continue

        log.info("Closing stale position: %s (%d shares)", ticker, current_shares)
        partial = _submit_market_order(client, ticker, current_shares, OrderSide.SELL, dry_run)
        records.append({
            "timestamp": now,
            "ticker": ticker,
            "action": "close",
            "dollar_exposure": 0,
            "forecast_vol_ann": 0,
            "predicted_sharpe": 0,
            **partial,
        })

    # --- Step 2: open / adjust the selected ticker ---
    current_shares = current_by_ticker.get(selected_ticker, 0)
    delta = desired_shares - current_shares
    log.info(
        "%s — desired=%d  current=%d  delta=%+d",
        selected_ticker, desired_shares, current_shares, delta,
    )

    base_record = {
        "timestamp": now,
        "ticker": selected_ticker,
        "dollar_exposure": dollar_exposure,
        "forecast_vol_ann": forecast_vol,
        "predicted_sharpe": predicted_sharpe,
    }

    if delta == 0:
        log.info("%s: already at target — no order needed.", selected_ticker)
        records.append({
            **base_record,
            "action": "no_change",
            "shares": 0,
            "side": "",
            "order_id": "",
            "status": "skipped",
            "error": "",
        })
    else:
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        action = "open" if current_shares == 0 and delta > 0 else "adjust"
        partial = _submit_market_order(client, selected_ticker, abs(delta), side, dry_run)
        records.append({**base_record, "action": action, **partial})

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute Alpaca paper trades based on alpaca_selector.py output."
    )
    parser.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Signal date (default: today).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would be traded without submitting orders.",
    )
    args = parser.parse_args()

    date_str = args.date or date.today().isoformat()
    log.info("=== Alpaca executor | date=%s  dry_run=%s ===", date_str, args.dry_run)

    # Read selected position from selector output
    selected = read_selected_position(date_str)
    if selected is None:
        return 1

    log.info(
        "Selected position: ticker=%s  shares=%s  $%s  vol=%.4f  sharpe=%.4f",
        selected.get("ticker"),
        selected.get("desired_shares_int"),
        selected.get("dollar_exposure"),
        float(selected.get("forecast_vol_ann", 0)),
        float(selected.get("predicted_sharpe", 0)),
    )

    # Execute rotation
    records: list[dict]
    try:
        records = execute_rotation(selected, dry_run=args.dry_run)
    except EnvironmentError as exc:
        log.error("%s", exc)
        return 1
    except Exception as exc:
        log.error("Unexpected error during execution: %s", exc, exc_info=True)
        records = [_blank_record(selected.get("ticker", "UNKNOWN"), "error", "error", str(exc))]

    # Log every order (including no-ops and errors)
    had_error = False
    for record in records:
        try:
            append_order_log(date_str, record)
        except Exception as exc:
            log.error("Failed to write order log entry: %s", exc)
        if record.get("status") == "error":
            had_error = True

    return 1 if had_error else 0


if __name__ == "__main__":
    sys.exit(main())
