"""
Cross-sectional volatility forecast comparison: run a universe of tickers,
rank models by average performance, and show which ticker each model wins on (heatmap/table).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import volatility_data as vd
import volatility_eval as ve
import volatility_models as vm
from mini_proj import load_prices, WINDOW, STEP, TRADING_DAYS

# Default universe: SPY + sector ETFs or a few names
DEFAULT_TICKERS = "SPY,QQQ,IWM,XLF,XLE,XLK,XLU,XLV,XLY,XLP,XLB,XLI"


def run_one_ticker(
    ticker: str,
    start: str,
    end: str,
    horizon: int = 1,
    use_cache: bool = True,
) -> pd.DataFrame | None:
    """Run pipeline for one ticker; return summary metrics (one row) or None on failure."""
    try:
        data = load_prices(ticker, start, end, use_cache=use_cache)
    except Exception:
        return None
    simple_ret = data["Simple"].dropna()
    data = data.loc[simple_ret.index].copy()
    if len(data) < 600:
        return None
    ann_factor = np.sqrt(TRADING_DAYS / horizon)
    rv_fwd = vd.forward_realized_vol(simple_ret, horizon, TRADING_DAYS)
    rv_daily = vm.annualize_vol(vd.realized_vol_close(simple_ret, 1), TRADING_DAYS, 1)
    data["roll_vol_fwd_ann"] = rv_fwd

    naive = vm.naive_vol_forecast(rv_daily, horizon)
    roll_mean = vm.rolling_mean_vol_forecast(rv_daily, 22, horizon)
    har = vm.har_rv_rolling_forecast(rv_daily, window=WINDOW, step=STEP, horizon_days=horizon)
    ewma = vm.ewma_volatility(simple_ret, lam=0.94, burn_in=20)
    garch = vm.garch_rolling_forecast(simple_ret, window=WINDOW, step=STEP).ffill()
    ridge_pred = vm.ridge_rolling_forecast(
        data,
        vm.build_volatility_features(data, simple_ret, include_range_vol=("High" in data.columns)),
        "roll_vol_fwd_ann",
        window=WINDOW,
        step=STEP,
        alpha=1.0,
    ).clip(lower=0).ffill()

    eval_df = pd.DataFrame({
        "realized": rv_fwd,
        "naive": naive, "roll_mean": roll_mean, "har_rv": har,
        "ewma": ewma, "garch": garch, "ridge": ridge_pred,
    }).dropna(subset=["realized"]).dropna(how="all", subset=["naive", "roll_mean", "har_rv", "ewma", "garch", "ridge"])
    if len(eval_df) < 100:
        return None
    y = eval_df["realized"].values
    row = {"ticker": ticker}
    for name in ["naive", "roll_mean", "har_rv", "ewma", "garch", "ridge"]:
        if name not in eval_df.columns:
            continue
        m = ve.volatility_metrics_full(y, eval_df[name].values)
        row[f"{name}_MAE"] = m["MAE"]
        row[f"{name}_QLIKE"] = m["QLIKE"]
    return pd.DataFrame([row]).set_index("ticker")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-sectional volatility model comparison")
    parser.add_argument("--tickers", default=DEFAULT_TICKERS, help="Comma-separated tickers")
    parser.add_argument("--start", default="2015-01-01", help="Start date")
    parser.add_argument("--end", default="2026-02-12", help="End date")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon in days")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache")
    parser.add_argument("--export", default="", help="Export CSV path (e.g. cross_section_results.csv)")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",")]
    results = []
    for t in tickers:
        print(f"Running {t}...")
        df = run_one_ticker(t, args.start, args.end, args.horizon, use_cache=not args.no_cache)
        if df is not None:
            results.append(df)
    if not results:
        print("No tickers produced results.")
        return
    full = pd.concat(results, axis=0)

    # Average performance per model (MAE, QLIKE)
    mae_cols = [c for c in full.columns if c.endswith("_MAE")]
    qlike_cols = [c for c in full.columns if c.endswith("_QLIKE")]
    model_mae = {c.replace("_MAE", ""): full[c].mean() for c in mae_cols}
    model_qlike = {c.replace("_QLIKE", ""): full[c].mean() for c in qlike_cols}
    rank_mae = pd.Series(model_mae).sort_values()
    rank_qlike = pd.Series(model_qlike).sort_values()
    print("\n--- Average MAE (lower better) ---")
    print(rank_mae.to_string())
    print("\n--- Average QLIKE (lower better) ---")
    print(rank_qlike.to_string())

    out_dir = Path("artifacts/cross_sectional")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "cross_section_results.csv"   # or your chosen filename
    full.to_csv(out_path)

    # Which model wins per ticker (by MAE)
    wins = {}
    for ticker in full.index:
        row = full.loc[ticker]
        mae_vals = {c.replace("_MAE", ""): row[c] for c in mae_cols if c in row}
        if mae_vals:
            wins[ticker] = min(mae_vals, key=mae_vals.get)
    win_counts = pd.Series(wins).value_counts()
    print("\n--- Model win count (best MAE per ticker) ---")
    print(win_counts.to_string())

    # Heatmap-style: ticker x model MAE (normalized per row so 0 = best, 1 = worst)
    mae_df = full[[c for c in mae_cols]].copy()
    mae_df.columns = [c.replace("_MAE", "") for c in mae_df.columns]
    # Rank within each row: 0 = best
    rank_df = mae_df.rank(axis=1, method="min")
    print("\n--- Ticker x Model rank (1 = best MAE for that ticker) ---")
    print(rank_df.to_string())

    if args.export:
        full.to_csv(args.export)
        rank_df.to_csv(args.export.replace(".csv", "_ranks.csv"))
        print(f"\nExported {args.export} and ranks.")


if __name__ == "__main__":
    main()
