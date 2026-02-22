"""
Read cross-sectional results CSV and output top N stocks for investing.
Ranking: best (lowest) ridge_MAE = most accurate volatility forecasts for vol-targeting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MAE_COL = "ridge_MAE"  # primary model for signals


def main() -> None:
    parser = argparse.ArgumentParser(description="Top stocks from cross-sectional results")
    parser.add_argument("csv", nargs="?", default="artifacts/cross_sectional/cross_section_results.csv", help="Cross-section results CSV")
    parser.add_argument("-n", "--top", type=int, default=10, help="Number of top stocks")
    parser.add_argument("--out", type=str, default="", help="Output file path (default: print only)")
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print(f"File not found: {path}")
        return

    df = pd.read_csv(path)
    # Ticker may be index or column
    if "ticker" in df.columns:
        ticker_col = "ticker"
    elif df.index.name == "ticker" or (isinstance(df.index, pd.RangeIndex) and "Unnamed: 0" in df.columns):
        ticker_col = "Unnamed: 0" if "Unnamed: 0" in df.columns else df.columns[0]
    else:
        ticker_col = df.columns[0]
    tickers = df[ticker_col].astype(str)

    if MAE_COL not in df.columns:
        # Fallback: any *_MAE column (prefer ridge)
        mae_cols = [c for c in df.columns if c.endswith("_MAE")]
        if not mae_cols:
            print("No MAE column found in CSV.")
            return
        MAE_COL_USE = "ridge_MAE" if "ridge_MAE" in mae_cols else mae_cols[0]
    else:
        MAE_COL_USE = MAE_COL

    df = df.assign(_ticker=tickers, _mae=pd.to_numeric(df[MAE_COL_USE], errors="coerce")).dropna(subset=["_mae"])
    top = df.nsmallest(args.top, "_mae")[["_ticker", "_mae"]].rename(columns={"_ticker": "ticker", "_mae": MAE_COL_USE})
    top = top.reset_index(drop=True)

    print(f"Top {args.top} stocks by {MAE_COL_USE} (lower = better forecast accuracy):")
    print(top.to_string(index=True))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        top.to_csv(out_path, index=False)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
