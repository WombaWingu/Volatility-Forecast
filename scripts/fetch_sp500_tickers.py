"""
Fetch current S&P 500 ticker list from Wikipedia.
Output: comma-separated tickers (for CLI) or save to a file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0"


def fetch_sp500_tickers() -> list[str]:
    """Scrape S&P 500 symbols from Wikipedia. Returns list of ticker strings."""
    # Fetch with a browser User-Agent to avoid 403 in CI (Wikipedia blocks default urllib)
    req = Request(WIKI_URL, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    # Use html5lib so we don't require lxml (works in CI and minimal installs)
    tables = pd.read_html(html, flavor="html5lib")
    # First table is current constituents
    df = tables[0]
    # Wikipedia column is usually "Symbol" (sometimes "Ticker")
    symbol_col = "Symbol" if "Symbol" in df.columns else "Ticker"
    tickers = df[symbol_col].astype(str).str.strip().tolist()
    # Remove any placeholder or empty
    tickers = [t for t in tickers if t and t != "nan" and len(t) <= 5]
    return tickers


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch S&P 500 ticker list from Wikipedia")
    parser.add_argument("--format", choices=["csv", "lines"], default="csv", help="Output: csv (comma-separated) or lines")
    parser.add_argument("--out", type=str, default="", help="Optional: write list to file (one ticker per line)")
    parser.add_argument("--limit", type=int, default=None, help="Use only first N tickers (default: all)")
    args = parser.parse_args()

    try:
        tickers = fetch_sp500_tickers()
        if args.limit is not None:
            tickers = tickers[: args.limit]
    except Exception as e:
        print(f"Error fetching S&P 500 list: {e}", file=sys.stderr)
        sys.exit(1)

    if args.format == "csv":
        print(",".join(tickers))
    else:
        print("\n".join(tickers))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        content = ",".join(tickers) if args.format == "csv" else "\n".join(tickers)
        out_path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
