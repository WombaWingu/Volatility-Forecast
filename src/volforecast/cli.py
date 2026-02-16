"""
Unified CLI entrypoint for volatility forecasting operations.
Commands: daily, cross-sectional, experiments, dashboard
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

# Import from root modules (package structure exists but modules still in root for compatibility)
import sys
from pathlib import Path

# Add project root to path for imports
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import from root modules (they're still there)
import volatility_paths as vpaths

# Import portfolio functions directly from src/volforecast/portfolio.py
import importlib.util
portfolio_path = _project_root / "src" / "volforecast" / "portfolio.py"
if portfolio_path.exists():
    spec = importlib.util.spec_from_file_location("portfolio_module", portfolio_path)
    portfolio_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(portfolio_module)
    compute_target_shares = portfolio_module.compute_target_shares
    generate_tomorrow_positions = portfolio_module.generate_tomorrow_positions
else:
    # Fallback: try root module
    try:
        import volatility_portfolio as portfolio_module
        compute_target_shares = portfolio_module.compute_target_shares
        generate_tomorrow_positions = portfolio_module.generate_tomorrow_positions
    except (ImportError, AttributeError):
        raise ImportError("Could not import portfolio functions. Make sure portfolio.py exists.")


def cmd_daily(args):
    """Run daily volatility forecast and generate signals + tomorrow positions."""
    # Import here to avoid circular dependencies
    # mini_proj is in scripts/ directory
    scripts_dir = _project_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import mini_proj
    
    # Run pipeline
    summary_df, eval_data, by_regime, extras = mini_proj.run_pipeline(
        args.ticker,
        args.start,
        args.end,
        horizons=[int(x) for x in args.horizons.split(",")],
        window=args.window,
        step=args.step,
        use_range_vol=not args.no_range_vol,
        run_backtest=args.backtest,
        run_ridge_tuned=args.ridge_tuned,
    )
    
    print(f"\nVolatility forecast comparison — {args.ticker} ({args.start} to {args.end})")
    print(summary_df.round(6))
    
    # Generate tomorrow positions if signals exist
    if "signals" in extras and not extras["signals"].empty:
        signals_df = extras["signals"]
        today_str = date.today().isoformat()
        signals_dir = vpaths.get_signals_dir(today_str)
        
        # Generate tomorrow positions
        model_used = args.model or "ridge"
        equity = args.equity or 10000.0
        
        positions_df = generate_tomorrow_positions(
            signals_df=signals_df,
            model_used=model_used,
            equity=equity,
            target_vol_ann=args.target_vol,
            cap=args.cap,
            floor=args.floor,
            integer_only=True,
        )
        positions_df["ticker"] = args.ticker
        
        # Save tomorrow positions
        positions_path = signals_dir / "tomorrow_positions.csv"
        positions_df.to_csv(positions_path, index=False)
        print(f"\nTomorrow positions saved to {positions_path}")
        print(positions_df.to_string())
    
    # Export if requested
    if args.export:
        today_str = date.today().isoformat()
        metrics_dir = vpaths.get_metrics_dir(today_str)
        signals_dir = vpaths.get_signals_dir(today_str)
        
        summary_df.to_csv(metrics_dir / f"{args.ticker}_volatility_metrics.csv")
        eval_data.to_csv(metrics_dir / f"{args.ticker}_volatility_forecasts.csv")
        extras["signals"].to_csv(signals_dir / "volatility_signals.csv")
        if by_regime is not None:
            by_regime.to_csv(metrics_dir / f"{args.ticker}_volatility_by_regime.csv")
        print(f"\nExported to artifacts/ (date: {today_str})")
    
    # Generate report if requested
    if args.report:
        report_path = mini_proj._write_html_report(
            args.ticker, args.start, args.end, summary_df, eval_data, by_regime, extras
        )
        if args.open_browser:
            import webbrowser
            webbrowser.open(report_path.as_uri())


def cmd_cross_sectional(args):
    """Run cross-sectional comparison across multiple tickers."""
    # Import from scripts directory
    scripts_dir = _project_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import run_cross_sectional
    # This will be refactored later, for now call the existing script
    sys.argv = ["run_cross_sectional.py"] + [
        "--tickers", args.tickers,
        "--start", args.start,
        "--end", args.end,
        "--horizon", str(args.horizon),
    ]
    if args.export:
        sys.argv.extend(["--export", args.export])
    run_cross_sectional.main()


def cmd_experiments(args):
    """Run experiments from config file."""
    # Import from scripts directory
    scripts_dir = _project_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import run_experiments
    sys.argv = ["run_experiments.py", "--config", args.config]
    run_experiments.main()


def cmd_tomorrow_position(args):
    """Generate tomorrow's position from latest signals."""
    # Find latest signals file
    signals_file = vpaths.get_latest_signals_file(args.ticker)
    if signals_file is None:
        print(f"Error: No signals file found. Run 'daily' command first.")
        return
    
    signals_df = pd.read_csv(signals_file, index_col=0, parse_dates=True)
    
    model_used = args.model or "ridge"
    equity = args.equity or 10000.0
    
    positions_df = generate_tomorrow_positions(
        signals_df=signals_df,
        model_used=model_used,
        equity=equity,
        target_vol_ann=args.target_vol,
        cap=args.cap,
        floor=args.floor,
        integer_only=True,
    )
    
    if args.ticker:
        positions_df["ticker"] = args.ticker
    
    # Save to signals directory
    today_str = date.today().isoformat()
    signals_dir = vpaths.get_signals_dir(today_str)
    positions_path = signals_dir / "tomorrow_positions.csv"
    positions_df.to_csv(positions_path, index=False)
    
    print(f"Tomorrow positions saved to {positions_path}")
    print("\n" + positions_df.to_string())


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Volatility Forecast CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Daily command
    daily_parser = subparsers.add_parser("daily", help="Run daily volatility forecast")
    daily_parser.add_argument("--ticker", default="NVDA", help="Ticker symbol")
    daily_parser.add_argument("--start", default="2010-01-27", help="Start date")
    daily_parser.add_argument("--end", default=None, help="End date (default: today)")
    daily_parser.add_argument("--horizons", default="1,5,21", help="Comma-separated horizons")
    daily_parser.add_argument("--window", type=int, default=1260, help="Rolling window")
    daily_parser.add_argument("--step", type=int, default=2, help="Step size")
    daily_parser.add_argument("--no-range-vol", action="store_true", help="Disable range-based vol")
    daily_parser.add_argument("--backtest", action="store_true", help="Run backtest")
    daily_parser.add_argument("--ridge-tuned", action="store_true", help="Use tuned Ridge")
    daily_parser.add_argument("--export", action="store_true", help="Export CSVs")
    daily_parser.add_argument("--report", action="store_true", help="Generate HTML report")
    daily_parser.add_argument("--open-browser", action="store_true", help="Open report in browser")
    daily_parser.add_argument("--model", default="ridge", help="Model for position sizing")
    daily_parser.add_argument("--equity", type=float, help="Account equity")
    daily_parser.add_argument("--target-vol", type=float, default=0.10, help="Target volatility")
    daily_parser.add_argument("--cap", type=float, default=1.0, help="Exposure cap")
    daily_parser.add_argument("--floor", type=float, default=0.05, help="Volatility floor")
    
    # Cross-sectional command
    cross_parser = subparsers.add_parser("cross-sectional", help="Cross-sectional comparison")
    cross_parser.add_argument("--tickers", default="SPY,QQQ,IWM", help="Comma-separated tickers")
    cross_parser.add_argument("--start", default="2015-01-01", help="Start date")
    cross_parser.add_argument("--end", default=None, help="End date")
    cross_parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon")
    cross_parser.add_argument("--export", help="Export CSV path")
    
    # Experiments command
    exp_parser = subparsers.add_parser("experiments", help="Run experiments")
    exp_parser.add_argument("--config", default="configs/sp500_sample.yaml", help="Config file")
    
    # Tomorrow position command
    pos_parser = subparsers.add_parser("tomorrow-position", help="Generate tomorrow's position")
    pos_parser.add_argument("--ticker", help="Ticker symbol")
    pos_parser.add_argument("--model", default="ridge", help="Model to use")
    pos_parser.add_argument("--equity", type=float, default=10000.0, help="Account equity")
    pos_parser.add_argument("--target-vol", type=float, default=0.10, help="Target volatility")
    pos_parser.add_argument("--cap", type=float, default=1.0, help="Exposure cap")
    pos_parser.add_argument("--floor", type=float, default=0.05, help="Volatility floor")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "daily":
        if args.end is None:
            args.end = date.today().isoformat()
        cmd_daily(args)
    elif args.command == "cross-sectional":
        if args.end is None:
            args.end = date.today().isoformat()
        cmd_cross_sectional(args)
    elif args.command == "experiments":
        cmd_experiments(args)
    elif args.command == "tomorrow-position":
        cmd_tomorrow_position(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
