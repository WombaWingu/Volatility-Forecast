"""
Stock Volatility Research — full pipeline.
Models: Naïve, Rolling mean, HAR-RV, EWMA, GARCH(1,1), GJR-GARCH, GARCH-t, Ridge (with tuning).
Evaluation: MAE, RMSE, QLIKE, MSE_var, Log_vol, Diebold–Mariano, regime breakdown, prediction intervals.
Optional: vol-targeting backtest, VaR/ES, Kupiec test.
"""

from __future__ import annotations

import argparse
import base64
import sys
import warnings
import webbrowser
from io import BytesIO
from pathlib import Path

# Suppress GARCH optimizer convergence warnings (from arch/scipy) before loading models
warnings.filterwarnings("ignore", message=".*optimizer.*")
warnings.filterwarnings("ignore", message=".*onvergence.*")
try:
    from scipy.optimize import OptimizeWarning
    warnings.filterwarnings("ignore", category=OptimizeWarning)
except ImportError:
    pass

import numpy as np
import pandas as pd
import yfinance as yf

import volatility_backtest as vbt
import volatility_data as vd
import volatility_distributions as vdist
import volatility_eval as ve
import volatility_models as vm
import volatility_paths as vpaths
import volatility_risk as vr

# === CONFIG ===
TICKER = "NVDA"
START = "2010-01-27"
END = "2026-02-15"
WINDOW = 1260
STEP = 2
TRADING_DAYS = 252
HORIZONS = [1, 5, 21]  # 1d, 5d, 21d ahead
CACHE_DIR = vpaths.CACHE_DIR  # Use centralized cache directory


def _cache_path(ticker: str, start: str, end: str) -> Path:
    return vpaths.get_cache_path(ticker, start, end)


def load_prices(ticker: str, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """Load OHLC; cache to disk when use_cache=True."""
    cache = _cache_path(ticker, start, end)
    if use_cache and cache.exists():
        data = pd.read_csv(cache, index_col=0, parse_dates=True)
        print(f"Loaded {ticker} from cache ({len(data)} rows)")
        return data
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(raw["Close"], pd.DataFrame):
        close = raw["Close"].squeeze()
        open_ = raw["Open"].squeeze() if "Open" in raw.columns else close
        high = raw["High"].squeeze() if "High" in raw.columns else close
        low = raw["Low"].squeeze() if "Low" in raw.columns else close
    else:
        close = raw["Close"]
        open_ = raw.get("Open", close)
        high = raw.get("High", close)
        low = raw.get("Low", close)
    data = pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close,
    }, index=close.index).dropna()
    data["Simple"] = data["Close"].pct_change()
    data["Log"] = np.log(data["Close"] / data["Close"].shift(1))
    if use_cache:
        data.to_csv(cache)
    return data


def run_pipeline(
    ticker: str,
    start: str,
    end: str,
    horizons: list[int],
    window: int,
    step: int,
    use_range_vol: bool = True,
    run_backtest: bool = False,
    run_ridge_tuned: bool = False,
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Run full pipeline for one ticker. Returns:
    - summary_df: model x metrics (for primary horizon)
    - eval_data: index x (realized, model forecasts...)
    - by_regime: regime x metrics
    - extras: backtest summary, DM p-values, etc.
    """
    data = load_prices(ticker, start, end)
    simple_ret = data["Simple"].dropna()
    data = data.loc[simple_ret.index].copy()

    # Primary horizon for model fitting (use first in list, e.g. 1)
    h_primary = horizons[0]
    ann_factor = np.sqrt(TRADING_DAYS / h_primary)

    # Realized vol: close-to-close and optional range-based (annualized)
    rv_close = vd.realized_vol_close(simple_ret, h_primary)
    rv_fwd_ann = vd.forward_realized_vol(simple_ret, h_primary, TRADING_DAYS)
    data["roll_vol_fwd_ann"] = rv_fwd_ann
    if use_range_vol and "High" in data.columns:
        data["rv_range"] = vd.range_based_vol_rolling(
            data["Open"], data["High"], data["Low"], data["Close"],
            window=h_primary, method="gk",
        )
        data["rv_range_ann"] = vm.annualize_vol(data["rv_range"], TRADING_DAYS, h_primary)
    rv_daily_1d = vd.realized_vol_close(simple_ret, 1)
    rv_daily_ann = vm.annualize_vol(rv_daily_1d, TRADING_DAYS, 1)

    # Baselines
    naive_pred = vm.naive_vol_forecast(rv_daily_ann, h_primary)
    roll_mean_pred = vm.rolling_mean_vol_forecast(rv_daily_ann, window=22, horizon_days=h_primary)

    # HAR-RV
    print("Fitting HAR-RV...")
    har_pred = vm.har_rv_rolling_forecast(rv_daily_ann, window=window, step=step, horizon_days=h_primary)

    # EWMA
    print("Fitting EWMA...")
    ewma_pred = vm.ewma_volatility(simple_ret, lam=0.94, burn_in=20)

    # GARCH family
    print("Fitting GARCH(1,1)...")
    garch_pred = vm.garch_rolling_forecast(simple_ret, window=window, step=step).ffill()
    print("Fitting GJR-GARCH...")
    gjr_pred = vm.gjr_garch_rolling_forecast(simple_ret, window=window, step=step).ffill()
    print("Fitting GARCH-t...")
    garch_t_pred = vm.garch_studentt_rolling_forecast(simple_ret, window=window, step=step).ffill()

    # Ridge features and forecast
    feat_cols = vm.build_volatility_features(
        data, simple_ret,
        include_range_vol=use_range_vol and "High" in data.columns,
        include_har_style=True,
        include_higher_moments=True,
    )
    if run_ridge_tuned:
        print("Fitting Ridge (tuned)...")
        ridge_pred = vm.ridge_rolling_forecast_tuned(
            data, feat_cols, "roll_vol_fwd_ann", window=window, step=step,
        ).clip(lower=0)
    else:
        print("Fitting Ridge...")
        ridge_pred = vm.ridge_rolling_forecast(
            data, feat_cols, "roll_vol_fwd_ann", window=window, step=step, alpha=1.0,
        ).clip(lower=0).ffill()

    # Align all to same index
    models = {
        "naive": naive_pred,
        "roll_mean": roll_mean_pred,
        "har_rv": har_pred,
        "ewma": ewma_pred,
        "garch": garch_pred,
        "gjr": gjr_pred,
        "garch_t": garch_t_pred,
        "ridge": ridge_pred,
    }

    eval_data = pd.DataFrame({"realized": rv_fwd_ann}, index=rv_fwd_ann.index)
    for k, v in models.items():
        eval_data[k] = v.reindex(eval_data.index).ffill()
    eval_data = eval_data.dropna(how="all", subset=list(models.keys())).dropna(subset=["realized"])

    signals = pd.DataFrame(index=data.index)
    signals.index.name = "Date"
    signals["Close"] = data["Close"]
    signals["Simple"] = data["Simple"]
    for k, v in models.items():
        signals[k] = v.reindex(signals.index).ffill()

    # Export signals to date-organized directory
    from datetime import date
    signals_dir = vpaths.get_signals_dir(date.today().isoformat())
    signals_path = signals_dir / "volatility_signals.csv"
    signals.to_csv(signals_path)
    print(f"Exported volatility_signals.csv to {signals_path}")

    # Regime
    regime = vm.volatility_regime(eval_data["realized"], 0.33, 0.67)
    eval_data["regime"] = regime

    # Full metrics (primary horizon)
    y = eval_data["realized"].values
    summary = []
    for name in models:
        yh = eval_data[name].values
        m = ve.volatility_metrics_full(y, yh)
        m["Model"] = name
        summary.append(m)
    summary_df = pd.DataFrame(summary).set_index("Model")

    # Diebold–Mariano (vs EWMA as benchmark)
    dm_results = {}
    e_bench = y - eval_data["ewma"].values
    for name in models:
        if name == "ewma":
            continue
        e_mod = y - eval_data[name].values
        dm_stat, dm_p = ve.diebold_mariano(e_bench, e_mod, loss="squared", h=h_primary)
        dm_results[f"DM_p_vs_EWMA_{name}"] = dm_p

    # By regime
    by_regime = ve.metrics_by_regime(
        y, eval_data["ridge"].values, eval_data["regime"].values,
        metrics=["MAE", "RMSE", "QLIKE"],
    )

    # Prediction intervals (Ridge, bootstrap)
    lower, upper = ve.prediction_interval_bootstrap(
        y, eval_data["ridge"].values, n_bootstrap=200, alpha=0.1,
    )
    eval_data["ridge_lower"] = lower
    eval_data["ridge_upper"] = upper

    extras = {"dm": dm_results, "by_regime": by_regime, "signals": signals}
    if run_backtest:
        bt = vbt.volatility_targeting_backtest(
            simple_ret.reindex(eval_data.index).ffill(),
            eval_data["ridge"].clip(lower=1e-8),
            target_vol_ann=0.10,
            trading_days=TRADING_DAYS,
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
        )
        extras["backtest"] = vbt.vol_targeting_economic_summary(bt, target_vol_ann=0.10)
        # VaR/ES + full validation (Christoffersen, DQ, FZ)
        vol_for_var = eval_data["ridge"].clip(lower=1e-8)
        var_series, es_series, params, pit = vdist.density_forecast_pipeline(
            simple_ret.reindex(eval_data.index).ffill(), vol_for_var, dist="studentt",
        )
        ret_align = simple_ret.reindex(var_series.dropna().index).ffill()
        var_align = var_series.reindex(ret_align.index).ffill()
        es_align = es_series.reindex(ret_align.index).ffill()
        extras["var_es_report"] = vr.var_es_validation_report(
            ret_align.values, var_align.values, es_align.values, alpha=0.01,
        )
        extras["pit"] = pit

    return summary_df, eval_data, by_regime, extras


def _plot_to_base64(ticker: str, eval_data: pd.DataFrame) -> str:
    """Build forecast vs realized plot, return base64-encoded PNG for embedding in HTML."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_data = eval_data.tail(252 * 2)
    if len(plot_data) == 0:
        plot_data = eval_data
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(plot_data.index, plot_data["realized"], label="Realized", color="black", alpha=0.8)
    for col in ["ewma", "garch", "ridge", "har_rv"]:
        if col in plot_data.columns:
            ax.plot(plot_data.index, plot_data[col], label=col, alpha=0.8)
    if "ridge_lower" in plot_data.columns:
        ax.fill_between(
            plot_data.index,
            plot_data["ridge_lower"],
            plot_data["ridge_upper"],
            alpha=0.2,
            label="Ridge 80% PI",
        )
    ax.set_title(f"{ticker} — Volatility forecasts vs realized")
    ax.legend(loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _write_html_report(
    ticker: str,
    start: str,
    end: str,
    summary_df: pd.DataFrame,
    eval_data: pd.DataFrame,
    by_regime: pd.DataFrame,
    extras: dict,
) -> Path:
    """Generate HTML report: headline table, plot, metrics, regime breakdown, backtest, DM. Returns path to file."""
    table_html = summary_df.round(6).to_html(classes="metrics-table")
    regime_html = by_regime.round(6).to_html(classes="regime-table") if by_regime is not None and len(by_regime) else "<p>No regime breakdown.</p>"
    dm_html = ""
    if "dm" in extras:
        dm_html = "<h3>Diebold–Mariano (p-values vs EWMA)</h3><ul>"
        for k, v in extras["dm"].items():
            dm_html += f"<li>{k}: {v:.4f}</li>"
        dm_html += "</ul>"
    backtest_html = ""
    if "backtest" in extras:
        bt = extras["backtest"]
        backtest_html = "<h3>Vol-targeting backtest (10% target)</h3><ul>"
        for k, v in bt.items():
            backtest_html += f"<li>{k}: {v:.4f}</li>"
        backtest_html += "</ul>"
    if "var_es_report" in extras:
        rep = extras["var_es_report"]
        backtest_html += "<h4>VaR/ES validation</h4><ul>"
        backtest_html += f"<li>Kupiec p-value: {rep.get('kupiec_pval', np.nan):.4f}</li>"
        backtest_html += f"<li>Christoffersen CC p-value: {rep.get('christoffersen_cc_pval', np.nan):.4f}</li>"
        backtest_html += f"<li>DQ p-value: {rep.get('dq_pval', np.nan):.4f}</li>"
        backtest_html += f"<li>FZ score: {rep.get('fz_score', np.nan):.6f}</li></ul>"

    plot_b64 = _plot_to_base64(ticker, eval_data)
    plot_img = f'<img src="data:image/png;base64,{plot_b64}" alt="Volatility forecast plot" style="max-width:100%; height:auto;">'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Volatility Report — {ticker}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; }}
    h1 {{ font-size: 1.5rem; }}
    h2, h3 {{ margin-top: 1.5rem; }}
    .meta {{ color: #555; margin-bottom: 1.5rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th, td {{ border: 1px solid #ddd; padding: 0.5rem 0.75rem; text-align: right; }}
    th {{ background: #f5f5f5; text-align: left; }}
    tr:nth-child(even) {{ background: #fafafa; }}
    .plot-wrap {{ margin: 1.5rem 0; }}
  </style>
</head>
<body>
  <h1>Volatility forecast comparison</h1>
  <p class="meta"><strong>{ticker}</strong> &middot; {start} to {end} &middot; {len(eval_data)} evaluation points</p>
  <p>Metrics: MAE, RMSE, Correlation, QLIKE, MSE_var, Log_vol. Lower is better except Correlation.</p>
  <h2>Forecast vs realized</h2>
  <div class="plot-wrap">{plot_img}</div>
  <h2>Headline table</h2>
  {table_html}
  <h2>Performance by regime</h2>
  {regime_html}
  {dm_html}
  {backtest_html}
  <p class="meta" style="margin-top: 2rem;">Generated by Stock Volatility Research pipeline.</p>
</body>
</html>"""
    from datetime import date
    reports_dir = vpaths.get_reports_dir(date.today().isoformat())
    out_path = reports_dir / f"volatility_report_{ticker}.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nReport written to {out_path}")
    return out_path


def main() -> None:
    # Support both: volforecast --ticker SPY  and  volforecast run --ticker SPY
    argv = list(sys.argv[1:])
    if argv and argv[0] == "run":
        argv = argv[1:]
    parser = argparse.ArgumentParser(description="Stock volatility research pipeline")
    parser.add_argument("--ticker", default=TICKER, help=f"Symbol (default: {TICKER})")
    parser.add_argument("--start", default=START, help=f"Start date (default: {START})")
    parser.add_argument("--end", default=END, help=f"End date (default: {END})")
    parser.add_argument("--horizons", type=str, default="1,5,21", help="Comma-separated horizons (default: 1,5,21)")
    parser.add_argument("--plot", action="store_true", help="Show comparison plot")
    parser.add_argument("--export", action="store_true", help="Export CSVs")
    parser.add_argument("--report", action="store_true", help="Generate HTML report")
    parser.add_argument("--backtest", action="store_true", help="Run vol-targeting backtest and VaR/Kupiec")
    parser.add_argument("--ridge-tuned", action="store_true", help="Use Ridge with walk-forward alpha tuning")
    parser.add_argument("--no-cache", action="store_true", help="Disable data cache")
    args = parser.parse_args(argv)

    horizons = [int(x) for x in args.horizons.split(",")]
    use_cache = not args.no_cache

    summary_df, eval_data, by_regime, extras = run_pipeline(
        args.ticker, args.start, args.end,
        horizons=horizons,
        window=WINDOW,
        step=STEP,
        use_range_vol=True,
        run_backtest=args.backtest,
        run_ridge_tuned=args.ridge_tuned,
    )

    print(f"\nVolatility forecast comparison — {args.ticker} ({args.start} to {args.end})")
    print(summary_df.round(6))
    if by_regime is not None and len(by_regime):
        print("\nBy regime:")
        print(by_regime.round(6))
    if "dm" in extras:
        print("\nDiebold–Mariano p-values vs EWMA:")
        for k, v in extras["dm"].items():
            print(f"  {k}: {v:.4f}")
    if "backtest" in extras:
        print("\nBacktest:", extras["backtest"])
    if "var_es_report" in extras:
        r = extras["var_es_report"]
        print("\nVaR/ES validation: Kupiec p=", r.get("kupiec_pval"), ", Christoffersen CC p=", r.get("christoffersen_cc_pval"), ", DQ p=", r.get("dq_pval"))

    if args.export:
        from datetime import date
        today_str = date.today().isoformat()
        metrics_dir = vpaths.get_metrics_dir(today_str)
        signals_dir = vpaths.get_signals_dir(today_str)
        
        summary_df.to_csv(metrics_dir / f"{args.ticker}_volatility_metrics.csv")
        eval_data.to_csv(metrics_dir / f"{args.ticker}_volatility_forecasts.csv")
        extras["signals"].to_csv(signals_dir / "volatility_signals.csv")
        if by_regime is not None:
            by_regime.to_csv(metrics_dir / f"{args.ticker}_volatility_by_regime.csv")
        print(f"\nExported to artifacts/ (date: {today_str})")
        print(f"  - Metrics: {metrics_dir}")
        print(f"  - Signals: {signals_dir}")

    if args.report:
        report_path = _write_html_report(
            args.ticker, args.start, args.end, summary_df, eval_data, by_regime, extras
        )
        webbrowser.open(report_path.as_uri())

    if args.plot:
        import matplotlib.pyplot as plt
        plot_data = eval_data.tail(252 * 2)  # last ~2 years
        if len(plot_data) == 0:
            plot_data = eval_data
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(plot_data.index, plot_data["realized"], label="Realized", color="black", alpha=0.8)
        for col in ["ewma", "garch", "ridge", "har_rv"]:
            if col in plot_data.columns:
                ax.plot(plot_data.index, plot_data[col], label=col, alpha=0.8)
        if "ridge_lower" in plot_data.columns:
            ax.fill_between(
                plot_data.index,
                plot_data["ridge_lower"],
                plot_data["ridge_upper"],
                alpha=0.2,
                label="Ridge 80% PI",
            )
        ax.set_title(f"{args.ticker} — Volatility forecasts vs realized")
        ax.legend(loc="best", ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
