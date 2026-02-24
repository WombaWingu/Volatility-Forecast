"""
Streamlit dashboard: ticker → models → plots → backtest → risk tests.
Reads latest from artifacts/ or runs pipeline on demand.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root and scripts are on path (for mini_proj and volatility_*)
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
src_dir = _project_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
scripts_dir = _project_root / "scripts"
if scripts_dir.exists() and str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from volforecast import paths as vpaths

st.set_page_config(page_title="Volatility Forecast", page_icon="📈", layout="wide")
st.title("📈 Volatility Forecast Dashboard")

# Option: load latest from artifacts or run pipeline
use_artifacts = st.sidebar.checkbox("Use latest artifacts (signals/metrics)", value=False)

signals_file = None
metrics_dir = None
if vpaths and use_artifacts:
    # Find latest signals date
    signals_base = vpaths.SIGNALS_DIR
    date_dirs = sorted([d for d in signals_base.iterdir() if d.is_dir()], reverse=True) if signals_base.exists() else []
    if date_dirs:
        latest_date = date_dirs[0].name
        st.sidebar.info(f"Latest artifacts: {latest_date}")
        signals_file = date_dirs[0] / "volatility_signals.csv"
        metrics_dir = vpaths.get_metrics_dir(latest_date)
        reports_dir = vpaths.get_reports_dir(latest_date)
    else:
        latest_date = None
        st.sidebar.warning("No artifacts found. Run pipeline or daily first.")
else:
    latest_date = None

ticker = st.sidebar.text_input("Ticker", value="NVDA")
start = st.sidebar.text_input("Start date", value="2020-01-01")
end = st.sidebar.text_input("End date", value="2026-02-12")
run_backtest = st.sidebar.checkbox("Run vol-targeting backtest", value=True)
run_var_es = st.sidebar.checkbox("Run VaR/ES validation", value=True)
cost_bps = st.sidebar.number_input("Transaction cost (bps)", min_value=0, value=5)
slippage_bps = st.sidebar.number_input("Slippage (bps)", min_value=0, value=2)

if use_artifacts and latest_date and signals_file is not None and signals_file.exists():
    # Load from artifacts
    if st.sidebar.button("Load from artifacts"):
        try:
            signals_df = pd.read_csv(signals_file, index_col=0, parse_dates=True)
            metrics_file = metrics_dir / f"{ticker}_volatility_metrics.csv"
            if metrics_file.exists():
                summary = pd.read_csv(metrics_file, index_col=0)
            else:
                summary = pd.DataFrame()
            eval_file = metrics_dir / f"{ticker}_volatility_forecasts.csv"
            if eval_file.exists():
                eval_data = pd.read_csv(eval_file, index_col=0, parse_dates=True)
            else:
                eval_data = pd.DataFrame()
            report_file = reports_dir / f"volatility_report_{ticker}.html"
            st.session_state["summary"] = summary
            st.session_state["eval_data"] = eval_data
            st.session_state["by_regime"] = None
            st.session_state["extras"] = {}
            st.session_state["ticker"] = ticker
            st.session_state["ready"] = True
            st.session_state["from_artifacts"] = True
        except Exception as e:
            st.error(f"Failed to load artifacts: {e}")
            st.session_state["ready"] = False

from mini_proj import load_prices, run_pipeline, HORIZONS, TRADING_DAYS, WINDOW, STEP
from volforecast import backtest as vbt
from volforecast import distributions as vdist
from volforecast import risk as vr

if st.sidebar.button("Run pipeline"):
    with st.spinner("Running pipeline..."):
        try:
            summary_df, eval_data, by_regime, extras = run_pipeline(
                ticker, start, end,
                horizons=HORIZONS, window=WINDOW, step=STEP,
                use_range_vol=True, run_backtest=run_backtest, run_ridge_tuned=False,
                cost_bps=float(cost_bps), slippage_bps=float(slippage_bps),
            )
            st.session_state["summary"] = summary_df
            st.session_state["eval_data"] = eval_data
            st.session_state["by_regime"] = by_regime
            st.session_state["extras"] = extras
            st.session_state["ticker"] = ticker
            st.session_state["ready"] = True
            st.session_state["from_artifacts"] = False
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.session_state["ready"] = False

if st.session_state.get("ready", False):
    summary = st.session_state["summary"]
    eval_data = st.session_state["eval_data"]
    by_regime = st.session_state["by_regime"]
    extras = st.session_state["extras"]
    ticker = st.session_state["ticker"]
    from_artifacts = st.session_state.get("from_artifacts", False)

    if not summary.empty:
        st.subheader(f"{ticker} — Headline metrics")
        st.dataframe(summary.round(6), use_container_width=True)

    st.subheader("Forecast vs realized")
    if eval_data.empty:
        st.warning("No evaluation data to plot. Run pipeline for this ticker or choose a ticker that has artifacts.")
    else:
        plot_data = eval_data.tail(252 * 2)
        plot_cols = ["realized"] + [c for c in ["ewma", "garch", "ridge", "har_rv"] if c in plot_data.columns]
        if plot_cols:
            st.line_chart(
                plot_data[plot_cols].rename(
                    columns={"realized": "Realized", "ewma": "EWMA", "garch": "GARCH", "ridge": "Ridge", "har_rv": "HAR-RV"}
                )
            )

    if by_regime is not None and len(by_regime) > 0:
        st.subheader("Performance by regime")
        st.dataframe(by_regime.round(6))

    if run_backtest and "backtest" in extras:
        st.subheader("Vol-targeting backtest (10% target)")
        bt = extras["backtest"]
        st.metric("Sharpe", f"{bt.get('sharpe', 0):.4f}")
        st.metric("Max drawdown", f"{bt.get('max_dd', 0):.4f}")
        st.metric("Turnover", f"{bt.get('turnover', 0):.4f}")

    if run_var_es and not from_artifacts and not eval_data.empty:
        st.subheader("VaR/ES validation")
        vol_for_var = eval_data["ridge"].clip(lower=1e-8) if "ridge" in eval_data.columns else eval_data["ewma"].clip(lower=1e-8)
        ret = load_prices(ticker, start, end)["Simple"].reindex(eval_data.index).ffill()
        var_s, es_s, params, pit = vdist.density_forecast_pipeline(ret, vol_for_var, dist="studentt")
        var_align = var_s.reindex(ret.dropna().index).ffill()
        es_align = es_s.reindex(ret.dropna().index).ffill()
        ret_align = ret.dropna()
        common = var_align.dropna().index.intersection(ret_align.index)
        report = vr.var_es_validation_report(
            ret_align.loc[common].values,
            var_align.loc[common].values,
            es_align.loc[common].values,
            alpha=0.01,
        )
        st.write("Kupiec p-value:", f"{report.get('kupiec_pval', 0):.4f}")
        st.write("Christoffersen CC p-value:", f"{report.get('christoffersen_cc_pval', 0):.4f}")
        st.write("DQ p-value:", f"{report.get('dq_pval', 0):.4f}")
        st.write("FZ score:", f"{report.get('fz_score', 0):.6f}")

else:
    st.info("Click **Run pipeline** to run the forecast, or check **Use latest artifacts** and click **Load from artifacts** to view the latest run.")
