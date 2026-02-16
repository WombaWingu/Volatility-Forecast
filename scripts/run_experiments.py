"""
Experiment runner: YAML/JSON config, save to /results/ with run id, reproducible.
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mini_proj import load_prices, run_pipeline, CACHE_DIR, HORIZONS, TRADING_DAYS, WINDOW, STEP
import volatility_backtest as vbt
import volatility_distributions as vdist
import volatility_eval as ve
import volatility_ensemble as vens
import volatility_models as vm
import volatility_risk as vr
import volatility_data as vd
import volatility_paths as vpaths

# Use artifacts/experiments/ for all experiment outputs
RESULTS_DIR = vpaths.EXPERIMENTS_DIR
CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
LEADERBOARD_CSV = vpaths.EXPERIMENTS_DIR / "leaderboard.csv"

def _json_safe(obj):
    """Convert config objects (date/datetime/Path/etc) into JSON-serializable forms."""
    # Primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    # Common non-JSON types
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    # YAML often parses ISO dates into datetime.date
    if hasattr(obj, "isoformat") and obj.__class__.__name__ == "date":
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    # Containers
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    # Fallback
    return str(obj)


def load_config(path: str | Path) -> dict:
    """Load YAML or JSON config."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(p, encoding="utf-8") as f:
        if p.suffix.lower() in (".yaml", ".yml"):
            return yaml.safe_load(f)
        return json.load(f)


def run_one_ticker_full(
    ticker: str,
    config: dict,
    run_id: str,
) -> dict:
    """Run full pipeline for one ticker including VaR/ES, density, economic value."""
    # Allow end: null in YAML to mean "today"
    if config.get("end", None) is None:
        config = dict(config)
        config["end"] = datetime.now().strftime("%Y-%m-%d")
    data = load_prices(
        ticker, config["start"], config["end"],
        use_cache=not config.get("no_cache", False),
    )
    simple_ret = data["Simple"].dropna()
    data = data.loc[simple_ret.index].copy()
    h = config["horizons"][0]
    ann_factor = np.sqrt(TRADING_DAYS / h)
    rv_fwd = vd.forward_realized_vol(simple_ret, h, TRADING_DAYS)
    rv_daily = vm.annualize_vol(vd.realized_vol_close(simple_ret, 1), TRADING_DAYS, 1)
    data["roll_vol_fwd_ann"] = rv_fwd

    models_req = config.get("models", ["naive", "roll_mean", "har_rv", "ewma", "garch", "ridge"])
    window = config.get("window", WINDOW)
    step = config.get("step", STEP)

    forecasts = {}
    if "naive" in models_req:
        forecasts["naive"] = vm.naive_vol_forecast(rv_daily, h)
    if "roll_mean" in models_req:
        forecasts["roll_mean"] = vm.rolling_mean_vol_forecast(rv_daily, 22, h)
    if "har_rv" in models_req:
        forecasts["har_rv"] = vm.har_rv_rolling_forecast(rv_daily, window=window, step=step, horizon_days=h)
    if "ewma" in models_req:
        forecasts["ewma"] = vm.ewma_volatility(simple_ret, lam=0.94, burn_in=20)
    if "garch" in models_req:
        forecasts["garch"] = vm.garch_rolling_forecast(simple_ret, window=window, step=step).ffill()
    if "gjr" in models_req:
        forecasts["gjr"] = vm.gjr_garch_rolling_forecast(simple_ret, window=window, step=step).ffill()
    if "garch_t" in models_req:
        forecasts["garch_t"] = vm.garch_studentt_rolling_forecast(simple_ret, window=window, step=step).ffill()
    if "ridge" in models_req:
        feat_cols = vm.build_volatility_features(data, simple_ret, include_range_vol="High" in data.columns)
        forecasts["ridge"] = vm.ridge_rolling_forecast(
            data, feat_cols, "roll_vol_fwd_ann", window=window, step=step, alpha=1.0,
        ).clip(lower=0).ffill()
    if "gas" in models_req:
        try:
            forecasts["gas"] = vens.gas_volatility_rolling(simple_ret, window=window, step=step)
        except Exception:
            pass
    if "ensemble" in models_req and "ridge" in forecasts and "garch" in forecasts:
        try:
            forecasts["ensemble"] = vens.rank_weighted_ensemble(
                {"ridge": forecasts["ridge"], "garch": forecasts["garch"]},
                rv_fwd, window=min(252, window),
            )
        except Exception:
            pass

    eval_df = pd.DataFrame({"realized": rv_fwd}, index=rv_fwd.index)
    for k, v in forecasts.items():
        eval_df[k] = v.reindex(eval_df.index).ffill()
    eval_df = eval_df.dropna(how="all", subset=list(forecasts.keys())).dropna(subset=["realized"])

    regime = vm.volatility_regime(eval_df["realized"], 0.33, 0.67)
    eval_df["regime"] = regime

    y = eval_df["realized"].values
    metrics = []
    for name in forecasts:
        m = ve.volatility_metrics_full(y, eval_df[name].values)
        m["Model"] = name
        m["ticker"] = ticker
        metrics.append(m)
    summary = pd.DataFrame(metrics).set_index(["ticker", "Model"])

    dm_results = {}
    bench_name = "ewma" if "ewma" in forecasts else list(forecasts.keys())[0]
    e_bench = y - eval_df[bench_name].values
    for name in forecasts:
        if name == bench_name:
            continue
        e_mod = y - eval_df[name].values
        dm_stat, dm_p = ve.diebold_mariano(e_bench, e_mod, loss="squared", h=h)
        dm_results[f"DM_p_vs_{bench_name}_{name}"] = dm_p

    by_regime = ve.metrics_by_regime(
        y, eval_df["ridge"].values if "ridge" in eval_df.columns else eval_df[list(forecasts.keys())[0]].values,
        regime.values, metrics=["MAE", "RMSE", "QLIKE"],
    )

    result = {
        "ticker": ticker,
        "summary": summary,
        "eval_data": eval_df,
        "by_regime": by_regime,
        "dm": dm_results,
    }

    if config.get("run_backtest", False):
        vol_for_bt = eval_df["ridge"].clip(lower=1e-8) if "ridge" in eval_df.columns else eval_df[list(forecasts.keys())[0]].clip(lower=1e-8)
        ret_bt = simple_ret.reindex(eval_df.index).ffill()
        bt = vbt.volatility_targeting_backtest(
            ret_bt, vol_for_bt, target_vol_ann=0.10, trading_days=TRADING_DAYS,
            cost_bps=config.get("cost_bps", 0), slippage_bps=config.get("slippage_bps", 0),
        )
        result["backtest"] = vbt.vol_targeting_economic_summary(bt, target_vol_ann=0.10)
        result["backtest_raw"] = bt

    if config.get("run_var_es", False):
        vol_for_var = eval_df["ridge"].clip(lower=1e-8) if "ridge" in eval_df.columns else eval_df[list(forecasts.keys())[0]].clip(lower=1e-8)
        sigma_daily = vol_for_var / np.sqrt(TRADING_DAYS)
        var_s, es_s, params, pit = vdist.density_forecast_pipeline(
            simple_ret.reindex(eval_df.index).ffill(), vol_for_var, dist="studentt",
        )
        ret_align = simple_ret.reindex(var_s.dropna().index).ffill()
        var_align = var_s.reindex(ret_align.index).ffill()
        es_align = es_s.reindex(ret_align.index).ffill()
        result["var_es_report"] = vr.var_es_validation_report(
            ret_align.values, var_align.values, es_align.values, alpha=0.01,
        )
        result["pit"] = pit

    return result


def run_experiment(config_path: str | Path) -> Path:
    """Run full experiment, save to artifacts/experiments/<run_id>/."""
    config = load_config(config_path)
    np.random.seed(config.get("seed", 42))
    run_id = str(uuid.uuid4())[:8] + "_" + datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = vpaths.get_experiment_dir(run_id)

    # Accept either:
    # - tickers: ["SPY","QQQ"]  (or "SPY,QQQ")
    # - ticker: "NVDA"
    tickers = config.get("tickers", None)
    if tickers is None:
        single = config.get("ticker", None)
        if single is None:
            raise KeyError("Config must include either 'tickers' or 'ticker'.")
        tickers = [single]
    elif isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split(",") if t.strip()]

    all_summaries = []
    all_regimes = []
    for ticker in tickers:
        try:
            res = run_one_ticker_full(ticker, config, run_id)
            all_summaries.append(res["summary"])
            if res.get("by_regime") is not None and len(res["by_regime"]) > 0:
                res["by_regime"].index = [f"{ticker}_{x}" for x in res["by_regime"].index]
                all_regimes.append(res["by_regime"])
            res["eval_data"].to_csv(out_dir / f"{ticker}_eval.csv")
            res["summary"].to_csv(out_dir / f"{ticker}_metrics.csv")
            if "backtest" in res:
                pd.DataFrame([res["backtest"]]).to_csv(out_dir / f"{ticker}_backtest.csv")
            if "var_es_report" in res:
                pd.DataFrame([res["var_es_report"]]).to_csv(out_dir / f"{ticker}_var_es.csv")
        except Exception as e:
            print(f"Failed {ticker}: {e}")
            continue

    if all_summaries:
        full = pd.concat(all_summaries, axis=0)
        full.to_csv(out_dir / "leaderboard_metrics.csv")
    if all_regimes:
        pd.concat(all_regimes, axis=0).to_csv(out_dir / "by_regime.csv")

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(config), f, indent=2)
    with open(out_dir / "run_id.txt", "w", encoding="utf-8") as f:
        f.write(run_id)

    # Update master leaderboard: append this run's summary to leaderboard.csv
    if all_summaries:
        full = pd.concat(all_summaries, axis=0)
        # Average metrics by model across tickers for this run
        model_avg = full.groupby(level="Model").mean()
        vpaths.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        leaderboard_row = pd.DataFrame([{
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "tickers": ",".join(tickers),
            "run_path": str(out_dir),
        }])
        if LEADERBOARD_CSV.exists():
            existing = pd.read_csv(LEADERBOARD_CSV)
            leaderboard_row = pd.concat([existing, leaderboard_row], ignore_index=True)
        leaderboard_row.to_csv(LEADERBOARD_CSV, index=False)
        print(f"Leaderboard updated: {LEADERBOARD_CSV}")

    print(f"Results saved to {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Volatility forecast experiment runner")
    parser.add_argument("run", nargs="?", default="run", help="Subcommand (run)")
    parser.add_argument("--config", "-c", default=str(CONFIGS_DIR / "sp500_sample.yaml"), help="Config path")
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
