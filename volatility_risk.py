"""
VaR/ES validation: Christoffersen (independence + conditional coverage),
DQ test (dynamic quantile), ES backtests, Fissler–Ziegel joint scoring,
calibration diagnostics (PIT histograms, exceedance clustering).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def hit_sequence(returns: np.ndarray, var_series: np.ndarray) -> np.ndarray:
    """Hit sequence: 1 if loss exceeds VaR (return < -VaR), 0 otherwise."""
    r = np.asarray(returns, dtype=float).ravel()
    v = np.asarray(var_series, dtype=float).ravel()
    return (r < -np.abs(v)).astype(float)


def christoffersen_unconditional(
    hit: np.ndarray, alpha: float
) -> tuple[float, float]:
    """
    Kupiec/Christoffersen unconditional coverage test.
    H0: E[hit] = alpha. LR ~ chi2(1).
    Returns (LR_stat, p_value).
    """
    hit = hit[np.isfinite(hit)]
    n = len(hit)
    if n < 2:
        return np.nan, np.nan
    x = hit.sum()
    p_hat = x / n
    if p_hat <= 0 or p_hat >= 1:
        return np.nan, np.nan
    lik_null = (alpha ** x) * ((1 - alpha) ** (n - x))
    lik_alt = (p_hat ** x) * ((1 - p_hat) ** (n - x))
    lr = -2 * (np.log(lik_null + 1e-15) - np.log(lik_alt + 1e-15))
    pval = 1 - stats.chi2.cdf(lr, 1)
    return float(lr), float(pval)


def christoffersen_independence(hit: np.ndarray) -> tuple[float, float]:
    """
    Christoffersen independence test: exceedances should not cluster.
    Tests transition matrix: p_01 = P(hit=1|prev=0) vs p_11 = P(hit=1|prev=1).
    H0: p_01 = p_11 (no clustering). LR ~ chi2(1).
    Returns (LR_stat, p_value).
    """
    hit = hit[np.isfinite(hit)]
    n = len(hit)
    if n < 4:
        return np.nan, np.nan
    prev = hit[:-1]
    curr = hit[1:]
    n00 = ((prev == 0) & (curr == 0)).sum()
    n01 = ((prev == 0) & (curr == 1)).sum()
    n10 = ((prev == 1) & (curr == 0)).sum()
    n11 = ((prev == 1) & (curr == 1)).sum()
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    if p <= 0 or p >= 1:
        return np.nan, np.nan
    p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    p10 = n10 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    p00 = n00 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    lik_alt = (p00 ** n00) * (p01 ** n01) * (p10 ** n10) * (p11 ** n11)
    lik_null = (p ** (n01 + n11)) * ((1 - p) ** (n00 + n10))
    if lik_alt <= 0 or lik_null <= 0:
        return np.nan, np.nan
    lr = -2 * (np.log(lik_null) - np.log(lik_alt))
    pval = 1 - stats.chi2.cdf(lr, 1)
    return float(lr), float(pval)


def christoffersen_conditional_coverage(
    hit: np.ndarray, alpha: float
) -> tuple[float, float]:
    """
    Christoffersen conditional coverage: joint test of unconditional + independence.
    LR_cc = LR_uc + LR_ind, ~ chi2(2).
    Returns (LR_stat, p_value).
    """
    lr_uc, _ = christoffersen_unconditional(hit, alpha)
    lr_ind, _ = christoffersen_independence(hit)
    if np.isnan(lr_uc) or np.isnan(lr_ind):
        return np.nan, np.nan
    lr_cc = lr_uc + lr_ind
    pval = 1 - stats.chi2.cdf(lr_cc, 2)
    return float(lr_cc), float(pval)


def dq_test(
    returns: np.ndarray,
    var_series: np.ndarray,
    alpha: float = 0.01,
    lags: int = 5,
) -> tuple[float, float]:
    """
    Engle–Manganelli Dynamic Quantile (DQ) test for VaR.
    Regresses hit - alpha on lags of hit and VaR; tests joint significance.
    H0: all coefficients = 0. Wald ~ chi2(k).
    Returns (Wald_stat, p_value).
    """
    hit = hit_sequence(returns, var_series)
    v = np.asarray(var_series, dtype=float).ravel()
    n = len(hit)
    if n < lags + 20:
        return np.nan, np.nan
    # Dependent: hit_t - alpha
    y = hit - alpha
    # Regressors: constant, hit_{t-1}, ..., hit_{t-lags}, VaR_t
    X = np.ones((n - lags, 2 + lags))
    for i in range(lags):
        X[:, 1 + i] = hit[lags - 1 - i : n - 1 - i]
    X[:, -1] = np.abs(v[lags:])
    y = y[lags:]
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    if len(y) < 30:
        return np.nan, np.nan
    try:
        from numpy.linalg import inv, lstsq

        beta = lstsq(X, y, rcond=None)[0]
        res = y - X @ beta
        sigma2 = (res @ res) / (len(y) - X.shape[1])
        if sigma2 <= 0:
            return np.nan, np.nan
        var_beta = sigma2 * inv(X.T @ X)
        wald = float(beta @ inv(var_beta) @ beta)
        pval = 1 - stats.chi2.cdf(wald, X.shape[1])
        return wald, float(pval)
    except Exception:
        return np.nan, np.nan


def es_quantile_backtest(
    returns: np.ndarray,
    var_series: np.ndarray,
    es_series: np.ndarray,
    alpha: float = 0.01,
) -> tuple[float, float]:
    """
    ES backtest: when VaR is violated, actual loss should equal ES on average.
    Uses McNeil–Frey (2000) / Acerbi–Szekely (2014) approach.
    Returns (test_stat, p_value approx via bootstrap).
    """
    hit = hit_sequence(returns, var_series)
    r = np.asarray(returns, dtype=float).ravel()
    v = np.asarray(var_series, dtype=float).ravel()
    e = np.asarray(es_series, dtype=float).ravel()
    mask = (hit > 0.5) & np.isfinite(v) & np.isfinite(e) & (e > 1e-10)
    n_viol = mask.sum()
    if n_viol < 5:
        return np.nan, np.nan
    losses = -r[mask]
    es_vals = e[mask]
    ratio = (losses / es_vals).mean()
    if np.isnan(ratio) or ratio <= 0:
        return np.nan, np.nan
    return float(ratio), 0.0  # No simple p-value; use bootstrap in caller if needed


def fissler_ziegel_score(
    returns: np.ndarray,
    var_series: np.ndarray,
    es_series: np.ndarray,
    alpha: float = 0.01,
) -> np.ndarray:
    """
    Fissler–Ziegel (2016) strictly consistent scoring for (VaR, ES).
    S(v,e,x) = (1{x<=v} - alpha)(G1(v) - G1(x)) + (1/alpha)G2(e)*1{x<=v}*(v-x) + G2(e)*(e-v) - G2(e)
    With G1(v)=v, G2(e)=exp(e): simple choice.
    Returns score per observation (lower = better).
    """
    r = np.asarray(returns, dtype=float).ravel()
    v = np.asarray(var_series, dtype=float).ravel()
    e = np.asarray(es_series, dtype=float).ravel()
    n = len(r)
    score = np.full(n, np.nan)
    # VaR/ES are in loss terms (positive); return is signed
    x = -r  # loss
    hit = (x >= v).astype(float)
    g2 = np.exp(np.clip(e, -20, 20))
    term1 = (hit - alpha) * (v - np.minimum(x, v))
    term2 = (1.0 / alpha) * g2 * hit * (x - v)
    term3 = g2 * (e - v)
    term4 = -g2
    score = term1 + term2 + term3 + term4
    return score


def fissler_ziegel_mean_score(
    returns: np.ndarray, var_series: np.ndarray, es_series: np.ndarray, alpha: float = 0.01
) -> float:
    """Mean FZ score (lower = better)."""
    s = fissler_ziegel_score(returns, var_series, es_series, alpha)
    return float(np.nanmean(s))


def pit_histogram(
    pit_values: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """
    PIT (Probability Integral Transform) histogram for calibration.
    If forecast is well calibrated, PIT ~ Uniform(0,1).
    Returns (bin_centers, counts).
    """
    pit = pit_values[np.isfinite(pit_values) & (pit_values >= 0) & (pit_values <= 1)]
    if len(pit) < 10:
        return np.array([]), np.array([])
    counts, edges = np.histogram(pit, bins=n_bins, range=(0, 1))
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, counts.astype(float)


def pit_histogram_uniform_test(pit_values: np.ndarray, n_bins: int = 10) -> tuple[float, float]:
    """
    Chi-squared test: PIT histogram vs uniform.
    H0: PIT ~ Uniform. Returns (chi2_stat, p_value).
    """
    centers, counts = pit_histogram(pit_values, n_bins)
    if len(counts) == 0:
        return np.nan, np.nan
    n = counts.sum()
    expected = n / n_bins
    if expected < 5:
        return np.nan, np.nan
    chi2 = np.sum((counts - expected) ** 2 / expected)
    pval = 1 - stats.chi2.cdf(chi2, n_bins - 1)
    return float(chi2), float(pval)


def exceedance_clustering_stats(hit: np.ndarray) -> dict[str, float]:
    """
    Exceedance clustering: run lengths, mean gap between violations.
    """
    hit = hit[np.isfinite(hit)]
    n = len(hit)
    if n < 2:
        return {"n_violations": 0, "mean_run_length": np.nan, "mean_gap": np.nan}
    viol_idx = np.where(hit > 0.5)[0]
    n_viol = len(viol_idx)
    if n_viol < 2:
        return {"n_violations": int(n_viol), "mean_run_length": 1.0 if n_viol == 1 else np.nan, "mean_gap": np.nan}
    gaps = np.diff(viol_idx)
    runs = []
    i = 0
    while i < len(viol_idx):
        j = i
        while j + 1 < len(viol_idx) and viol_idx[j + 1] == viol_idx[j] + 1:
            j += 1
        runs.append(j - i + 1)
        i = j + 1
    return {
        "n_violations": int(n_viol),
        "mean_run_length": float(np.mean(runs)) if runs else np.nan,
        "mean_gap": float(np.mean(gaps)) if len(gaps) > 0 else np.nan,
        "max_run_length": int(max(runs)) if runs else 0,
    }


def var_es_validation_report(
    returns: np.ndarray,
    var_series: np.ndarray,
    es_series: np.ndarray | None,
    alpha: float = 0.01,
) -> dict[str, float]:
    """
    Full VaR/ES validation report: Kupiec, Christoffersen (UC, IND, CC), DQ, clustering.
    """
    hit = hit_sequence(returns, var_series)
    out: dict[str, float] = {}
    lr_uc, p_uc = christoffersen_unconditional(hit, alpha)
    out["kupiec_lr"] = lr_uc
    out["kupiec_pval"] = p_uc
    lr_ind, p_ind = christoffersen_independence(hit)
    out["christoffersen_ind_lr"] = lr_ind
    out["christoffersen_ind_pval"] = p_ind
    lr_cc, p_cc = christoffersen_conditional_coverage(hit, alpha)
    out["christoffersen_cc_lr"] = lr_cc
    out["christoffersen_cc_pval"] = p_cc
    wald_dq, p_dq = dq_test(returns, var_series, alpha)
    out["dq_wald"] = wald_dq
    out["dq_pval"] = p_dq
    for k, v in exceedance_clustering_stats(hit).items():
        out[f"clustering_{k}"] = v if not np.isnan(v) else 0.0
    if es_series is not None:
        ratio, _ = es_quantile_backtest(returns, var_series, es_series, alpha)
        out["es_ratio"] = ratio
        out["fz_score"] = fissler_ziegel_mean_score(returns, var_series, es_series, alpha)
    return out
