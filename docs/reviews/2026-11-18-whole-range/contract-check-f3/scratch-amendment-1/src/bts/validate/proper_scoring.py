"""Probabilistic forecast evaluation suite — proper scoring rules + calibration.

SOTA tracker item #12 phase 1. Implements log loss, Brier score, Murphy
decomposition, reliability table, top-bin calibration, and a high-level
entry point that runs all metrics across the `all_top10` and `rank1`
decision buckets on backtest profile rows.

Game/profile-level only. Inputs are `p_game_hit` and `actual_hit` columns
from backtest profile parquet files (top-10 candidates per date).

References:
- Gneiting & Raftery 2007 — strictly proper scoring rules
- Murphy 1973 — Brier decomposition into reliability/resolution/uncertainty

The Murphy identity Brier = reliability - resolution + uncertainty is
exact only when probabilities are constant within each bin. With
continuous probabilities and quantile/fixed bins, a within-bin variance
residual remains; we expose it as `recomposition_error`.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def binary_log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-15) -> float:
    """Mean binary log loss with clipping at [eps, 1-eps].

    -mean(y log p + (1-y) log(1-p))

    Clipping prevents infinite loss when p=0 or p=1 with the wrong y.
    """
    p_arr = np.asarray(p, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    p_clipped = np.clip(p_arr, eps, 1.0 - eps)
    losses = -(y_arr * np.log(p_clipped) + (1.0 - y_arr) * np.log(1.0 - p_clipped))
    return float(losses.mean())


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error: mean((p - y)^2)."""
    p_arr = np.asarray(p, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    return float(np.mean((p_arr - y_arr) ** 2))


def _bin_assignments(p: np.ndarray, n_bins: int, binning: str) -> np.ndarray:
    """Assign each p to a bin. Returns 0-indexed bin_idx with the same length as p.

    binning="quantile" produces ~equal-count bins via numpy.quantile, with
    duplicate edges collapsed (clusters of identical p produce fewer bins).
    binning="fixed" produces equal-width bins on [0, 1].

    Degenerate case (all p identical → 0 or 1 unique edge): every input
    is assigned to bin 0. Callers report p_lo/p_hi as observed min/max
    within the resulting single bin.
    """
    if binning == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(p, quantiles)
        edges = np.unique(edges)
        if len(edges) < 2:
            # All p identical → single bin; everything maps to bin 0.
            return np.zeros(len(p), dtype=int)
    elif binning == "fixed":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(f"binning must be 'quantile' or 'fixed', got {binning!r}")

    # np.digitize on internal edges (skip outer min/max) places p into 0..n_bins-1.
    bin_idx = np.digitize(p, edges[1:-1], right=False)
    bin_idx = np.clip(bin_idx, 0, len(edges) - 2)
    return bin_idx


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = successes / n
    denom = 1.0 + z**2 / n
    center = (p_hat + z**2 / (2.0 * n)) / denom
    half = z * math.sqrt((p_hat * (1.0 - p_hat) + z**2 / (4.0 * n)) / n) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi)


def reliability_table(
    p: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
    binning: str = "quantile",
) -> pd.DataFrame:
    """Per-bin reliability metrics with Wilson CIs on the observed rate.

    Columns: bin_idx, p_lo, p_hi, mean_p, mean_y, n, ci_lo, ci_hi.

    `p_lo` and `p_hi` are the observed min/max forecast probability
    within each bin (NOT the bin-assignment edges) — this avoids
    implying forecast ranges that were never observed.

    Empty bins are dropped.
    """
    p_arr = np.asarray(p, dtype=float)
    y_arr = np.asarray(y, dtype=int)
    bin_idx = _bin_assignments(p_arr, n_bins, binning)
    rows = []
    for k in np.unique(bin_idx):
        mask = bin_idx == k
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        successes = int(y_arr[mask].sum())
        p_in_bin = p_arr[mask]
        ci_lo, ci_hi = _wilson_ci(successes, n_k)
        rows.append({
            "bin_idx": int(k),
            "p_lo": float(p_in_bin.min()),
            "p_hi": float(p_in_bin.max()),
            "mean_p": float(p_in_bin.mean()),
            "mean_y": successes / n_k,
            "n": n_k,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        })
    return pd.DataFrame(rows)


def murphy_decomposition(
    p: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
    binning: str = "quantile",
) -> dict:
    """Murphy 1973 Brier decomposition into reliability/resolution/uncertainty.

    Returns a dict with:
    - reliability: sum_k (n_k/N) * (mean_p_k - mean_y_k)^2
    - resolution: sum_k (n_k/N) * (mean_y_k - mean_y)^2
    - uncertainty: mean_y * (1 - mean_y)
    - brier: mean((p - y)^2) computed directly
    - recomposition_error: brier - (reliability - resolution + uncertainty),
      reflecting the within-bin variance residual when p is non-constant
      within bins. Zero for discrete forecasts; small but nonzero for
      continuous forecasts.
    - n_bins, binning: metadata.
    """
    p_arr = np.asarray(p, dtype=float)
    y_arr = np.asarray(y, dtype=int)
    n = len(p_arr)
    bin_idx = _bin_assignments(p_arr, n_bins, binning)
    mean_y = float(y_arr.mean())

    reliability = 0.0
    resolution = 0.0
    n_used_bins = 0
    for k in np.unique(bin_idx):
        mask = bin_idx == k
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        n_used_bins += 1
        mean_p_k = float(p_arr[mask].mean())
        mean_y_k = float(y_arr[mask].mean())
        weight = n_k / n
        reliability += weight * (mean_p_k - mean_y_k) ** 2
        resolution += weight * (mean_y_k - mean_y) ** 2

    uncertainty = mean_y * (1.0 - mean_y)
    brier = brier_score(p_arr, y_arr)
    recomposition = brier - (reliability - resolution + uncertainty)

    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "brier": brier,
        "recomposition_error": abs(recomposition),
        "n_bins": n_used_bins,
        "binning": binning,
    }


def top_bin_calibration(
    p: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
    binning: str = "quantile",
) -> dict:
    """Calibration of the top (highest-p) bin only.

    Returns mean_p, mean_y, gap (mean_p - mean_y), n, ci_lo, ci_hi.
    """
    table = reliability_table(p, y, n_bins=n_bins, binning=binning)
    if len(table) == 0:
        return {
            "mean_p": float("nan"),
            "mean_y": float("nan"),
            "gap": float("nan"),
            "n": 0,
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
        }
    top = table.iloc[-1]
    return {
        "mean_p": float(top["mean_p"]),
        "mean_y": float(top["mean_y"]),
        "gap": float(top["mean_p"] - top["mean_y"]),
        "n": int(top["n"]),
        "ci_lo": float(top["ci_lo"]),
        "ci_hi": float(top["ci_hi"]),
    }


def _bucket_metrics(
    df: pd.DataFrame,
    p_col: str,
    y_col: str,
    n_bins: int,
    binning: str,
) -> dict:
    """Compute all metrics for one decision bucket (subset of profiles)."""
    p = df[p_col].to_numpy()
    y = df[y_col].to_numpy()
    decomp = murphy_decomposition(p, y, n_bins=n_bins, binning=binning)
    rel_table = reliability_table(p, y, n_bins=n_bins, binning=binning)
    return {
        "n": int(len(df)),
        "log_loss": binary_log_loss(p, y),
        "brier": brier_score(p, y),
        "decomposition": decomp,
        "reliability_table": rel_table.to_dict(orient="records"),
        "top_bin": top_bin_calibration(p, y, n_bins=n_bins, binning=binning),
    }


def compute_proper_scoring(
    profiles_df: pd.DataFrame,
    p_col: str = "p_game_hit",
    y_col: str = "actual_hit",
    rank_col: str = "rank",
    n_bins: int = 10,
    binning: str = "quantile",
    interval_method: str = "wilson",
) -> dict:
    """High-level entry point: compute proper-scoring metrics over decision buckets.

    Decision buckets:
    - `all_top10`: every row in the profile DataFrame (top-10 per date)
    - `rank1`: only `rank == 1` rows (one selected top candidate per date)

    Returns a dict with each bucket's metrics plus a metadata block that
    records the interval method, bin count, binning method, and source
    column names — enough metadata to avoid overclaim downstream.
    """
    if interval_method != "wilson":
        raise NotImplementedError(
            f"interval_method={interval_method!r} not implemented; "
            "use 'wilson' or extend this function with bootstrap support"
        )

    rank1 = profiles_df[profiles_df[rank_col] == 1]
    metrics = {
        "all_top10": _bucket_metrics(profiles_df, p_col, y_col, n_bins, binning),
        "rank1": _bucket_metrics(rank1, p_col, y_col, n_bins, binning),
        "metadata": {
            "interval_method": interval_method,
            "n_bins": n_bins,
            "binning": binning,
            "p_col": p_col,
            "y_col": y_col,
            "rank_col": rank_col,
        },
    }
    return metrics
