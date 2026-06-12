"""Paired per-day top-weighted slate ranking metric — the swing campaign's
PRIMARY confirmation metric (spec 2026-06-12, Codex-resolved hierarchy).

daily_ndcg: NDCG@k with the standard log2 discount over one day's ranked
slate, binary game-level got-a-hit labels. paired_daily_delta: candidate-vs-
baseline as paired per-day deltas with a season-stratified day-level block
bootstrap (days resampled within season; PA-level independence never assumed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_ndcg(day: pd.DataFrame, score_col: str, k: int = 10,
               label_col: str = "actual_hit") -> float:
    """NDCG@k for a single day's slate (binary labels, log2 discount)."""
    d = day.dropna(subset=[score_col])
    if d.empty or d[label_col].sum() == 0:
        return np.nan
    order = d.sort_values(score_col, ascending=False, kind="mergesort")
    labels = order[label_col].to_numpy(dtype=float)[:k]
    discounts = 1.0 / np.log2(np.arange(2, len(labels) + 2))
    dcg = float((labels * discounts).sum())
    ideal = np.sort(d[label_col].to_numpy(dtype=float))[::-1][:k]
    idcg = float((ideal * discounts[: len(ideal)]).sum())
    return dcg / idcg if idcg > 0 else np.nan


def paired_daily_delta(
    slate: pd.DataFrame,
    score_a: str,
    score_b: str,
    k: int = 10,
    n_boot: int = 10_000,
    seed: int = 20260612,
    label_col: str = "actual_hit",
) -> dict:
    """Paired per-day NDCG@k delta (A − B) with season-stratified bootstrap.

    slate needs columns: date, season, {score_a}, {score_b}, {label_col}.
    Returns {delta, ci_low, ci_high, n_days, per_season_delta}.
    """
    per_day = []
    for (season, _date), day in slate.groupby(["season", "date"]):
        a = daily_ndcg(day, score_a, k=k, label_col=label_col)
        b = daily_ndcg(day, score_b, k=k, label_col=label_col)
        if not (np.isnan(a) or np.isnan(b)):
            per_day.append((season, a - b))
    if not per_day:
        return {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "n_days": 0, "per_season_delta": {}}
    df = pd.DataFrame(per_day, columns=["season", "d"])
    rng = np.random.default_rng(seed)
    by_season = {s: g["d"].to_numpy() for s, g in df.groupby("season")}
    boots = np.empty(n_boot)
    for i in range(n_boot):
        parts = [g[rng.integers(0, len(g), len(g))] for g in by_season.values()]
        boots[i] = float(np.concatenate(parts).mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "delta": float(df["d"].mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_days": int(len(df)),
        "per_season_delta": {int(s): float(g.mean()) for s, g in df.groupby("season")["d"]},
    }
