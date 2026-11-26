"""Tests for the paired daily NDCG@10 slate-ranking metric (campaign primary)."""
import numpy as np
import pandas as pd

from bts.validate.slate_rank import daily_ndcg, paired_daily_delta


def _slate(date, scores, hits):
    return pd.DataFrame({
        "date": pd.to_datetime([date] * len(scores)),
        "score": scores,
        "actual_hit": hits,
    })


def test_perfect_ranking_is_1():
    s = _slate("2025-06-01", [0.9, 0.8, 0.7, 0.6], [1, 1, 0, 0])
    assert daily_ndcg(s, "score", k=10) == 1.0


def test_worst_ranking_below_1():
    s = _slate("2025-06-01", [0.9, 0.8, 0.7, 0.6], [0, 0, 1, 1])
    assert daily_ndcg(s, "score", k=10) < 1.0


def test_k_truncation_ignores_tail():
    # 12 candidates; hits beyond rank 10 don't affect DCG but do affect ideal
    scores = list(np.linspace(0.9, 0.4, 12))
    hits = [1] + [0] * 10 + [1]
    s = _slate("2025-06-01", scores, hits)
    v = daily_ndcg(s, "score", k=10)
    assert 0 < v < 1.0


def test_paired_delta_zero_for_identical_scores():
    days = []
    rng = np.random.default_rng(7)
    for i in range(20):
        scores = rng.random(15)
        hits = (rng.random(15) < 0.65).astype(int)
        d = _slate(f"2025-06-{i+1:02d}", scores, hits)
        d["score_b"] = d["score"]
        d["season"] = 2025
        days.append(d)
    slate = pd.concat(days)
    res = paired_daily_delta(slate, "score", "score_b", k=10, n_boot=200, seed=1)
    assert res["delta"] == 0.0
    assert res["ci_low"] == 0.0 and res["ci_high"] == 0.0
    assert res["n_days"] == 20


def test_paired_delta_detects_better_ranker():
    days = []
    rng = np.random.default_rng(7)
    for i in range(60):
        n = 20
        hits = (rng.random(n) < 0.65).astype(int)
        good = hits * 1.0 + rng.normal(0, 0.3, n)   # correlated with outcome
        bad = rng.random(n)                          # noise
        d = pd.DataFrame({
            "date": pd.to_datetime(["2025-06-01"] * n) + pd.Timedelta(days=i),
            "score": good, "score_b": bad, "actual_hit": hits, "season": 2025,
        })
        days.append(d)
    slate = pd.concat(days)
    res = paired_daily_delta(slate, "score", "score_b", k=10, n_boot=500, seed=1)
    assert res["delta"] > 0
    assert res["ci_low"] > 0  # clearly separated


def test_bootstrap_stratifies_by_season():
    days = []
    rng = np.random.default_rng(7)
    for i in range(10):
        for season in (2025, 2026):
            n = 12
            hits = (rng.random(n) < 0.65).astype(int)
            d = pd.DataFrame({
                "date": pd.to_datetime(f"{season}-06-01") + pd.Timedelta(days=i),
                "score": rng.random(n), "score_b": rng.random(n),
                "actual_hit": hits, "season": season,
            })
            days.append(d)
    slate = pd.concat(days)
    res = paired_daily_delta(slate, "score", "score_b", k=10, n_boot=100, seed=1)
    assert res["n_days"] == 20
    assert set(res["per_season_delta"]) == {2025, 2026}
