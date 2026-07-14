"""Tests for the stage-1 repeat-batter conditioning machinery."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.audit.repeat_batter_conditioning import add_repeat_flags, gap_contrast


def _frame(rows):
    return pd.DataFrame(
        rows, columns=["file_idx", "season", "seed", "date", "batter_id", "stated", "hit"]
    )


def test_add_repeat_flags_previous_slate_day_semantics():
    df = _frame(
        [
            # file 0: A, A, B, A, C -> repeat_1: F, T, F, F, F; repeat_3 catches d4's A
            (0, 2021, 7, "d1", "A", 0.78, 1),
            (0, 2021, 7, "d2", "A", 0.77, 1),
            (0, 2021, 7, "d3", "B", 0.76, 0),
            (0, 2021, 7, "d4", "A", 0.79, 1),
            (0, 2021, 7, "d5", "C", 0.75, 0),
            # file 1 starts fresh — no bleed across files even with same batter
            (1, 2021, 9, "d1", "C", 0.74, 1),
            (1, 2021, 9, "d2", "C", 0.76, 0),
        ]
    )
    out = add_repeat_flags(df, ks=(1, 3))
    f0 = out[out.file_idx == 0]
    assert list(f0["repeat_1"]) == [False, True, False, False, False]
    assert list(f0["repeat_3"]) == [False, True, False, True, False]
    f1 = out[out.file_idx == 1]
    assert list(f1["repeat_1"]) == [False, True]  # first day never a repeat


def test_add_repeat_flags_window_is_positional_not_calendar():
    # a gap in calendar dates does not matter: previous slate day = previous row
    df = _frame(
        [
            (0, 2021, 7, "2021-04-01", "A", 0.78, 1),
            (0, 2021, 7, "2021-04-09", "A", 0.77, 1),  # 8 calendar days later
        ]
    )
    out = add_repeat_flags(df, ks=(1,))
    assert list(out["repeat_1"]) == [False, True]


def test_gap_contrast_recovers_planted_overconfidence():
    # synthetic: repeats realize 8pp below stated, fresh picks calibrated;
    # 3 correlated "seeds" per date must not shrink the cluster CI
    rng = np.random.default_rng(11)
    rows = []
    # 1200 dates so the planted 8pp beats the cluster CI half-width (~5pp)
    for d in range(1200):
        repeat = d % 3 == 0  # same repeat status across seeds for the date
        p = 0.76
        true_p = p - 0.08 if repeat else p
        hit = float(rng.random() < true_p)  # shared outcome across seeds
        for seed in (1, 2, 3):
            rows.append((seed, 2021 + d % 5, seed, f"d{d:04d}", "A" if repeat else f"B{d}", p, hit))
    df = _frame(rows)
    df["repeat_1"] = df["batter_id"] == "A"
    c = gap_contrast(df, "repeat_1", n_boot=500, seed=3)
    # machinery exactness: the contrast equals the direct group computation
    rep, fresh = df[df["repeat_1"]], df[~df["repeat_1"]]
    direct = (rep["hit"].mean() - rep["stated"].mean()) - (
        fresh["hit"].mean() - fresh["stated"].mean()
    )
    assert c["diff"] == pytest.approx(direct, abs=1e-12)
    lo, hi = c["diff_ci95_cluster_boot"]
    assert lo < c["diff"] < hi
    assert hi < 0  # planted 8pp effect detected under this frozen draw
    # determinism
    again = gap_contrast(df, "repeat_1", n_boot=500, seed=3)
    assert again["diff_ci95_cluster_boot"] == c["diff_ci95_cluster_boot"]


def test_gap_contrast_invariant_to_within_cluster_duplication():
    # duplicating every row within its cluster must change NOTHING about the
    # cluster bootstrap (point estimate and CI identical) — a row-resampling
    # regression would tighten the CI instead
    rng = np.random.default_rng(17)
    rows = []
    for d in range(300):
        p = 0.75
        hit = float(rng.random() < p)
        batter = "A" if d % 3 == 0 else f"B{d}"
        rows.append((0, 2021 + d % 5, 1, f"d{d:04d}", batter, p, hit))
    df = _frame(rows)
    df["repeat_1"] = df["batter_id"] == "A"
    dup = pd.concat([df, df], ignore_index=True)  # 2x rows, same clusters
    c1 = gap_contrast(df, "repeat_1", n_boot=400, seed=8)
    c2 = gap_contrast(dup, "repeat_1", n_boot=400, seed=8)
    assert c2["diff"] == pytest.approx(c1["diff"], abs=1e-12)
    assert c2["diff_ci95_cluster_boot"] == pytest.approx(c1["diff_ci95_cluster_boot"], abs=1e-12)
    assert c2["n_clusters"] == c1["n_clusters"]


def test_stratified_contrast_matches_direct_computation():
    from scripts.audit.repeat_batter_conditioning import stratified_contrast

    rng = np.random.default_rng(4)
    n = 600
    stated = rng.uniform(0.70, 0.85, n)
    rows = [
        (0, 2021, 1, f"d{i:04d}", "A" if i % 4 == 0 else f"B{i}", stated[i], float(rng.random() < 0.75))
        for i in range(n)
    ]
    df = _frame(rows)
    df["repeat_1"] = df["batter_id"] == "A"
    strata = stratified_contrast(df, "repeat_1", n_strata=4)
    assert len(strata) == 4
    assert sum(s["n_repeat"] + s["n_fresh"] for s in strata) == n
    # spot-verify one stratum against a direct mask computation
    edges = np.quantile(df["stated"], [0.25, 0.5, 0.75])
    m = np.digitize(df["stated"], edges) == 2
    g = df[m]
    r, f = g[g["repeat_1"]], g[~g["repeat_1"]]
    direct = (r["hit"].mean() - r["stated"].mean()) - (f["hit"].mean() - f["stated"].mean())
    assert strata[2]["diff"] == pytest.approx(direct, abs=1e-12)


def test_gap_contrast_null_ci_covers_zero():
    rng = np.random.default_rng(5)
    rows = []
    for d in range(400):
        p = 0.75
        hit = float(rng.random() < p)
        batter = "A" if d % 4 == 0 else f"B{d}"
        rows.append((0, 2021 + d % 5, 42, f"d{d:04d}", batter, p, hit))
    df = _frame(rows)
    df["repeat_1"] = df["batter_id"] == "A"
    c = gap_contrast(df, "repeat_1", n_boot=500, seed=9)
    lo, hi = c["diff_ci95_cluster_boot"]
    assert lo < 0 < hi  # no planted effect -> CI covers zero
