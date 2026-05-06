from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bts.simulate.mdp import solve_mdp
from bts.simulate.quality_bins import QualityBin, QualityBins
from scripts.dr_mdp_gap_measure import (
    build_simplex_frequency_candidates,
    main,
    measure_gap,
    pair_frame_from_profiles,
    policy_disagreement_rate,
    quality_bins_from_pairs,
    solve_robust_mdp,
)


def _two_bins() -> QualityBins:
    return QualityBins(
        bins=[
            QualityBin(index=0, p_range=(0.6, 0.75), p_hit=0.55, p_both=0.35, frequency=0.4),
            QualityBin(index=1, p_range=(0.75, 0.95), p_hit=0.90, p_both=0.78, frequency=0.6),
        ],
        boundaries=[0.75],
    )


def _direct_profiles() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(7)
    for season in (2024, 2025):
        for day in range(40):
            for seed in range(3):
                p = 0.58 + 0.35 * ((day + seed) % 10) / 9
                hit_prob = 0.52 if p < 0.75 else 0.86
                rows.append({
                    "season": season,
                    "date": pd.Timestamp(f"{season}-04-01") + pd.Timedelta(days=day),
                    "seed": seed,
                    "top1_p": p,
                    "top1_hit": int(rng.random() < hit_prob),
                    "top2_p": max(0.50, p - 0.04),
                    "top2_hit": int(rng.random() < hit_prob * 0.90),
                })
    return pd.DataFrame(rows)


def test_singleton_robust_mdp_matches_point_estimate_solver():
    bins = _two_bins()
    point = solve_mdp(bins, season_length=75)
    hit_candidates = [[(b.p_hit, b.p_both)] for b in bins.bins]
    freq_candidates = [np.array([b.frequency for b in bins.bins])]

    robust = solve_robust_mdp(
        bins,
        hit_candidates,
        freq_candidates,
        season_length=75,
    )

    assert robust.robust_p57 == pytest.approx(point.optimal_p57, abs=1e-12)
    assert policy_disagreement_rate(point.policy_table, robust.policy_table) == 0.0


def test_lower_hit_candidate_cannot_raise_robust_value():
    bins = _two_bins()
    point = solve_mdp(bins, season_length=75)
    hit_candidates = [
        [(b.p_hit, b.p_both), (max(0.0, b.p_hit - 0.20), max(0.0, b.p_both - 0.20))]
        for b in bins.bins
    ]
    freq_candidates = [np.array([b.frequency for b in bins.bins])]

    robust = solve_robust_mdp(
        bins,
        hit_candidates,
        freq_candidates,
        season_length=75,
    )

    assert robust.robust_p57 <= point.optimal_p57 + 1e-12


def test_frequency_candidates_stay_on_simplex_and_include_empirical():
    base = np.array([0.2, 0.3, 0.5])
    lower = np.array([0.1, 0.2, 0.4])
    upper = np.array([0.4, 0.5, 0.7])

    candidates = build_simplex_frequency_candidates(base, lower, upper)

    assert any(np.allclose(candidate, base) for candidate in candidates)
    assert len(candidates) > 1
    for candidate in candidates:
        assert np.all(candidate >= 0.0)
        assert candidate.sum() == pytest.approx(1.0)


def test_pair_frame_supports_ranked_schema_without_cross_seed_join():
    ranked = pd.DataFrame([
        {"season": 2025, "date": "2025-04-01", "seed": 1, "rank": 1, "p_game_hit": 0.8, "actual_hit": 1},
        {"season": 2025, "date": "2025-04-01", "seed": 1, "rank": 2, "p_game_hit": 0.7, "actual_hit": 1},
        {"season": 2025, "date": "2025-04-01", "seed": 2, "rank": 1, "p_game_hit": 0.75, "actual_hit": 0},
        {"season": 2025, "date": "2025-04-01", "seed": 2, "rank": 2, "p_game_hit": 0.65, "actual_hit": 1},
    ])

    pairs = pair_frame_from_profiles(ranked)

    assert len(pairs) == 2
    assert set(pairs["seed"]) == {1, 2}
    assert set(pairs.columns).issuperset({"p_game_hit", "actual_hit", "top2_hit"})


def test_measure_gap_emits_both_constructions():
    profiles = _direct_profiles()

    result = measure_gap(
        profiles,
        season_length=65,
        n_bins=3,
        ci_half_width=0.01,
        n_bootstrap=8,
        seed=3,
    )

    assert result["schema_version"] == 1
    assert result["point_p57"] >= 0.0
    assert result["max_delta_p57"] >= 0.0
    assert result["max_delta_exceeds_ci_half_width"] in {True, False}
    assert [c["name"] for c in result["constructions"]] == [
        "wilson_simplex",
        "paired_day_bootstrap_multinomial",
    ]
    assert len(result["bin_stats"]) == 3


def test_quality_bins_from_pairs_retains_requested_bin_count():
    pairs = pair_frame_from_profiles(_direct_profiles())

    bins, stats = quality_bins_from_pairs(pairs, n_bins=4)

    assert len(bins.bins) == 4
    assert len(stats) == 4
    assert sum(stat.n for stat in stats) == len(pairs)
    assert sum(bin.frequency for bin in bins.bins) == pytest.approx(1.0)


def test_measure_gap_outputs_strict_json_when_bins_are_empty():
    profiles = _direct_profiles().head(30).copy()
    profiles["top1_p"] = 0.75

    result = measure_gap(
        profiles,
        season_length=10,
        n_bins=4,
        n_bootstrap=0,
    )

    assert any(stat["n"] == 0 for stat in result["bin_stats"])
    json.dumps(result, allow_nan=False)


def test_cli_writes_json(tmp_path: Path, capsys):
    profiles = _direct_profiles()
    profiles_path = tmp_path / "profiles.parquet"
    out_path = tmp_path / "dr_mdp_gap.json"
    profiles.to_parquet(profiles_path)

    rc = main([
        "--profiles-glob", str(profiles_path),
        "--out", str(out_path),
        "--season-length", "60",
        "--n-bins", "3",
        "--n-bootstrap-candidates", "4",
    ])

    assert rc == 0
    captured = capsys.readouterr()
    assert "DR-MDP gap screen:" in captured.err
    assert "max_delta=" in captured.err
    data = json.loads(out_path.read_text())
    assert data["source_profiles"] == [str(profiles_path.resolve())]
    assert data["method"] == "finite_candidate_rectangular_dr_mdp_screen"
    assert len(data["constructions"]) == 2
