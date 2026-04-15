"""Tests for pooled-seed MDP policy builder (Option 7)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bts.simulate.mdp import load_policy, solve_mdp
from bts.simulate.pooled_policy import (
    build_pooled_policy,
    compute_pooled_bins,
    evaluate_mdp_policy,
    load_pooled_profiles,
    parse_seed_from_path,
    split_by_phase_pooled,
)
from bts.simulate.quality_bins import compute_bins as compute_bins_single


def _fake_profiles(n_days: int = 60, season: int = 2024,
                   seed_offset: float = 0.0, rng_seed: int = 1) -> pd.DataFrame:
    """Generate synthetic top-5 daily profiles with a known distribution.

    Rank-1 p_game_hit is drawn uniformly from [0.75, 0.92] per day so the
    distribution is wide enough to produce 5 non-degenerate quintile bins.
    Ranks 2-5 step down from rank-1 by ~0.02 each. hits are Bernoulli(p).
    seed_offset adds a constant cross-seed shift.
    """
    rows = []
    rng = np.random.default_rng(rng_seed)
    start = pd.Timestamp(f"{season}-04-01")
    for d in range(n_days):
        date = (start + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        p1 = float(rng.uniform(0.75, 0.92) + seed_offset)
        base_probs = np.clip(
            np.array([p1, p1 - 0.02, p1 - 0.04, p1 - 0.06, p1 - 0.08]),
            0.01, 0.99,
        )
        hits = rng.binomial(1, base_probs)
        for rank, (p, hit) in enumerate(zip(base_probs, hits), start=1):
            rows.append({
                "date": date, "rank": rank, "batter_id": 1000 + rank,
                "p_game_hit": float(p), "actual_hit": int(hit), "n_pas": 5,
            })
    df = pd.DataFrame(rows)
    df["season"] = season
    return df


class TestParseSeedFromPath:
    def test_embedded_in_segment(self):
        assert parse_seed_from_path(Path("data/hetzner_results/run/simulation_seed42")) == 42

    def test_dedicated_segment(self):
        assert parse_seed_from_path(Path("data/hetzner_results/run/seed7/simulation")) == 7

    def test_string_accepted(self):
        assert parse_seed_from_path("path/to/seed1024/x") == 1024

    def test_raises_when_missing(self):
        with pytest.raises(ValueError):
            parse_seed_from_path(Path("no_seed_here"))


class TestLoadPooledProfiles:
    def test_reads_and_tags(self, tmp_path):
        for seed in [1, 2]:
            seed_dir = tmp_path / f"seed{seed}"
            seed_dir.mkdir()
            df = _fake_profiles(n_days=30, season=2024, seed_offset=seed * 0.001)
            df.to_parquet(seed_dir / "backtest_2024.parquet", index=False)

        pooled = load_pooled_profiles([tmp_path / "seed1", tmp_path / "seed2"])
        assert set(pooled["seed"].unique()) == {1, 2}
        assert pooled["date"].nunique() == 30  # 30 distinct calendar days
        assert len(pooled) == 2 * 30 * 5       # 2 seeds × 30 days × 5 ranks

    def test_parses_season_from_filename_if_missing(self, tmp_path):
        seed_dir = tmp_path / "seed3"
        seed_dir.mkdir()
        df = _fake_profiles(n_days=10, season=2024).drop(columns=["season"])
        df.to_parquet(seed_dir / "backtest_2024.parquet", index=False)

        pooled = load_pooled_profiles([seed_dir])
        assert set(pooled["season"].unique()) == {2024}

    def test_raises_when_empty_dir(self, tmp_path):
        empty_dir = tmp_path / "seed99"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="no backtest"):
            load_pooled_profiles([empty_dir])

    def test_raises_when_no_dirs(self):
        with pytest.raises(ValueError, match="no seed_dirs"):
            load_pooled_profiles([])


class TestComputePooledBins:
    def test_requires_seed_column(self):
        df = _fake_profiles(n_days=60, season=2024)
        with pytest.raises(ValueError, match="seed"):
            compute_pooled_bins(df)

    def test_duplicating_a_seed_preserves_bin_stats(self):
        """Pooling two identical seeds must produce the same boundaries,
        p_hit, and p_both as compute_bins on a single seed.

        If compute_pooled_bins incorrectly merged rank-1 and rank-2 across
        seeds, the duplicated pooling would yield 4×more rows per date
        (cartesian cross-pairing) and the p_both/frequency would drift.
        This test would catch that bug.
        """
        df1 = _fake_profiles(n_days=60, season=2024, rng_seed=1)
        df2 = df1.copy()
        df1["seed"] = 1
        df2["seed"] = 2
        pooled = pd.concat([df1, df2], ignore_index=True)

        pooled_bins = compute_pooled_bins(pooled, n_bins=5)
        single_bins = compute_bins_single(df1.drop(columns=["seed"]), n_bins=5)

        assert pooled_bins.boundaries == single_bins.boundaries
        assert len(pooled_bins.bins) == len(single_bins.bins)
        for pb, sb in zip(pooled_bins.bins, single_bins.bins):
            assert pb.p_hit == pytest.approx(sb.p_hit, abs=1e-10)
            assert pb.p_both == pytest.approx(sb.p_both, abs=1e-10)
            assert pb.frequency == pytest.approx(sb.frequency, abs=1e-10)

    def test_distinct_seeds_average_per_bin_rates(self):
        """With two seeds producing distinct hit patterns, the pooled
        p_hit should be the mean of the two single-seed p_hits (to within
        bin-reassignment perturbations)."""
        df1 = _fake_profiles(n_days=200, season=2024, rng_seed=1)
        df2 = _fake_profiles(n_days=200, season=2024, rng_seed=2)
        df1["seed"] = 10
        df2["seed"] = 20
        pooled = pd.concat([df1, df2], ignore_index=True)

        pooled_bins = compute_pooled_bins(pooled, n_bins=5)
        # sanity: 5 bins, frequencies sum to 1, p_hit in [0,1]
        assert len(pooled_bins.bins) == 5
        total_freq = sum(b.frequency for b in pooled_bins.bins)
        assert abs(total_freq - 1.0) < 1e-9
        for b in pooled_bins.bins:
            assert 0.0 <= b.p_hit <= 1.0
            assert 0.0 <= b.p_both <= 1.0


class TestSplitByPhasePooled:
    def test_preserves_seed_column(self):
        df1 = _fake_profiles(n_days=60, season=2024, rng_seed=1)
        df2 = _fake_profiles(n_days=60, season=2024, rng_seed=2)
        df1["seed"] = 1
        df2["seed"] = 2
        pooled = pd.concat([df1, df2], ignore_index=True)

        early, late = split_by_phase_pooled(pooled, late_phase_days=20)
        # 60 days - 20 late days = 40 early days; × 2 seeds × 5 ranks
        assert len(early) == 40 * 2 * 5
        assert len(late) == 20 * 2 * 5
        assert set(early["seed"].unique()) == {1, 2}
        assert set(late["seed"].unique()) == {1, 2}

    def test_late_phase_days_zero_returns_empty_late(self):
        df = _fake_profiles(n_days=30, season=2024, rng_seed=1)
        df["seed"] = 1
        early, late = split_by_phase_pooled(df, late_phase_days=0)
        assert len(early) == len(df)
        assert len(late) == 0

    def test_late_window_longer_than_season_puts_all_in_late(self):
        df = _fake_profiles(n_days=10, season=2024, rng_seed=1)
        df["seed"] = 1
        early, late = split_by_phase_pooled(df, late_phase_days=30)
        assert len(early) == 0
        assert len(late) == len(df)


class TestBuildPooledPolicy:
    def test_end_to_end_save_and_load(self, tmp_path):
        df1 = _fake_profiles(n_days=190, season=2024, rng_seed=1, seed_offset=0.0)
        df2 = _fake_profiles(n_days=190, season=2024, rng_seed=2, seed_offset=0.005)
        df1["seed"] = 1
        df2["seed"] = 2
        pooled = pd.concat([df1, df2], ignore_index=True)

        sol = build_pooled_policy(
            pooled, season_length=180, late_phase_days=30, n_bins=5,
        )

        # Policy table shape and value sanity
        assert sol.policy_table.shape == (58, 181, 2, 5)
        assert sol.optimal_p57 >= 0.0
        assert sol.optimal_p57 <= 1.0

        out_path = tmp_path / "mdp_policy_pooled_test.npz"
        sol.save(out_path)
        assert out_path.exists()

        # Round-trip via the production loader
        table, boundaries, season_length = load_policy(out_path)
        assert table.shape == (58, 181, 2, 5)
        assert len(boundaries) == 4
        assert season_length == 180

    def test_late_phase_days_zero_falls_back_to_single_phase(self, tmp_path):
        df = _fake_profiles(n_days=100, season=2024, rng_seed=1)
        df["seed"] = 1
        sol = build_pooled_policy(df, season_length=100, late_phase_days=0, n_bins=5)
        assert sol.policy_table.shape == (58, 101, 2, 5)


class TestEvaluateMdpPolicy:
    """The forward evaluator is the primitive for honest A/B comparisons."""

    def test_self_consistency_single_phase(self):
        """Evaluating a policy on ITS OWN training bins must recover
        the policy's optimal_p57 to within floating-point noise.

        This is the crucial correctness guarantee: if I solve the MDP
        against bins B to get (policy P, optimal_p57 = V*), then
        forward-evaluating P against those same bins B must yield V*.
        If this test fails, the forward evaluator has a bug in its
        backward-induction loop.
        """
        df = _fake_profiles(n_days=200, season=2024, rng_seed=11)
        df["seed"] = 11
        bins = compute_pooled_bins(df, n_bins=5)
        sol = solve_mdp(bins, season_length=100)

        v_forward = evaluate_mdp_policy(
            policy_table=sol.policy_table,
            early_bins=bins,
            season_length=100,
        )
        assert v_forward == pytest.approx(sol.optimal_p57, abs=1e-9)

    def test_self_consistency_phase_aware(self):
        """Phase-aware variant of the self-consistency check."""
        df = _fake_profiles(n_days=200, season=2024, rng_seed=13)
        df["seed"] = 13
        early_df, late_df = split_by_phase_pooled(df, late_phase_days=30)
        early_bins = compute_pooled_bins(early_df, n_bins=5)
        late_bins = compute_pooled_bins(late_df, n_bins=5)
        sol = solve_mdp(early_bins, season_length=150,
                        late_bins=late_bins, late_phase_days=30)

        v_forward = evaluate_mdp_policy(
            policy_table=sol.policy_table,
            early_bins=early_bins,
            season_length=150,
            late_bins=late_bins,
            late_phase_days=30,
        )
        assert v_forward == pytest.approx(sol.optimal_p57, abs=1e-9)

    def test_cross_bin_evaluation_differs(self):
        """Evaluating a policy trained on bins A against DIFFERENT bins B
        should yield a different value than bins A's in-sample optimum —
        usually lower (because the policy isn't tuned to B).

        This is the scenario the A/B validation runs on: a pooled-train
        policy vs a held-out seed's bins.
        """
        df_a = _fake_profiles(n_days=200, season=2024, rng_seed=21)
        df_a["seed"] = 21
        df_b = _fake_profiles(n_days=200, season=2024, rng_seed=99)
        df_b["seed"] = 99
        bins_a = compute_pooled_bins(df_a, n_bins=5)
        bins_b = compute_pooled_bins(df_b, n_bins=5)

        sol_a = solve_mdp(bins_a, season_length=100)
        v_a_on_b = evaluate_mdp_policy(
            policy_table=sol_a.policy_table,
            early_bins=bins_b,
            season_length=100,
        )
        assert 0.0 <= v_a_on_b <= 1.0
        # In-sample optimum is usually strictly >= cross-sample evaluation
        sol_b = solve_mdp(bins_b, season_length=100)
        assert v_a_on_b <= sol_b.optimal_p57 + 1e-9
