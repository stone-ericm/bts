"""Doubly Robust Off-Policy Evaluation for the BTS MDP.

References:
- Jiang & Li 2016. Doubly Robust Off-policy Value Evaluation for Reinforcement
  Learning. ICML.
- Le, Voloshin & Yue 2019. Batch Policy Learning under Constraints. ICML.
- Precup, Sutton & Singh 2000. Eligibility Traces for Off-Policy Policy
  Evaluation.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def fitted_q_evaluation(
    df: pd.DataFrame,
    target_policy: Callable[[int, int], int],
    *,
    n_states: int,
    n_actions: int,
    horizon: int,
) -> float:
    """Tabular FQE: estimate V^pi(s_0=0) via backward induction on observed transitions.

    Args:
        df: dataframe with columns t, s, a, sn, r.
        target_policy: callable (state, t) -> action.
        n_states, n_actions, horizon: MDP dimensions.

    Returns:
        Estimated V^pi(s=0) at t=0.
    """
    Q = np.zeros((horizon + 1, n_states, n_actions))
    counts = np.zeros((horizon, n_states, n_actions, n_states))
    rew_sum = np.zeros((horizon, n_states, n_actions, n_states))
    for row in df.itertuples():
        counts[row.t, row.s, row.a, row.sn] += 1
        rew_sum[row.t, row.s, row.a, row.sn] += row.r
    P = np.zeros_like(counts)
    R = np.zeros_like(counts)
    for t in range(horizon):
        for s in range(n_states):
            for a in range(n_actions):
                tot = counts[t, s, a].sum()
                if tot > 0:
                    P[t, s, a] = counts[t, s, a] / tot
                    R[t, s, a] = rew_sum[t, s, a] / np.maximum(counts[t, s, a], 1)
    for t in reversed(range(horizon)):
        for s in range(n_states):
            for a in range(n_actions):
                v_next = sum(
                    P[t, s, a, sn] * (R[t, s, a, sn] + Q[t + 1, sn, target_policy(sn, t + 1)])
                    for sn in range(n_states)
                )
                Q[t, s, a] = v_next
    return float(Q[0, 0, target_policy(0, 0)])


@dataclass
class DROPEResult:
    """Result of one DR-OPE evaluation."""

    point_estimate: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    n_trajectories: int = 0
    nuisance_v_hat: float | None = None
    bootstrap_distribution: np.ndarray | None = None
    fold_metadata: list[dict] = field(default_factory=list)  # NEW (v2): per-fold dependence params


def dr_ope_full_information(
    df: pd.DataFrame,
    target_policy: Callable[[int, int], int],
    *,
    n_states: int,
    n_actions: int,
    horizon: int,
) -> float:
    """DR-OPE estimator under full-information action replay (rho=1).

    For each trajectory i:
        V_DR_i = V_hat(s_0)
                 + sum_t [r_t + V_hat(s_{t+1}) − Q_hat(s_t, a_t)]

    Returns the mean V_DR_i across trajectories.

    Assumes full-information replay where the target policy's action outcome is
    observed for each (s, t) — appropriate for BTS where rank-1, rank-2, and
    skip outcomes are all logged daily.
    """
    counts = np.zeros((horizon, n_states, n_actions, n_states))
    rew_sum = np.zeros((horizon, n_states, n_actions, n_states))
    for row in df.itertuples():
        counts[row.t, row.s, row.a, row.sn] += 1
        rew_sum[row.t, row.s, row.a, row.sn] += row.r
    P = np.zeros_like(counts)
    R = np.zeros_like(counts)
    for t in range(horizon):
        for s in range(n_states):
            for a in range(n_actions):
                tot = counts[t, s, a].sum()
                if tot > 0:
                    P[t, s, a] = counts[t, s, a] / tot
                    R[t, s, a] = rew_sum[t, s, a] / np.maximum(counts[t, s, a], 1)
    Q = np.zeros((horizon + 1, n_states, n_actions))
    for t in reversed(range(horizon)):
        for s in range(n_states):
            for a in range(n_actions):
                Q[t, s, a] = sum(
                    P[t, s, a, sn] * (R[t, s, a, sn] + Q[t + 1, sn, target_policy(sn, t + 1)])
                    for sn in range(n_states)
                )
    V = np.array([
        [Q[t, s, target_policy(s, t)] for s in range(n_states)]
        for t in range(horizon + 1)
    ])

    v_dr_values = []
    for traj_id, traj in df.groupby("trajectory_id"):
        traj = traj.sort_values("t")
        v_correction = 0.0
        for row in traj.itertuples():
            target_a = target_policy(row.s, row.t)
            if row.a == target_a:
                v_next = V[row.t + 1, row.sn]
                q_t = Q[row.t, row.s, row.a]
                v_correction += (row.r + v_next - q_t)
        v_dr_i = V[0, 0] + v_correction
        v_dr_values.append(v_dr_i)
    return float(np.mean(v_dr_values))


def stationary_bootstrap_indices(
    n_days: int,
    *,
    expected_block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Politis & Romano 1994 stationary bootstrap.

    Resamples a length-n_days index array using geometric block lengths with
    expected length `expected_block_length`. Wraps around the day axis.

    Reference: Politis & Romano 1994, "The Stationary Bootstrap." JASA.
    """
    p = 1.0 / expected_block_length
    out = np.empty(n_days, dtype=np.int64)
    out[0] = rng.integers(n_days)
    for i in range(1, n_days):
        if rng.random() < p:
            out[i] = rng.integers(n_days)
        else:
            out[i] = (out[i - 1] + 1) % n_days
    return out


def paired_hierarchical_bootstrap_sample(
    df: pd.DataFrame,
    *,
    expected_block_length: int = 7,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Resample the day axis with stationary bootstrap; keep all seeds per day together.

    The dataframe must have at least: 'season', 'date', 'seed', plus payload columns.
    Within each season, dates are resampled via stationary bootstrap. For each
    resampled date, ALL rows (across seeds and other within-day groupings) are
    included — the day is the unit of dependence in BTS, so all 24 seeds share
    the realized baseball outcomes for that day.

    The output date column is reassigned to the *slot* date (i.e., the original date
    at position i in the sorted unique-dates array), not the source date that was
    drawn. This ensures each output date slot appears exactly once, preserving the
    block-contiguous temporal structure while allowing repeated draws to be identified
    by their assigned slot rather than their source date.
    """
    # Pre-group by (season, date) once (Codex round 4 perf fix). Avoids
    # repeated O(N) `season_df[season_df["date"] == source_date]` scans
    # inside the bootstrap loop. With 24 seeds × ~150 days × ~5 seasons,
    # the precompute is O(N) and lookups are O(1).
    out_chunks = []
    for season, season_df in df.groupby("season"):
        unique_dates = season_df["date"].drop_duplicates().sort_values().to_numpy()
        n_days = len(unique_dates)
        # Precompute date → chunk lookup for this season.
        chunks_by_date = {date: grp for date, grp in season_df.groupby("date")}
        idx = stationary_bootstrap_indices(
            n_days, expected_block_length=expected_block_length, rng=rng
        )
        resampled_dates = unique_dates[idx]
        for slot_date, source_date in zip(unique_dates, resampled_dates):
            chunk = chunks_by_date[source_date].copy()
            chunk["date"] = slot_date
            out_chunks.append(chunk)
    return pd.concat(out_chunks, ignore_index=True)


def _compute_bins_from_direct_profiles(
    profiles: pd.DataFrame,
    n_bins: int = 5,
) -> "QualityBins":
    """Compute QualityBins from direct-format profiles (top1_p/top1_hit/top2_p/top2_hit).

    This internal helper works with the daily-profile format produced by
    backtest_blend and used in audit_fixed_policy / audit_pipeline, which has
    columns: season, date, seed, top1_p, top1_hit, top2_p, top2_hit.

    Unlike compute_pooled_bins (which merges rank-1/rank-2 rows within
    (seed, date)), this format already has top1 and top2 in the same row,
    so no merge step is needed.

    Boundaries are computed over ALL rows (pooled across seeds).
    """
    from bts.simulate.quality_bins import QualityBin, QualityBins

    p = profiles["top1_p"].to_numpy()
    hit1 = profiles["top1_hit"].to_numpy().astype(bool)
    hit2 = profiles["top2_hit"].to_numpy().astype(bool)

    quantiles = [i / n_bins for i in range(1, n_bins)]
    boundaries = [float(np.quantile(p, q)) for q in quantiles]
    bin_idx = np.digitize(p, boundaries)

    bins: list[QualityBin] = []
    for i in range(n_bins):
        mask = bin_idx == i
        if mask.sum() == 0:
            continue
        bins.append(QualityBin(
            index=i,
            p_range=(float(p[mask].min()), float(p[mask].max())),
            p_hit=float(hit1[mask].mean()),
            p_both=float((hit1[mask] & hit2[mask]).mean()),
            frequency=float(mask.sum() / len(p)),
        ))
    return QualityBins(bins=bins, boundaries=boundaries)


def dr_ope_with_bootstrap(
    df: pd.DataFrame,
    target_policy: "Callable[[int, int], int]",
    *,
    n_states: int,
    n_actions: int,
    horizon: int,
    n_bootstrap: int = 2000,
    expected_block_length: int = 7,
    seed: int = 42,
    alpha: float = 0.05,
) -> "DROPEResult":
    """DR-OPE with paired hierarchical block bootstrap CI.

    Computes the DR-OPE point estimate via dr_ope_full_information, then
    n_bootstrap paired-hierarchical resamples to estimate (1 - alpha) percentile CI.
    """
    point = dr_ope_full_information(
        df, target_policy, n_states=n_states, n_actions=n_actions, horizon=horizon
    )
    rng = np.random.default_rng(seed)
    bootstrap_values = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        bs_df = paired_hierarchical_bootstrap_sample(
            df, expected_block_length=expected_block_length, rng=rng
        )
        bootstrap_values[b] = dr_ope_full_information(
            bs_df, target_policy, n_states=n_states, n_actions=n_actions, horizon=horizon
        )
    lo = float(np.quantile(bootstrap_values, alpha / 2))
    hi = float(np.quantile(bootstrap_values, 1 - alpha / 2))
    return DROPEResult(
        point_estimate=point,
        ci_lower=lo,
        ci_upper=hi,
        n_trajectories=df["trajectory_id"].nunique() if "trajectory_id" in df.columns else 0,
        bootstrap_distribution=bootstrap_values,
    )


def _trajectory_dataframe_from_profiles(
    profiles: pd.DataFrame,
    policy_action_table: np.ndarray,
    bins,  # QualityBins
) -> pd.DataFrame:
    """Convert daily profiles into trajectory-form DataFrame for DR-OPE.

    Each (season, seed) pair becomes one trajectory; days are time steps. State
    at each step is (streak, days_remaining, saver, quality_bin); action is
    `policy_action_table[state]`; outcome is realized hit (or skip). Reward
    R = 1 if streak reaches 57 during the trajectory, else 0.
    """
    # Performance refactor (Codex round 4): the inner state machine
    # (streak / saver) is genuinely sequential and can't be vectorized,
    # but everything ELSE can be precomputed:
    #   1. qbin: vectorized np.digitize on top1_p over the whole df, instead
    #      of bins.classify per row (Python overhead × 18K rows per audit).
    #   2. itertuples: ~5x faster than iterrows for row iteration.
    #   3. Integer action constants (ACTION_SKIP=0, etc.) compared instead of
    #      strings, avoiding ACTIONS[action_idx] lookup per row.

    # Action constants matching bts.simulate.mdp.ACTIONS = ("skip", "single", "double").
    ACTION_SKIP, ACTION_SINGLE, ACTION_DOUBLE = 0, 1, 2

    # Vectorized qbin assignment (replaces bins.classify per row).
    boundaries = np.asarray(bins.boundaries, dtype=float)
    p_arr = profiles["top1_p"].to_numpy()
    qbins_all = np.digitize(p_arr, boundaries) if boundaries.size else np.zeros(len(profiles), dtype=int)
    profiles = profiles.copy()
    profiles["_qbin"] = qbins_all

    n_streak_dim = policy_action_table.shape[0] - 1
    n_days_dim = policy_action_table.shape[1] - 1

    rows = []
    for (season, seed), group in profiles.groupby(["season", "seed"]):
        group = group.sort_values("date").reset_index(drop=True)
        streak = 0
        saver = 1
        n_steps = len(group)
        # Pre-extract numpy arrays for the columns we touch in the hot loop.
        top1_hits = group["top1_hit"].to_numpy()
        top2_hits = group["top2_hit"].to_numpy()
        qbins = group["_qbin"].to_numpy()
        dates = group["date"].to_numpy()
        traj_id = f"{season}_{seed}"
        for t in range(n_steps):
            if streak >= 57:
                break
            days_remaining = n_steps - t
            qbin = int(qbins[t])
            d_clamped = days_remaining if days_remaining <= n_days_dim else n_days_dim
            s_clamped = streak if streak <= n_streak_dim else n_streak_dim
            action_idx = int(policy_action_table[s_clamped, d_clamped, saver, qbin])
            top1 = bool(top1_hits[t])
            if action_idx == ACTION_SKIP:
                r = 0
                next_streak = streak
                next_saver = saver
            elif action_idx == ACTION_SINGLE:
                if top1:
                    next_streak = streak + 1 if streak + 1 < 57 else 57
                    next_saver = saver
                    r = 1 if (next_streak == 57 and streak < 57) else 0
                elif saver and 10 <= streak <= 15:
                    next_streak = streak
                    next_saver = 0
                    r = 0
                else:
                    next_streak = 0
                    next_saver = saver
                    r = 0
            else:  # ACTION_DOUBLE
                if top1 and bool(top2_hits[t]):
                    next_streak = streak + 2 if streak + 2 < 57 else 57
                    next_saver = saver
                    r = 1 if (next_streak == 57 and streak < 57) else 0
                elif saver and 10 <= streak <= 15:
                    next_streak = streak
                    next_saver = 0
                    r = 0
                else:
                    next_streak = 0
                    next_saver = saver
                    r = 0
            rows.append({
                "trajectory_id": traj_id,
                "season": season,
                "seed": seed,
                "date": dates[t],
                "t": t,
                "s_streak": streak,
                "s_days": days_remaining,
                "s_saver": saver,
                "s_qbin": qbin,
                "a": action_idx,
                "sn_streak": next_streak,
                "r": r,
            })
            streak = next_streak
            saver = next_saver
    return pd.DataFrame(rows)


def _run_terminal_r_mc_bootstrap(traj_df, *, n_bootstrap, seed):
    """Terminal-reward MC over trajectories with simple bootstrap CI.

    BTS rewards are purely terminal (R = 1 if streak reaches 57, else 0). So
    per-trajectory total reward is the trajectory's policy value, and the mean
    across trajectories is the policy value estimate.

    NOTE: This is naive terminal-reward MC, NOT DR-OPE. The earlier name
    `_run_dr_ope_with_bootstrap` was misleading — there is no nuisance correction
    or action-propensity correction here. The estimator is unbiased for the
    realized-trajectory policy value but provides no DR-style robustness against
    model misspecification. Bootstrap resamples trajectories (not days);
    underestimates uncertainty if seeds within a season share day/player shocks.

    `expected_block_length` is intentionally NOT a parameter here — it's a day-
    bootstrap concept and this estimator is trajectory-bootstrap.
    """
    rng = np.random.default_rng(seed)
    if "trajectory_id" not in traj_df.columns or len(traj_df) == 0:
        return DROPEResult(point_estimate=0.0, ci_lower=0.0, ci_upper=0.0, n_trajectories=0)
    terminal_R = traj_df.groupby("trajectory_id")["r"].sum().to_numpy()
    point = float(terminal_R.mean())
    bootstrap_values = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = rng.choice(len(terminal_R), size=len(terminal_R), replace=True)
        bootstrap_values[b] = terminal_R[sample].mean()
    return DROPEResult(
        point_estimate=point,
        ci_lower=float(np.quantile(bootstrap_values, 0.025)),
        ci_upper=float(np.quantile(bootstrap_values, 0.975)),
        n_trajectories=len(terminal_R),
        bootstrap_distribution=bootstrap_values,
    )


def audit_fixed_policy(
    profiles: pd.DataFrame,
    *,
    frozen_policy: dict,
    test_seasons: list[int],
    n_bootstrap: int = 2000,
    expected_block_length: int = 7,
    seed: int = 42,
) -> "DROPEResult":
    """Audit mode 1: frozen policy evaluated on held-out seasons.

    `frozen_policy` is a dict with key 'action_table' = the policy_table as
    saved by mdp.MDPSolution.save().
    """
    test_profiles = profiles[profiles["season"].isin(test_seasons)].copy()
    train_profiles = profiles[~profiles["season"].isin(test_seasons)].copy()
    bins = _compute_bins_from_direct_profiles(train_profiles)
    traj_df = _trajectory_dataframe_from_profiles(
        test_profiles, frozen_policy["action_table"], bins
    )
    return _run_terminal_r_mc_bootstrap(
        traj_df,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


def audit_pipeline(
    profiles: pd.DataFrame,
    *,
    fold_seasons: list[int],
    n_bootstrap: int = 2000,
    expected_block_length: int = 7,
    seed: int = 42,
) -> "DROPEResult":
    """Audit mode 2: leave-one-season-out, refit bins + re-solve MDP per fold."""
    from bts.simulate.mdp import solve_mdp

    fold_estimates = []
    for held_out in fold_seasons:
        train = profiles[profiles["season"] != held_out].copy()
        test = profiles[profiles["season"] == held_out].copy()
        bins = _compute_bins_from_direct_profiles(train)
        mdp_solution = solve_mdp(bins)
        traj_df = _trajectory_dataframe_from_profiles(test, mdp_solution.policy_table, bins)
        fold_result = _run_terminal_r_mc_bootstrap(
            traj_df,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        fold_estimates.append(fold_result.point_estimate)
    return DROPEResult(
        point_estimate=float(np.mean(fold_estimates)),
        ci_lower=float(np.quantile(fold_estimates, 0.025)) if len(fold_estimates) >= 5 else None,
        ci_upper=float(np.quantile(fold_estimates, 0.975)) if len(fold_estimates) >= 5 else None,
        n_trajectories=len(fold_estimates),
    )


def corrected_audit_pipeline(
    profiles: pd.DataFrame,
    pa_df: pd.DataFrame,
    *,
    fold_seasons: list[int],
    mdp_solve_fn: "Callable",
    n_bootstrap: int = 2000,
    seed: int = 42,
    n_pa_per_game: int = 5,
    rho_pair_n_permutations: int = 300,
    pa_n_bootstrap: int = 300,
) -> "DROPEResult":
    """LOSO audit with FOLD-LOCAL dependence parameters and corrected policy.

    For each held-out season:
      1. Slice training data to the 4 held-in seasons (no leakage).
      2. Fit fold-local bins from training profiles.
      3. Fit fold-local rho_PA, tau, rho_pair_per_bin from training PA + profiles.
      4. Build corrected QualityBins with the fold-local parameters.
      5. Solve MDP on corrected bins (via mdp_solve_fn).
      6. Replay on held-out season; compute terminal-reward MC point estimate.

    mdp_solve_fn contract (Codex round 1 fix): callable that takes
    `corrected_bins: QualityBins` and returns either an `np.ndarray`
    policy_table of shape (57, season_length+1, 2, n_bins) OR an object with
    `.policy_table` attribute (e.g., MDPSolution from solve_mdp). Adapter logic
    normalizes both.

    v2 closes v1's overfit gap: v1 used a GLOBAL corrected policy built from
    ALL data, causing the 0.1183 (in-sample) vs 0.0083 (LOSO) discrepancy.
    Each fold now refits rho_PA / tau / rho_pair_per_bin on its own 4 training
    seasons; the corrected bins and policy are fold-specific.

    Args:
        profiles: backtest profiles DataFrame with season, date, seed, top1_p,
            top1_hit, top2_p, top2_hit columns.
        pa_df: PA-level DataFrame with season, batter_game_id, p_pa, actual_hit
            columns.  Used to fit rho_PA (pa_residual_correlation) and tau
            (fit_logistic_normal_random_intercept) per fold.
        fold_seasons: seasons to use as held-out folds (LOSO).
        mdp_solve_fn: callable(corrected_bins) -> np.ndarray or MDPSolution.
        n_bootstrap: bootstrap replicates for the per-fold trajectory bootstrap CI.
        seed: base rng seed; fold i uses seed+i for independent realizations.
        n_pa_per_game: PAs per game for the tau→p_hit correction in
            build_corrected_transition_table.
        rho_pair_n_permutations: permutations for pair_residual_correlation per
            fold.  300 is adequate for fold-level estimates; use 1000+ for
            production-level CIs.
        pa_n_bootstrap: bootstrap replicates for pa_residual_correlation per fold.

    Returns:
        DROPEResult with mean across folds + percentile CI when n_folds >= 5,
        plus fold_metadata list populated with rho_PA, tau, rho_pair_per_bin
        (shape K indexed by bin.index 0..K-1), rho_pair_per_bin_ci, n_per_bin,
        stability dict, and per-fold P(57).
    """
    from bts.validate.dependence import (
        build_corrected_transition_table,
        fit_logistic_normal_random_intercept,
        pa_residual_correlation,
        pair_residual_correlation,
    )

    fold_estimates = []
    fold_metadata = []

    for fold_idx, held_out in enumerate(fold_seasons):
        # Fold-specific RNG seed (Codex round 1 minor): different fold gets
        # different bootstrap/permutation realizations.
        fold_seed = seed + fold_idx

        # Within-fold training data: 4 held-in seasons (NO leakage).
        # isin(held_in) guards against any season outside fold_seasons leaking
        # into training — e.g., a 2020 row in profiles would pass != held_out
        # but must not be in the training set.
        held_in = set(fold_seasons) - {held_out}
        train_profiles = profiles[profiles["season"].isin(held_in)].copy()
        train_pa = pa_df[pa_df["season"].isin(held_in)].copy()
        test_profiles = profiles[profiles["season"] == held_out].copy()

        # Fold-local bins.
        train_bins = _compute_bins_from_direct_profiles(train_profiles)
        n_bins = len(train_bins.bins)

        # Fold-local rho_PA via PA-level cluster bootstrap.
        rho_PA, rho_PA_lo, rho_PA_hi, _ = pa_residual_correlation(
            train_pa, n_bootstrap=pa_n_bootstrap, seed=fold_seed,
        )

        # Fold-local tau via logistic-normal random-intercept MoM inversion.
        train_pa_for_lnri = train_pa.rename(columns={
            "batter_game_id": "group_id",
            "p_pa": "p_pred",
            "actual_hit": "y",
        })
        tau_hat, _, lnri_stability = fit_logistic_normal_random_intercept(train_pa_for_lnri)

        # Fold-local per-bin rho_pair. CRITICAL: pass expected_bin_indices to
        # guarantee the returned vector is indexed by bin.index 0..K-1, NOT
        # by sorted-unique-of-data (which silently shifts when a bin is empty
        # for this fold's smaller training set).
        pair_df = train_profiles[
            ["date", "top1_p", "top1_hit", "top2_p", "top2_hit"]
        ].rename(columns={
            "top1_p": "p_rank1", "top1_hit": "y_rank1",
            "top2_p": "p_rank2", "top2_hit": "y_rank2",
        })
        bin_assignment = pair_df["p_rank1"].apply(train_bins.classify)
        rho_result = pair_residual_correlation(
            pair_df,
            n_permutations=rho_pair_n_permutations,
            bin_assignment=bin_assignment,
            expected_bin_indices=np.arange(n_bins),  # CRITICAL: stable indexing
            seed=fold_seed,
        )

        # Build fold-local corrected bins (rho_per_bin indexed by bin.index).
        corrected_bins = build_corrected_transition_table(
            train_bins,
            rho_PA_within_game=rho_PA,
            tau_squared=tau_hat ** 2,
            rho_pair_cross_game=rho_result["rho_per_bin"],
            n_pa_per_game=n_pa_per_game,
        )

        # Solve MDP on this fold's corrected bins. Adapter normalizes the two
        # legitimate return shapes (Codex round 1 fix):
        #   - np.ndarray: direct policy_table (e.g., _all_skip_policy in tests)
        #   - object with .policy_table: MDPSolution from solve_mdp
        solver_out = mdp_solve_fn(corrected_bins)
        if hasattr(solver_out, "policy_table"):
            policy_table = solver_out.policy_table
        elif isinstance(solver_out, np.ndarray):
            policy_table = solver_out
        else:
            raise TypeError(
                f"mdp_solve_fn returned unsupported type {type(solver_out)}; "
                f"must be np.ndarray or have .policy_table attribute"
            )

        # Replay on held-out season using fold-local corrected bins for state
        # encoding. Both the policy and the bins are fold-local — consistent.
        traj_df = _trajectory_dataframe_from_profiles(
            test_profiles, policy_table, corrected_bins,
        )
        fold_result = _run_terminal_r_mc_bootstrap(
            traj_df, n_bootstrap=n_bootstrap, seed=fold_seed,
        )
        fold_estimates.append(fold_result.point_estimate)

        fold_metadata.append({
            "held_out_season": int(held_out),
            "rho_PA": float(rho_PA),
            "rho_PA_ci_lo": float(rho_PA_lo),
            "rho_PA_ci_hi": float(rho_PA_hi),
            "tau": float(tau_hat),
            "rho_pair_per_bin": rho_result["rho_per_bin"],          # shape (n_bins,)
            "rho_pair_per_bin_ci_lo": rho_result["ci_lo_per_bin"],  # shape (n_bins,)
            "rho_pair_per_bin_ci_hi": rho_result["ci_hi_per_bin"],  # shape (n_bins,)
            "rho_pair_per_bin_p_value": rho_result["p_value_per_bin"],
            "rho_pair_n_per_bin": rho_result["n_per_bin"],
            "rho_pair_global": float(rho_result["global_rho"]),
            "bin_indices": rho_result["bin_indices"],
            "stability": lnri_stability,
            "fold_p57": float(fold_result.point_estimate),
            "n_trajectories": int(fold_result.n_trajectories),  # actual trajectories in this fold
        })

    point = float(np.mean(fold_estimates))
    if len(fold_estimates) >= 5:
        ci_lo = float(np.quantile(fold_estimates, 0.025))
        ci_hi = float(np.quantile(fold_estimates, 0.975))
    else:
        ci_lo, ci_hi = None, None

    # Sum actual trajectories across folds (not fold count).
    total_trajectories = sum(fm.get("n_trajectories", 1) for fm in fold_metadata)
    return DROPEResult(
        point_estimate=point,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        n_trajectories=total_trajectories,  # actual trajectories aggregated across folds
        fold_metadata=fold_metadata,
    )


def policy_regret_table(
    profiles: pd.DataFrame,
    *,
    target_policy_table: np.ndarray,
    bins,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute policy regret of target vs canonical baselines on the same bootstrap reps.

    Returns a dataframe with columns: baseline, point_regret, ci_lower, ci_upper.
    Baselines: always_skip (zeros action table), always_rank1 (ones action table).
    """
    rng = np.random.default_rng(seed)
    target_traj = _trajectory_dataframe_from_profiles(profiles, target_policy_table, bins)
    always_skip = np.zeros_like(target_policy_table)
    always_rank1 = np.ones_like(target_policy_table)
    skip_traj = _trajectory_dataframe_from_profiles(profiles, always_skip, bins)
    rank1_traj = _trajectory_dataframe_from_profiles(profiles, always_rank1, bins)

    target_terminal = target_traj.groupby("trajectory_id")["r"].sum()
    skip_terminal = skip_traj.groupby("trajectory_id")["r"].sum()
    rank1_terminal = rank1_traj.groupby("trajectory_id")["r"].sum()

    rows = []
    for baseline_name, baseline_R in [
        ("always_skip", skip_terminal),
        ("always_rank1", rank1_terminal),
    ]:
        # Align indices for paired difference.
        aligned = pd.concat({"t": target_terminal, "b": baseline_R}, axis=1).dropna()
        regret_per_traj = (aligned["t"] - aligned["b"]).to_numpy()
        traj_ids = aligned.index.to_numpy()
        bs_regrets = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            sample = rng.choice(len(traj_ids), size=len(traj_ids), replace=True)
            bs_regrets[b] = regret_per_traj[sample].mean()
        rows.append({
            "baseline": baseline_name,
            "point_regret": float(regret_per_traj.mean()),
            "ci_lower": float(np.quantile(bs_regrets, 0.025)),
            "ci_upper": float(np.quantile(bs_regrets, 0.975)),
        })
    return pd.DataFrame(rows)
