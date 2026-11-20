"""Policy-value evaluation over a #5 manifest — SOTA tracker item #13 P0/P1.

Implements the design memo `docs/superpowers/specs/2026-05-04-bts-sota-13-ope-design.md`
(merged in PR #12). Per fold: solve target policy on fold train, evaluate the
fixed policy against fold holdout bins via `evaluate_mdp_policy`, compute
terminal-MC replay on the same holdout profiles as a cross-check. Lockbox
held out per #5; aggregate CI deferred (per Codex #99 blocker 3).

P0/P1 baselines: `mdp_optimal` (default), `always_skip`, `always_rank1`.
Named-strategy adapter is deferred to P1.5+.

Output schema: `policy_value_eval_v1` per Codex #99 q5 (name the estimand,
not the estimator).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bts.simulate.mdp import solve_mdp
from bts.simulate.pooled_policy import (
    compute_pooled_bins,
    evaluate_mdp_policy,
    split_by_phase_pooled,
)
from bts.simulate.quality_bins import QualityBins


SCHEMA_VERSION = "policy_value_eval_v1"
DEFAULT_TARGET_POLICY = "mdp_optimal"
DEFAULT_MIN_BIN_N = 200
DEFAULT_SEASON_LENGTH = 180
DEFAULT_LATE_PHASE_DAYS = 30
DEFAULT_N_BINS = 5

VALID_TARGET_POLICIES = ("mdp_optimal", "always_skip", "always_rank1")


def _ensure_seed_column(df: pd.DataFrame) -> pd.DataFrame:
    """Per Codex #106 #3: backtest profiles may lack `seed` (current
    `data/simulation/backtest_*.parquet` are rank-row without it). Insert a
    deterministic seed=0 if missing; preserve when present."""
    if "seed" in df.columns:
        return df
    out = df.copy()
    out["seed"] = 0
    return out


def _build_baseline_policy_table(
    name: str, *, season_length: int, n_bins: int
) -> np.ndarray:
    """Construct a synthetic policy table for a baseline name.

    Shape (58, season_length+1, 2, n_bins) per Codex #106 #9. Action codes:
    0=skip, 1=single, 2=double (matching `bts.simulate.mdp.ACTIONS`).
    """
    shape = (58, season_length + 1, 2, n_bins)
    if name == "always_skip":
        return np.zeros(shape, dtype=int)
    if name == "always_rank1":
        return np.ones(shape, dtype=int)
    raise ValueError(
        f"_build_baseline_policy_table only handles always_skip / always_rank1; "
        f"got {name!r}. mdp_optimal is solved per-fold."
    )


def _solve_or_baseline_policy(
    target_policy_name: str,
    train_df: pd.DataFrame,
    *,
    season_length: int,
    late_phase_days: int,
    n_bins: int,
) -> np.ndarray:
    """Return the policy_table for the requested target.

    For mdp_optimal: solve the MDP on the train split's bins.
    For always_skip / always_rank1: synthetic table with the matching n_bins.
    """
    if target_policy_name not in VALID_TARGET_POLICIES:
        raise ValueError(
            f"target_policy must be one of {VALID_TARGET_POLICIES}; "
            f"got {target_policy_name!r}"
        )

    if target_policy_name in ("always_skip", "always_rank1"):
        return _build_baseline_policy_table(
            target_policy_name, season_length=season_length, n_bins=n_bins
        )

    # mdp_optimal: solve_mdp on fold-train bins
    train_df = _ensure_seed_column(train_df)
    train_early_df, train_late_df = split_by_phase_pooled(train_df, late_phase_days)
    train_early_bins = compute_pooled_bins(train_early_df, n_bins=n_bins)
    # Mirror the holdout late-bin guard (Codex #111 #2): only pass
    # train_late_bins when len(bins) == n_bins, otherwise fall back to None.
    # solve_mdp would otherwise index late arrays with a 5-bin policy when
    # train_late_df produces fewer quintile bins.
    train_late_bins = None
    if len(train_late_df) > 0:
        try:
            candidate_late = compute_pooled_bins(train_late_df, n_bins=n_bins)
            if len(candidate_late.bins) == n_bins:
                train_late_bins = candidate_late
        except (ValueError, IndexError):
            train_late_bins = None
    sol = solve_mdp(
        train_early_bins,
        season_length=season_length,
        late_bins=train_late_bins,
        late_phase_days=late_phase_days,
    )
    return sol.policy_table


def _bin_index_for_p(p: float, boundaries: list[float]) -> int:
    """Map a probability to a quantile bin index using boundaries from
    QualityBins. Mirrors `QualityBins.classify` but without requiring the
    full QualityBins object."""
    if not boundaries:
        return 0
    return int(np.digitize(p, np.asarray(boundaries, dtype=float)))


def _terminal_mc_replay(
    holdout_df: pd.DataFrame,
    policy_table: np.ndarray,
    early_bins: QualityBins,
    *,
    season_length: int,
    late_bins: QualityBins | None = None,
    late_dates: set | None = None,
) -> tuple[float, int, int]:
    """Empirical terminal-MC replay of `policy_table` on rank-row holdout
    profiles, classifying each day's rank-1 prediction into a holdout bin.

    Per Codex #106 #4: V_replay must use the SAME bins `evaluate_mdp_policy`
    uses, so it's comparable to V_pi. Per Codex #111 #1: when phase-aware
    bins are active (late_bins + late_dates non-None), classify each date's
    p_pred through the boundaries that match the date's phase — late dates
    use late_bins, early dates use early_bins. Without phase info it
    falls back to early-only.

    Returns (V_replay, n_trajectories, n_terminal_successes).
    """
    early_boundaries = list(early_bins.boundaries)
    late_boundaries = list(late_bins.boundaries) if late_bins is not None else None
    use_phase = late_bins is not None and late_dates is not None
    df = _ensure_seed_column(holdout_df).copy()
    df = df.sort_values(["season", "seed", "date", "rank"])

    n_traj = 0
    n_terminal = 0
    terminal_R = []
    for (season, seed), group in df.groupby(["season", "seed"]):
        # Per (season, seed): walk forward day-by-day, take rank-1 (and rank-2
        # if action=double) for each date.
        n_traj += 1
        streak = 0
        saver = 1  # available
        # Sort by date inside the group
        group_sorted = group.sort_values(["date", "rank"])
        days_remaining = season_length
        # Group by date so we can pull rank-1 / rank-2 per day
        reached_57 = False
        for date_val, day_group in group_sorted.groupby("date"):
            if streak >= 57:
                reached_57 = True
                break
            if days_remaining <= 0:
                break
            # rank-1 row
            r1_rows = day_group[day_group["rank"] == 1]
            r2_rows = day_group[day_group["rank"] == 2]
            if len(r1_rows) == 0:
                # No rank-1 — treat as forced skip
                action = 0
            else:
                p1 = float(r1_rows.iloc[0]["p_game_hit"])
                # Phase-aware classification: late_dates → late_boundaries
                # (when active); else early_boundaries.
                date_key = date_val if not hasattr(date_val, "date") else date_val.date()
                if use_phase and date_key in late_dates:
                    boundaries_today = late_boundaries
                else:
                    boundaries_today = early_boundaries
                qb = _bin_index_for_p(p1, boundaries_today)
                qb = max(0, min(qb, policy_table.shape[3] - 1))
                d = min(days_remaining, policy_table.shape[1] - 1)
                action = int(policy_table[streak, d, saver, qb])

            saver_active = bool(saver) and (10 <= streak <= 15)
            if action == 0:  # skip
                pass  # streak unchanged, saver unchanged
            elif action == 1:  # single
                hit = int(r1_rows.iloc[0]["actual_hit"]) if len(r1_rows) > 0 else 0
                if hit == 1:
                    streak = min(streak + 1, 57)
                else:
                    if saver_active:
                        saver = 0  # consume saver, streak unchanged
                    else:
                        streak = 0
            else:  # double (action == 2)
                if len(r2_rows) == 0:
                    # No rank-2 — treat as forced single
                    hit = int(r1_rows.iloc[0]["actual_hit"]) if len(r1_rows) > 0 else 0
                    if hit == 1:
                        streak = min(streak + 1, 57)
                    else:
                        if saver_active:
                            saver = 0
                        else:
                            streak = 0
                else:
                    h1 = int(r1_rows.iloc[0]["actual_hit"])
                    h2 = int(r2_rows.iloc[0]["actual_hit"])
                    both = h1 * h2
                    if both == 1:
                        streak = min(streak + 2, 57)
                    else:
                        if saver_active:
                            saver = 0
                        else:
                            streak = 0
            if streak >= 57:
                reached_57 = True
                break
            days_remaining -= 1
        terminal_R.append(1.0 if reached_57 else 0.0)
        if reached_57:
            n_terminal += 1

    v_replay = float(np.mean(terminal_R)) if terminal_R else 0.0
    return v_replay, n_traj, n_terminal


def _compute_per_bin_n(holdout_df: pd.DataFrame, holdout_bins: QualityBins) -> list[int]:
    """Per Codex #106 #6: QualityBins doesn't store n; compute per_bin_n from
    rank-1 holdout rows directly using the same boundaries used by the bins."""
    boundaries = list(holdout_bins.boundaries)
    n_bins = len(holdout_bins.bins)
    rank1 = holdout_df[holdout_df["rank"] == 1]
    if len(rank1) == 0:
        return [0] * n_bins
    p_arr = rank1["p_game_hit"].to_numpy()
    bin_idx = np.digitize(p_arr, np.asarray(boundaries, dtype=float))
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    counts = [int((bin_idx == k).sum()) for k in range(n_bins)]
    return counts


def _safe_relative_disagreement(disagreement_abs: float, v_pi: float) -> float | None:
    """JSON-safe relative disagreement.

    Returns 0.0 when both V_pi and disagreement_abs are 0 (perfect agreement
    at zero). Returns None when V_pi is 0 but disagreement_abs is nonzero
    (undefined relative quantity). Otherwise returns the ratio.

    Per Codex #109 #2: avoid float('inf') because json.dumps emits 'Infinity'
    which is not strict JSON.
    """
    if v_pi == 0:
        if disagreement_abs == 0:
            return 0.0
        return None  # undefined: nonzero abs over zero V_pi
    return disagreement_abs / v_pi


def evaluate_target_policy_on_manifest(
    profiles_df: pd.DataFrame,
    manifest_path: str | Path,
    *,
    target_policy_name: str = DEFAULT_TARGET_POLICY,
    season_length: int = DEFAULT_SEASON_LENGTH,
    late_phase_days: int = DEFAULT_LATE_PHASE_DAYS,
    min_bin_n: int = DEFAULT_MIN_BIN_N,
    n_bins: int = DEFAULT_N_BINS,
) -> dict:
    """Run the per-fold policy-value evaluation contract from the design memo.

    For each fold in the manifest:
    - solve target policy on fold train (or use baseline table)
    - compute holdout bins from fold holdout
    - V_pi = evaluate_mdp_policy(policy_table_train, holdout_bins)
    - V_replay = empirical terminal-MC replay on holdout profiles
    - per-bin diagnostic counts + sparse-support verdict_flag

    Returns a `policy_value_eval_v1` schema dict. `aggregate_deferred=True`.
    """
    from bts.validate.splits import (
        load_manifest,
        apply_fold,
        assert_no_lockbox_leakage,
    )

    if target_policy_name not in VALID_TARGET_POLICIES:
        raise ValueError(
            f"target_policy must be one of {VALID_TARGET_POLICIES}; "
            f"got {target_policy_name!r}"
        )

    folds, lockbox = load_manifest(manifest_path)
    assert_no_lockbox_leakage(folds, lockbox)
    raw = json.loads(Path(manifest_path).read_text())

    profiles_df = _ensure_seed_column(profiles_df)
    fold_results = []
    for fold in folds:
        train_df, holdout_df = apply_fold(profiles_df, fold)
        train_df = _ensure_seed_column(train_df)
        holdout_df = _ensure_seed_column(holdout_df)

        # Build train-side policy table
        policy_table = _solve_or_baseline_policy(
            target_policy_name, train_df,
            season_length=season_length,
            late_phase_days=late_phase_days,
            n_bins=n_bins,
        )

        # Build holdout-side bins
        holdout_early_df, holdout_late_df = split_by_phase_pooled(
            holdout_df, late_phase_days
        )
        holdout_early_bins = compute_pooled_bins(holdout_early_df, n_bins=n_bins)
        # Late bins only used if there's enough late data to produce the
        # same n_bins as the policy. With sparse late data, compute_pooled_bins
        # may produce fewer quintile bins; in that case fall back to early-only
        # (per Codex #109: late support is reported, but not forced into the
        # evaluator path when its n_bins doesn't match the policy).
        holdout_late_bins = None
        if len(holdout_late_df) > 0:
            try:
                candidate_late = compute_pooled_bins(holdout_late_df, n_bins=n_bins)
                if len(candidate_late.bins) == n_bins:
                    holdout_late_bins = candidate_late
            except (ValueError, IndexError):
                # Late phase too sparse — fall back to early-only evaluation.
                holdout_late_bins = None

        # V_pi: model-based forward eval
        v_pi = float(evaluate_mdp_policy(
            policy_table,
            holdout_early_bins,
            season_length=season_length,
            late_bins=holdout_late_bins,
            late_phase_days=late_phase_days,
        ))

        # V_replay: empirical terminal MC on holdout profiles, phase-aware
        # so V_replay uses the same per-date bin boundaries as V_pi
        # (per Codex #111 #1).
        late_dates_set = (
            set(holdout_late_df["date"].apply(
                lambda x: x.date() if hasattr(x, "date") else x
            ).tolist())
            if (holdout_late_bins is not None and len(holdout_late_df) > 0)
            else None
        )
        v_replay, n_traj, n_terminal = _terminal_mc_replay(
            holdout_df,
            policy_table,
            holdout_early_bins,  # early bins (default for non-late dates)
            season_length=season_length,
            late_bins=holdout_late_bins,
            late_dates=late_dates_set,
        )

        # Sparse-support diagnostics — per Codex #109 #4, cover the support
        # actually used by V_pi (both early and late phases when applicable).
        per_bin_n_early = _compute_per_bin_n(holdout_early_df, holdout_early_bins)
        early_min = min(per_bin_n_early) if per_bin_n_early else 0
        if holdout_late_bins is not None:
            per_bin_n_late = _compute_per_bin_n(holdout_late_df, holdout_late_bins)
            late_min = min(per_bin_n_late) if per_bin_n_late else 0
            # Top-level min across non-empty phases — V_pi can fail validity
            # in EITHER phase if its bin support is sparse there.
            holdout_bin_min_n = min(early_min, late_min)
        else:
            per_bin_n_late = None
            holdout_bin_min_n = early_min
        verdict_flag = (
            "OK" if holdout_bin_min_n >= min_bin_n else "SPARSE_HOLDOUT_SUPPORT"
        )

        disagreement_abs = abs(v_pi - v_replay)
        disagreement_rel = _safe_relative_disagreement(disagreement_abs, v_pi)

        sparse_support_block = {
            "n_holdout_trajectories": n_traj,
            "n_terminal_successes": n_terminal,
            "holdout_bin_min_n": holdout_bin_min_n,
            "per_bin_n_early": per_bin_n_early,
            "per_bin_n_late": per_bin_n_late,  # None when no late phase
            "verdict_flag": verdict_flag,
        }

        fold_results.append({
            "fold_idx": fold.fold_idx,
            "n_train_dates": len(fold.train_dates),
            "n_holdout_dates": len(fold.holdout_dates),
            "V_pi": v_pi,
            "V_replay": v_replay,
            "disagreement_abs": disagreement_abs,
            "disagreement_rel": disagreement_rel,
            "sparse_support": sparse_support_block,
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "estimand": {
            "name": "p57",
            "description": "P(streak reaches 57) under target policy",
            "horizon": season_length,
            "initial_state": {"streak": 0, "days_remaining": season_length, "saver": 1},
        },
        "estimator": {
            "primary": "evaluate_mdp_policy",
            "cross_check": "terminal_mc_replay",
            "version": "v1",
        },
        "target_policy": target_policy_name,
        "manifest_metadata": {
            "manifest_path": str(manifest_path),
            "schema_version": raw.get("schema_version"),
            "created_at": raw.get("created_at"),
            "split_params": raw.get("split_params"),
            "universe": raw.get("universe"),
        },
        "lockbox_held_out": True,
        "lockbox": {
            "start_date": lockbox.start_date.isoformat(),
            "end_date": lockbox.end_date.isoformat(),
            "description": lockbox.description,
        },
        "n_folds": len(folds),
        "fold_results": fold_results,
        "aggregate_deferred": True,
        "thresholds": {
            "min_bin_n": min_bin_n,
            "season_length": season_length,
            "late_phase_days": late_phase_days,
            "n_bins": n_bins,
        },
    }
