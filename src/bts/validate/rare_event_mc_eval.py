"""CE-IS rare-event MC over a #5 manifest — SOTA tracker item #14 P0/P1.

Implements the design memo merged in PR #14 (`docs/superpowers/specs/
2026-05-04-bts-sota-14-rare-event-mc-design.md`). Per fold: fit theta on
fold-train profiles via CE rounds (the train point estimate is incurred
but discarded; there is no public tune-only API in v1's
`bts.simulate.rare_event_mc`); then run final IS estimate on fold-holdout
profiles with n_rounds=0 and the trained theta.

**Estimand**: P(max consecutive rank-1 hits >= streak_threshold) over the
ordered fold-holdout date sequence under independent Bernoulli rank-1
hits at the holdout's p_game_hit. Horizon = n_holdout_dates per fold.
**NOT comparable to #13 V_pi season-horizon policy value.**

Black-box wrapper. NO source-side changes to `rare_event_mc.py` in P0/P1.
Diagnostics use only public `CEISResult` fields. Strict JSON
(allow_nan=False) on output.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bts.simulate.rare_event_mc import estimate_p57_with_ceis


SCHEMA_VERSION = "rare_event_ce_is_v1"

DEFAULT_N_ROUNDS_TRAIN = 8
DEFAULT_N_PER_ROUND_TRAIN = 5000
DEFAULT_N_FINAL_TRAIN = 2000  # smaller than holdout: train estimate is discarded
DEFAULT_N_FINAL_HOLDOUT = 20000
DEFAULT_SEED = 42
DEFAULT_STREAK_THRESHOLD = 57
DEFAULT_MIN_ESS = 1000
DEFAULT_MAX_WEIGHT_SHARE = 0.1


def _ensure_seed_column(df: pd.DataFrame) -> pd.DataFrame:
    """Insert deterministic seed=0 if missing; preserve when present.

    Mirrors the pattern in `bts.validate.ope_eval` (PR #13). The current
    `data/simulation/backtest_*.parquet` files lack a seed column; using
    a synthetic 0 lets `apply_fold` work without per-call modification.
    """
    if "seed" in df.columns:
        return df
    out = df.copy()
    out["seed"] = 0
    return out


def _build_holdout_profiles(holdout_df: pd.DataFrame) -> list[dict[str, float]]:
    """Reduce rank-row holdout DataFrame to per-day p_game profiles.

    Per-fold ordered list of `{p_game: rank-1's p_game_hit}` per date. Order
    is stable in date ascending. Per Codex #123 second blocker, this matches
    the v1 CE-IS estimator's expected `profiles: list[dict]` shape exactly.

    **Fails closed (per Codex #137 #1) when rank-1 has duplicate dates** —
    e.g., a multi-seed backtest with one rank-1 row per (seed, date) pair.
    The schema's `horizon_basis = n_holdout_dates` claim only holds when
    there's one rank-1 per date; aggregating multiple seeds into a single
    horizon would silently extend the estimator's effective horizon and
    misreport it. P0/P1 explicitly does NOT support multi-seed input on
    this path; a future P1.5+ cycle can decide aggregation semantics.
    """
    rank1 = holdout_df[holdout_df["rank"] == 1].copy()
    n_rows = len(rank1)
    n_unique_dates = rank1["date"].nunique()
    if n_rows != n_unique_dates:
        n_dup = n_rows - n_unique_dates
        raise ValueError(
            f"Holdout has {n_rows} rank-1 rows across {n_unique_dates} "
            f"unique dates ({n_dup} duplicate-date rows). The fixed-window "
            "estimand requires one rank-1 row per date — multi-seed input "
            "is not supported in P0/P1 (would silently extend the horizon "
            "while reporting n_holdout_dates). Aggregate to one rank-1 per "
            "date upstream, or wait for the P1.5+ multi-seed cycle."
        )
    rank1 = rank1.sort_values("date")
    return [{"p_game": float(p)} for p in rank1["p_game_hit"].to_numpy()]


def _verdict_flag(
    ess: float,
    max_weight_share: float,
    *,
    min_ess: float,
    max_weight_share_threshold: float,
) -> str:
    """Diagnostic flag from public CEISResult fields. NOT a gate."""
    if ess >= min_ess and max_weight_share <= max_weight_share_threshold:
        return "OK"
    return "IS_DIAGNOSTIC_WARNING"


def evaluate_ceis_on_manifest(
    profiles_df: pd.DataFrame,
    manifest_path: str | Path,
    *,
    n_rounds_train: int = DEFAULT_N_ROUNDS_TRAIN,
    n_per_round_train: int = DEFAULT_N_PER_ROUND_TRAIN,
    n_final_train: int = DEFAULT_N_FINAL_TRAIN,
    n_final_holdout: int = DEFAULT_N_FINAL_HOLDOUT,
    seed: int = DEFAULT_SEED,
    streak_threshold: int = DEFAULT_STREAK_THRESHOLD,
    min_ess: float = DEFAULT_MIN_ESS,
    max_weight_share_threshold: float = DEFAULT_MAX_WEIGHT_SHARE,
) -> dict:
    """Run the per-fold CE-IS evaluation contract from the #14 design memo.

    For each fold:
    1. Build train profiles + holdout profiles from rank-1 p_game_hit.
    2. Call `estimate_p57_with_ceis` on TRAIN with n_rounds=n_rounds_train,
       n_final=n_final_train. Read `theta_final`; discard the train point
       estimate (the current API has no tune-only function).
    3. Call `estimate_p57_with_ceis` on HOLDOUT with n_rounds=0,
       theta=train_theta, n_final=n_final_holdout. Capture the holdout
       point estimate as the fold's `fixed_window_estimate`.
    4. Report public diagnostics + verdict_flag.

    Returns a `rare_event_ce_is_v1` schema dict. Lockbox held out per #5;
    `aggregate_deferred=True`.
    """
    from bts.validate.splits import (
        load_manifest,
        apply_fold,
        assert_no_lockbox_leakage,
    )

    folds, lockbox = load_manifest(manifest_path)
    assert_no_lockbox_leakage(folds, lockbox)
    raw = json.loads(Path(manifest_path).read_text())

    profiles_df = _ensure_seed_column(profiles_df)

    fold_results = []
    for fold in folds:
        train_df, holdout_df = apply_fold(profiles_df, fold)
        train_profiles = _build_holdout_profiles(train_df)
        holdout_profiles = _build_holdout_profiles(holdout_df)

        # Train phase: CE rounds learn theta on train profiles.
        # The train point estimate is incurred but DISCARDED.
        train_result = estimate_p57_with_ceis(
            train_profiles,
            strategy=None,
            n_rounds=n_rounds_train,
            n_per_round=n_per_round_train,
            n_final=n_final_train,
            seed=seed,
            streak_threshold=streak_threshold,
        )
        train_theta = train_result.theta_final

        # Holdout phase: final IS estimate using the train-tuned proposal.
        holdout_result = estimate_p57_with_ceis(
            holdout_profiles,
            strategy=None,
            n_rounds=0,
            n_final=n_final_holdout,
            theta=train_theta,
            seed=seed,
            streak_threshold=streak_threshold,
        )

        flag = _verdict_flag(
            holdout_result.ess,
            holdout_result.max_weight_share,
            min_ess=min_ess,
            max_weight_share_threshold=max_weight_share_threshold,
        )

        fold_results.append({
            "fold_idx": fold.fold_idx,
            "n_train_dates": len(fold.train_dates),
            "n_holdout_dates": len(fold.holdout_dates),
            "fixed_window_estimate": holdout_result.point_estimate,
            "ci_lower": holdout_result.ci_lower,
            "ci_upper": holdout_result.ci_upper,
            "theta_train": train_theta.tolist(),
            "diagnostics": {
                "ess": holdout_result.ess,
                "max_weight_share": holdout_result.max_weight_share,
                "log_weight_variance": holdout_result.log_weight_variance,
                "n_final": holdout_result.n_final,
                "verdict_flag": flag,
            },
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "estimand": {
            "name": "p_max_streak_ge_threshold_over_holdout_window",
            "description": (
                "P(max consecutive rank-1 hits >= streak_threshold) over the "
                "ordered fold-holdout date sequence under independent Bernoulli "
                "rank-1 hits. Horizon = n_holdout_dates per fold. NOT a "
                "season-level P57 estimate."
            ),
            "horizon_basis": "n_holdout_dates",
            "streak_threshold": streak_threshold,
        },
        "estimator": {
            "primary": "estimate_p57_with_ceis",
            "v1_simplifications": [
                "fits theta_0 only (constant logit shift); per-action/per-day tilt deferred to #14 P1.5+",
                "elite-set hit-count refit, NOT LR-weighted MLE",
                "always-play event indicator (strategy arg ignored); strategy-aware tilt deferred to P1.5+",
            ],
            "n_rounds_train": n_rounds_train,
            "n_per_round_train": n_per_round_train,
            "n_final_train": n_final_train,
            "n_final_holdout": n_final_holdout,
        },
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
            "min_ess": min_ess,
            "max_weight_share": max_weight_share_threshold,
            "n_rounds_train": n_rounds_train,
            "n_per_round_train": n_per_round_train,
            "n_final_train": n_final_train,
            "n_final_holdout": n_final_holdout,
            "seed": seed,
            "streak_threshold": streak_threshold,
        },
    }
