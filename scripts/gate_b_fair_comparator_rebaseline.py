#!/usr/bin/env python3
"""Gate B fair-comparator re-baseline on estimated-PA profiles.

This is an evidence-only harness. It decomposes the prior Gate B positive
screen by holding the corrected estimated-PA bin boundaries fixed and varying
only the action table:

* Arm A: solve a fresh MDP action table on prior estimated-PA seasons.
* Arm B: apply the deployed policy action table to the same estimated-PA bins.

The resulting gap estimates the value of re-solving the action table beyond
the boundary-scale fix. It never writes or swaps a production policy artifact.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bts.simulate.mdp import ACTIONS, load_policy, solve_mdp
from bts.simulate.pooled_policy import evaluate_mdp_policy
from bts.simulate.quality_bins import compute_bins, compute_bins_with_boundaries
from scripts.gate_b_walk_forward_policy_eval import (
    DEFAULT_N_BINS,
    DEFAULT_PROD_POLICY_PATH,
    DEFAULT_PROFILES_DIR,
    DEFAULT_SEASON_LENGTH,
    DEFAULT_SEASONS,
    _bins_summary,
    _dropped_starter_matchup_summary,
    _summarize_gaps,
    load_profiles,
    parse_seasons,
)


DEFAULT_OUTPUT = Path("data/validation/gate_b_fair_comparator_rebaseline_2026-05-24.json")


def _action_count_summary(actions: np.ndarray) -> dict[str, dict[str, float | int]]:
    flat = actions.astype(int).ravel()
    total = int(flat.size)
    counts = np.bincount(flat, minlength=len(ACTIONS))
    return {
        action: {
            "n": int(counts[index]),
            "fraction": 0.0 if total == 0 else float(counts[index] / total),
        }
        for index, action in enumerate(ACTIONS)
    }


def _policy_action_comparison(
    candidate_table: np.ndarray,
    deployed_table: np.ndarray,
    *,
    season_length: int,
) -> dict[str, Any]:
    if candidate_table.shape[3] != deployed_table.shape[3]:
        raise ValueError(
            "candidate and deployed action tables must have the same number of bins: "
            f"{candidate_table.shape[3]} != {deployed_table.shape[3]}"
        )

    n_streaks = min(57, candidate_table.shape[0], deployed_table.shape[0])
    max_day = min(season_length, candidate_table.shape[1] - 1, deployed_table.shape[1] - 1)
    n_saver = min(2, candidate_table.shape[2], deployed_table.shape[2])
    n_bins = candidate_table.shape[3]
    if max_day < 1:
        raise ValueError("policy tables must include at least one positive day state")

    candidate = candidate_table[:n_streaks, 1:max_day + 1, :n_saver, :n_bins].astype(int)
    deployed = deployed_table[:n_streaks, 1:max_day + 1, :n_saver, :n_bins].astype(int)
    same = candidate == deployed
    total = int(candidate.size)

    matrix: list[dict[str, Any]] = []
    for deployed_index, deployed_action in enumerate(ACTIONS):
        row = {"deployed_action": deployed_action}
        for candidate_index, candidate_action in enumerate(ACTIONS):
            row[f"candidate_{candidate_action}"] = int(
                ((deployed == deployed_index) & (candidate == candidate_index)).sum()
            )
        matrix.append(row)

    return {
        "state_count": total,
        "streaks_compared": int(n_streaks),
        "days_compared": int(max_day),
        "saver_states_compared": int(n_saver),
        "bins_compared": int(n_bins),
        "same_action_count": int(same.sum()),
        "different_action_count": int(total - same.sum()),
        "same_action_fraction": float(same.mean()),
        "candidate_action_counts": _action_count_summary(candidate),
        "deployed_action_counts": _action_count_summary(deployed),
        "deployed_by_candidate_action_matrix": matrix,
    }


def _decision(overall: dict[str, Any]) -> str:
    if overall["mean_gap"] > 0 and overall["n_negative"] == 0:
        return "RE_SOLVE_ACTION_TABLE_SIGNAL_POSITIVE_REQUIRES_FULL_GATE"
    if overall["mean_gap"] > 0:
        return "MIXED_RE_SOLVE_ACTION_TABLE_SIGNAL_REQUIRES_REVIEW"
    return "NO_RE_SOLVE_ACTION_TABLE_IMPROVEMENT"


def run_rebaseline(
    *,
    profiles_dir: Path,
    prod_policy_path: Path,
    seasons: Sequence[int] = DEFAULT_SEASONS,
    n_bins: int = DEFAULT_N_BINS,
    season_length: int = DEFAULT_SEASON_LENGTH,
    require_estimated_basis: bool = True,
    today: date | None = None,
) -> dict[str, Any]:
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    profiles = load_profiles(
        profiles_dir,
        seasons,
        require_estimated_basis=require_estimated_basis,
    )
    deployed_table, deployed_boundaries, deployed_policy_length = load_policy(prod_policy_path)
    if deployed_table.shape[3] != n_bins:
        raise ValueError(
            f"deployed policy has {deployed_table.shape[3]} bins; "
            f"fair comparator requested n_bins={n_bins}"
        )

    rows = []
    for holdout_season in seasons[1:]:
        train_seasons = [season for season in seasons if season < holdout_season]
        train_profiles = profiles[profiles["season"].isin(train_seasons)].copy()
        holdout_profiles = profiles[profiles["season"] == holdout_season].copy()
        if train_profiles.empty or holdout_profiles.empty:
            raise ValueError(f"empty train/holdout fold for {holdout_season}")

        shared_train_bins = compute_bins(train_profiles, n_bins=n_bins)
        re_solved_solution = solve_mdp(shared_train_bins, season_length=season_length)
        shared_holdout_bins = compute_bins_with_boundaries(
            holdout_profiles,
            shared_train_bins.boundaries,
        )

        v_re_solved = evaluate_mdp_policy(
            re_solved_solution.policy_table,
            shared_holdout_bins,
            season_length=season_length,
        )
        v_deployed_action_structure = evaluate_mdp_policy(
            deployed_table,
            shared_holdout_bins,
            season_length=season_length,
        )

        n_train_rank1 = int((train_profiles["rank"] == 1).sum())
        n_holdout_rank1 = int((holdout_profiles["rank"] == 1).sum())
        rows.append({
            "holdout_season": int(holdout_season),
            "train_seasons": [int(season) for season in train_seasons],
            "n_train_rank1": n_train_rank1,
            "n_holdout_rank1": n_holdout_rank1,
            "re_solved_candidate_p57": float(v_re_solved),
            "deployed_action_structure_p57": float(v_deployed_action_structure),
            "gap": float(v_re_solved - v_deployed_action_structure),
            "shared_train_bins": _bins_summary(shared_train_bins, n_train_rank1),
            "shared_holdout_bins": _bins_summary(shared_holdout_bins, n_holdout_rank1),
            "action_table_comparison": _policy_action_comparison(
                re_solved_solution.policy_table,
                deployed_table,
                season_length=season_length,
            ),
        })

    overall = _summarize_gaps(rows)
    decision = _decision(overall)
    return {
        "schema_version": "gate_b_fair_comparator_rebaseline_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date": (today or date.today()).isoformat(),
        "artifact_role": "gate_b_fair_comparator_rebaseline",
        "production_deploy_claim": False,
        "writes_policy_artifact": False,
        "decision": decision,
        "decision_rule": (
            "A positive re-solve signal requires the re-solved candidate action "
            "table to meet or beat the deployed action structure on every "
            "reported holdout season, with positive aggregate mean gap. A null "
            "or negative result is informative: it means the earlier Gate B "
            "positive direction was explained by the boundary-scale fix rather "
            "than by re-optimizing the action table."
        ),
        "inputs": {
            "profiles_dir": str(profiles_dir),
            "prod_policy_path": str(prod_policy_path),
            "seasons": [int(season) for season in seasons],
            "n_bins": int(n_bins),
            "season_length": int(season_length),
            "require_estimated_basis": bool(require_estimated_basis),
        },
        "methodology": {
            "folds": "expanding_origin_train_prior_seasons_holdout_next_season",
            "first_season_role": "warmup_train_only",
            "shared_boundary_fit": (
                "Both arms classify holdout rows with equal-frequency boundaries "
                "fit only on prior estimated-PA profile seasons."
            ),
            "arm_a_re_solved_candidate": (
                "Fresh MDP action table solved on the same prior estimated-PA "
                "bins used for holdout classification."
            ),
            "arm_b_fair_comparator": (
                "Deployed policy action table applied by bin index to those same "
                "estimated-PA holdout bins. This varies the action table while "
                "holding the corrected boundary scale fixed."
            ),
            "comparator_caveat": (
                "The deployed action table was originally optimized on old "
                "actual-PA reward statistics and higher absolute probabilities. "
                "This fair comparator tests whether that relative action "
                "structure transfers to correctly scaled estimated-PA bins."
            ),
            "remaining_caveat": (
                "Estimated-PA profiles may still use actual historical lineup "
                "slot and batter universe; projected-lineup availability remains "
                "a separate production replay caveat."
            ),
        },
        "production_policy": {
            "path": str(prod_policy_path),
            "season_length": int(deployed_policy_length),
            "original_boundaries_not_used_for_fair_comparator": [
                float(x) for x in deployed_boundaries
            ],
        },
        "starter_matchup_drop_summary": _dropped_starter_matchup_summary(profiles),
        "overall": overall,
        "folds": rows,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES_DIR)
    parser.add_argument("--prod-policy-path", type=Path, default=DEFAULT_PROD_POLICY_PATH)
    parser.add_argument("--seasons", type=parse_seasons, default=DEFAULT_SEASONS)
    parser.add_argument("--n-bins", type=int, default=DEFAULT_N_BINS)
    parser.add_argument("--season-length", type=int, default=DEFAULT_SEASON_LENGTH)
    parser.add_argument("--allow-unmarked-profiles", action="store_true")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    today = date.fromisoformat(args.date)
    result = run_rebaseline(
        profiles_dir=args.profiles_dir,
        prod_policy_path=args.prod_policy_path,
        seasons=args.seasons,
        n_bins=args.n_bins,
        season_length=args.season_length,
        require_estimated_basis=not args.allow_unmarked_profiles,
        today=today,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    print(f"decision={result['decision']}")
    for fold in result["folds"]:
        print(
            f"  holdout={fold['holdout_season']} "
            f"re_solved={fold['re_solved_candidate_p57']:.10f} "
            f"deployed_action_structure={fold['deployed_action_structure_p57']:.10f} "
            f"gap={fold['gap']:+.10f} "
            f"same_actions={fold['action_table_comparison']['same_action_fraction']:.3f}"
        )
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
