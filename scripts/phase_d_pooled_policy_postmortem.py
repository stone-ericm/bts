#!/usr/bin/env python3
"""Post-mortem diagnostics for the Phase D pooled-policy falsification.

This script is deliberately diagnostic-only. It reuses the Phase D profile
surface, compares the 2021-2024 selection manifold to the 2025 outer manifold,
and evaluates a small set of policy references to explain the failure mode.

It does not save or overwrite any production policy file.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bts.simulate.mdp import load_policy
from bts.simulate.pooled_policy import build_pooled_policy, evaluate_mdp_policy
from scripts.phase_d_pooled_policy_outer_eval import (
    DEFAULT_PROFILE_ROOTS,
    DEFAULT_PROD_POLICY_PATH,
    _bins_for_eval,
    _quality_bins_summary,
    discover_seed_records,
    load_profiles_for_seasons,
    parse_seasons,
)


DEFAULT_OUTPUT = Path("data/validation/phase_d_pooled_policy_postmortem_2026-05-08.json")
SCHEMA_VERSION = "phase_d_pooled_policy_postmortem_v1"
ACTION_NAMES = {0: "skip", 1: "single", 2: "double"}


def _decision_states(policy_table: np.ndarray) -> np.ndarray:
    """Return non-terminal, positive-days decision states only."""
    return policy_table[:57, 1:, :, :]


def action_counts(policy_table: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(_decision_states(policy_table), return_counts=True)
    return {
        ACTION_NAMES[int(value)]: int(count)
        for value, count in zip(values, counts)
    }


def action_transition_counts(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    """Count action changes between two policy tables on decision states."""
    ref = _decision_states(reference).ravel()
    cand = _decision_states(candidate).ravel()
    if ref.shape != cand.shape:
        raise ValueError(f"policy shape mismatch: {ref.shape} vs {cand.shape}")
    pairs: dict[str, int] = {}
    for from_action, to_action in zip(ref, cand):
        key = f"{ACTION_NAMES[int(from_action)]}->{ACTION_NAMES[int(to_action)]}"
        pairs[key] = pairs.get(key, 0) + 1
    total = int(ref.size)
    differing = int(np.sum(ref != cand))
    return {
        "total_decision_states": total,
        "differing_states": differing,
        "differing_fraction": differing / total if total else 0.0,
        "transition_counts": dict(sorted(pairs.items())),
    }


def compare_bin_summaries(
    selection: dict[str, Any] | None,
    outer: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if selection is None or outer is None:
        return None
    rows = []
    for sel_bin, out_bin in zip(selection["bins"], outer["bins"]):
        rows.append({
            "index": int(sel_bin["index"]),
            "selection_p_hit": float(sel_bin["p_hit"]),
            "outer_p_hit": float(out_bin["p_hit"]),
            "delta_p_hit": float(out_bin["p_hit"] - sel_bin["p_hit"]),
            "selection_p_both": float(sel_bin["p_both"]),
            "outer_p_both": float(out_bin["p_both"]),
            "delta_p_both": float(out_bin["p_both"] - sel_bin["p_both"]),
            "selection_frequency": float(sel_bin["frequency"]),
            "outer_frequency": float(out_bin["frequency"]),
            "delta_frequency": float(out_bin["frequency"] - sel_bin["frequency"]),
        })
    return {
        "selection_boundaries": selection["boundaries"],
        "outer_boundaries": outer["boundaries"],
        "boundary_deltas": [
            float(out - sel)
            for sel, out in zip(selection["boundaries"], outer["boundaries"])
        ],
        "bins": rows,
    }


def rank_pair_metrics(profiles: pd.DataFrame) -> dict[str, Any]:
    """Summarize rank-1 and rank-1/rank-2 paired outcomes on a profile surface."""
    rank1 = profiles[profiles["rank"] == 1].copy()
    rank2 = profiles[profiles["rank"] == 2].copy()
    paired = rank1[["seed", "season", "date", "p_game_hit", "actual_hit"]].merge(
        rank2[["seed", "season", "date", "actual_hit"]].rename(
            columns={"actual_hit": "rank2_actual_hit"}
        ),
        on=["seed", "season", "date"],
    )
    if len(paired) == 0:
        raise ValueError("profile surface has no paired rank-1/rank-2 days")
    both = (paired["actual_hit"].astype(bool) & paired["rank2_actual_hit"].astype(bool))
    return {
        "n_rows": int(len(profiles)),
        "n_seed_dates": int(len(paired)),
        "n_seeds": int(profiles["seed"].nunique()),
        "seasons": sorted(int(s) for s in profiles["season"].unique()),
        "rank1_mean_p_game_hit": float(rank1["p_game_hit"].mean()),
        "rank1_actual_hit_rate": float(rank1["actual_hit"].mean()),
        "rank1_calibration_gap_mean_p_minus_actual": float(
            rank1["p_game_hit"].mean() - rank1["actual_hit"].mean()
        ),
        "rank2_actual_hit_rate": float(rank2["actual_hit"].mean()),
        "rank1_rank2_both_hit_rate": float(both.mean()),
    }


def evaluate_references(
    *,
    prod_policy_path: Path,
    selection_profiles: pd.DataFrame,
    outer_profiles: pd.DataFrame,
    season_length: int,
    late_phase_days: int,
    n_bins: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    prod_table, prod_boundaries, prod_policy_len = load_policy(prod_policy_path)
    selection_solution = build_pooled_policy(
        selection_profiles,
        season_length=season_length,
        late_phase_days=late_phase_days,
        n_bins=n_bins,
    )
    full_solution = build_pooled_policy(
        pd.concat([selection_profiles, outer_profiles], ignore_index=True),
        season_length=season_length,
        late_phase_days=late_phase_days,
        n_bins=n_bins,
    )
    outer_solution = build_pooled_policy(
        outer_profiles,
        season_length=season_length,
        late_phase_days=late_phase_days,
        n_bins=n_bins,
    )

    surfaces = {}
    for name, profiles in [("selection", selection_profiles), ("outer", outer_profiles)]:
        early_bins, late_bins, diagnostics = _bins_for_eval(
            profiles,
            late_phase_days=late_phase_days,
            n_bins=n_bins,
        )
        surfaces[name] = {
            "early_bins": early_bins,
            "late_bins": late_bins,
            "diagnostics": diagnostics,
        }

    policies = {
        "production_shipped": prod_table,
        "selection_pooled_candidate": selection_solution.policy_table,
        "full_surface_pooled_leaky_diagnostic": full_solution.policy_table,
        "outer_only_hindsight_oracle": outer_solution.policy_table,
    }

    evaluations: dict[str, Any] = {}
    for surface_name, surface in surfaces.items():
        evaluations[surface_name] = {
            policy_name: float(evaluate_mdp_policy(
                table,
                surface["early_bins"],
                season_length=season_length,
                late_bins=surface["late_bins"],
                late_phase_days=late_phase_days,
            ))
            for policy_name, table in policies.items()
        }

    report = {
        "production_policy": {
            "path": prod_policy_path.as_posix(),
            "season_length": int(prod_policy_len),
            "boundaries": [float(x) for x in prod_boundaries],
        },
        "policy_references": {
            "production_shipped": (
                "Fixed local production policy loaded from data/models/mdp_policy.npz."
            ),
            "selection_pooled_candidate": (
                "Pooled policy built from 2021-2024 profiles only."
            ),
            "full_surface_pooled_leaky_diagnostic": (
                "Pooled policy built from 2021-2025 profiles; diagnostic only because it "
                "uses the outer-evaluation year."
            ),
            "outer_only_hindsight_oracle": (
                "Pooled policy built from 2025 profiles only; diagnostic oracle only."
            ),
        },
        "evaluations": evaluations,
        "action_counts": {
            name: action_counts(table)
            for name, table in policies.items()
        },
        "action_transitions_vs_production": {
            name: action_transition_counts(prod_table, table)
            for name, table in policies.items()
            if name != "production_shipped"
        },
        "selection_solution": {
            "optimal_p57": float(selection_solution.optimal_p57),
        },
        "full_surface_solution": {
            "optimal_p57": float(full_solution.optimal_p57),
            "diagnostic_only": True,
        },
        "outer_oracle_solution": {
            "optimal_p57": float(outer_solution.optimal_p57),
            "diagnostic_only": True,
        },
    }
    return report, policies


def build_report(
    *,
    roots: list[Path],
    prod_policy_path: Path,
    selection_seasons: list[int],
    outer_eval_seasons: list[int],
    expect_seeds: int | None,
    season_length: int,
    late_phase_days: int,
    n_bins: int,
) -> dict[str, Any]:
    records, _root_metadata = discover_seed_records(
        roots,
        selection_seasons=selection_seasons,
        outer_eval_seasons=outer_eval_seasons,
        expect_seeds=expect_seeds,
    )
    selection_profiles = load_profiles_for_seasons(records, selection_seasons)
    outer_profiles = load_profiles_for_seasons(records, outer_eval_seasons)

    selection_early, selection_late, selection_diag = _bins_for_eval(
        selection_profiles,
        late_phase_days=late_phase_days,
        n_bins=n_bins,
    )
    outer_early, outer_late, outer_diag = _bins_for_eval(
        outer_profiles,
        late_phase_days=late_phase_days,
        n_bins=n_bins,
    )
    policy_report, _policies = evaluate_references(
        prod_policy_path=prod_policy_path,
        selection_profiles=selection_profiles,
        outer_profiles=outer_profiles,
        season_length=season_length,
        late_phase_days=late_phase_days,
        n_bins=n_bins,
    )

    selection_metrics = rank_pair_metrics(selection_profiles)
    outer_metrics = rank_pair_metrics(outer_profiles)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_role": "phase_d_postmortem_diagnostic",
        "production_deploy_claim": False,
        "split": {
            "selection_seasons": selection_seasons,
            "outer_eval_seasons": outer_eval_seasons,
        },
        "seed_pool": {
            "n": len(records),
            "providers": {
                provider: int(sum(record.provider == provider for record in records))
                for provider in sorted({record.provider for record in records})
            },
        },
        "surface_metrics": {
            "selection": selection_metrics,
            "outer": outer_metrics,
            "outer_minus_selection": {
                key: float(outer_metrics[key] - selection_metrics[key])
                for key in [
                    "rank1_mean_p_game_hit",
                    "rank1_actual_hit_rate",
                    "rank1_calibration_gap_mean_p_minus_actual",
                    "rank2_actual_hit_rate",
                    "rank1_rank2_both_hit_rate",
                ]
            },
        },
        "bin_manifold": {
            "selection_diagnostics": selection_diag,
            "outer_diagnostics": outer_diag,
            "early": compare_bin_summaries(
                _quality_bins_summary(selection_early),
                _quality_bins_summary(outer_early),
            ),
            "late": compare_bin_summaries(
                _quality_bins_summary(selection_late),
                _quality_bins_summary(outer_late),
            ),
        },
        "policy_diagnostics": policy_report,
        "interpretation": {
            "diagnostic_only": True,
            "summary": (
                "This artifact explains the Phase D failure mode. It must not be "
                "used as a production deployment claim because it includes hindsight "
                "policy references trained on the outer-evaluation surface."
            ),
            "next_candidate_direction": (
                "Treat plain pooled-policy as falsified under temporal outer evaluation. "
                "Use rolling-origin candidate selection for any recency-weighted, "
                "drift-aware, or robust-policy successor, with a fresh lockbox/live "
                "evaluation target."
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-root", action="append", type=Path, dest="profile_roots")
    ap.add_argument("--prod-policy", type=Path, default=DEFAULT_PROD_POLICY_PATH)
    ap.add_argument("--selection-seasons", default="2021,2022,2023,2024")
    ap.add_argument("--outer-eval-seasons", default="2025")
    ap.add_argument("--expect-seeds", type=int, default=100)
    ap.add_argument("--season-length", type=int, default=180)
    ap.add_argument("--late-phase-days", type=int, default=30)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    roots = args.profile_roots if args.profile_roots else DEFAULT_PROFILE_ROOTS
    report = build_report(
        roots=roots,
        prod_policy_path=args.prod_policy,
        selection_seasons=parse_seasons(args.selection_seasons),
        outer_eval_seasons=parse_seasons(args.outer_eval_seasons),
        expect_seeds=args.expect_seeds,
        season_length=args.season_length,
        late_phase_days=args.late_phase_days,
        n_bins=args.n_bins,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    outer = report["policy_diagnostics"]["evaluations"]["outer"]
    print(
        "wrote {path} | outer production={prod:.6f} selection_pooled={pool:.6f} "
        "outer_oracle={oracle:.6f}".format(
            path=args.out,
            prod=outer["production_shipped"],
            pool=outer["selection_pooled_candidate"],
            oracle=outer["outer_only_hindsight_oracle"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
