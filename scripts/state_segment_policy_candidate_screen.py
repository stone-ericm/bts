#!/usr/bin/env python3
"""Rolling-origin screen for production-anchored state-segment policies.

This is a diagnostic follow-up to the Phase D pooled-policy falsification and
the first rolling-origin candidate screen. It tests whether the consumed 2025
early-phase lift is concentrated in coarse, pre-specified state regions.

The script is intentionally not a deployment path. It uses the already-consumed
2021-2025 Phase C profile surface for candidate generation only, keeps the
shipped production policy everywhere outside each segment, and writes a local
validation artifact instead of touching production policy files.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bts.simulate.mdp import load_policy
from bts.simulate.pooled_policy import evaluate_mdp_policy
from bts.validate.fdr import bh_qvalues, by_qvalues
from scripts.phase_d_pooled_policy_outer_eval import (
    DEFAULT_PROFILE_ROOTS,
    DEFAULT_PROD_POLICY_PATH,
    _bins_for_eval,
    discover_seed_records,
    load_profiles_for_seasons,
    parse_seasons,
)
from scripts.rolling_origin_policy_candidate_screen import (
    CandidateSpec,
    build_base_candidate_policy,
    summarize_gaps,
)


DEFAULT_OUTPUT = Path("data/validation/state_segment_policy_candidate_screen_2026-05-08.json")
SCHEMA_VERSION = "state_segment_policy_candidate_screen_v1"


@dataclass(frozen=True)
class StateSegment:
    name: str
    day_min: int
    day_max: int
    streak_min: int
    streak_max: int
    q_bins: tuple[int, ...]

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["q_bins"] = list(self.q_bins)
        out["q_labels"] = [f"Q{q + 1}" for q in self.q_bins]
        return out


DEFAULT_BASE_POLICIES = (CandidateSpec("cumulative", "cumulative"),)


DAY_BUCKETS = (
    ("late_d1_30", 1, 30),
    ("mid_d31_90", 31, 90),
    ("early_d91_180", 91, 180),
)

STREAK_BUCKETS = (
    ("s0_9", 0, 9),
    ("s10_29", 10, 29),
    ("s30_56", 30, 56),
)

QUALITY_GROUPS = (
    ("q1", (0,)),
    ("q2", (1,)),
    ("q3", (2,)),
    ("q4", (3,)),
    ("q5", (4,)),
)


def default_segments(n_bins: int = 5) -> tuple[StateSegment, ...]:
    if n_bins != 5:
        raise ValueError("default state-segment screen is pre-specified for 5 quality bins")
    segments: list[StateSegment] = []
    for day_name, day_min, day_max in DAY_BUCKETS:
        for streak_name, streak_min, streak_max in STREAK_BUCKETS:
            for quality_name, q_bins in QUALITY_GROUPS:
                segments.append(StateSegment(
                    name=f"{day_name}_{streak_name}_{quality_name}",
                    day_min=day_min,
                    day_max=day_max,
                    streak_min=streak_min,
                    streak_max=streak_max,
                    q_bins=tuple(q_bins),
                ))
    return tuple(segments)


def segment_mask(segment: StateSegment, policy_shape: tuple[int, ...]) -> np.ndarray:
    if len(policy_shape) != 4:
        raise ValueError(f"expected 4D policy table shape, got {policy_shape}")
    n_streak, n_days, _n_saver, n_bins = policy_shape
    if segment.streak_min < 0 or segment.streak_max >= n_streak:
        raise ValueError(f"segment {segment.name} streak bounds exceed policy shape")
    if segment.day_min < 1 or segment.day_max >= n_days:
        raise ValueError(f"segment {segment.name} day bounds exceed policy shape")
    if any(q < 0 or q >= n_bins for q in segment.q_bins):
        raise ValueError(f"segment {segment.name} quality bins exceed policy shape")

    mask = np.zeros(policy_shape, dtype=bool)
    streak_slice = slice(segment.streak_min, segment.streak_max + 1)
    day_slice = slice(segment.day_min, segment.day_max + 1)
    for q in segment.q_bins:
        mask[streak_slice, day_slice, :, q] = True
    return mask


def apply_state_segment(
    *,
    base_table: np.ndarray,
    prod_table: np.ndarray,
    segment: StateSegment,
) -> np.ndarray:
    if base_table.shape != prod_table.shape:
        raise ValueError(f"policy shape mismatch: {base_table.shape} vs {prod_table.shape}")
    mask = segment_mask(segment, prod_table.shape)
    table = prod_table.copy()
    table[mask] = base_table[mask]
    return table


def candidate_name(base_spec: CandidateSpec, segment: StateSegment) -> str:
    return f"{base_spec.name}__{segment.name}"


def table_diagnostics(
    *,
    table: np.ndarray,
    prod_table: np.ndarray,
    segment: StateSegment,
) -> dict[str, Any]:
    mask = segment_mask(segment, prod_table.shape)
    changed = table != prod_table
    changed_in_segment = changed & mask
    return {
        "segment_states": int(mask.sum()),
        "n_changed_states": int(changed.sum()),
        "n_changed_states_in_segment": int(changed_in_segment.sum()),
        "changed_fraction_of_policy": float(changed.sum() / changed.size),
        "changed_fraction_of_segment": (
            float(changed_in_segment.sum() / mask.sum()) if int(mask.sum()) else 0.0
        ),
    }


def fdr_table(
    summaries: dict[str, dict[str, Any]],
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    names = sorted(summaries)
    pvalues = np.array([
        float(summaries[name]["bootstrap"]["p_one_sided_positive"])
        for name in names
    ])
    q_bh = bh_qvalues(pvalues)
    q_by = by_qvalues(pvalues)

    rows = []
    for name, pvalue, bh, by in zip(names, pvalues, q_bh, q_by):
        summary = summaries[name]
        rows.append({
            "candidate": name,
            "mean_gap": float(summary["mean_gap"]),
            "n_positive": int(summary["n_positive"]),
            "n": int(summary["n"]),
            "p_one_sided_positive": float(pvalue),
            "q_BH": float(bh),
            "q_BY": float(by),
            "survives_BH_0_05": bool(summary["mean_gap"] > 0 and bh <= alpha),
            "survives_BY_0_05": bool(summary["mean_gap"] > 0 and by <= alpha),
        })
    rows.sort(key=lambda row: (row["p_one_sided_positive"], -row["mean_gap"], row["candidate"]))

    n_bh = int(sum(row["survives_BH_0_05"] for row in rows))
    n_by = int(sum(row["survives_BY_0_05"] for row in rows))
    if n_bh == 0:
        end_state = "E1_no_BH_survivors"
    elif n_bh <= 3:
        end_state = "E2_freeze_surviving_segments_for_fresh_lockbox"
    else:
        end_state = "E3_over_survival_revisit_family_control_before_conclusions"

    return {
        "method": "p-value FDR baseline using one-sided positive bootstrap p-values",
        "alpha": alpha,
        "m": int(len(rows)),
        "n_survive_BH_0_05": n_bh,
        "n_survive_BY_0_05": n_by,
        "end_state_by_BH": end_state,
        "rows": rows,
    }


def build_screen(
    *,
    roots: list[Path],
    prod_policy_path: Path,
    seasons: list[int],
    expect_seeds: int | None,
    base_specs: tuple[CandidateSpec, ...],
    segments: tuple[StateSegment, ...],
    season_length: int,
    late_phase_days: int,
    n_bins: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    if len(seasons) < 2:
        raise ValueError("state-segment screen requires at least two seasons")

    selection_seasons = seasons[:-1]
    outer_eval_seasons = [seasons[-1]]
    records, _metadata = discover_seed_records(
        roots,
        selection_seasons=selection_seasons,
        outer_eval_seasons=outer_eval_seasons,
        expect_seeds=expect_seeds,
    )
    profiles = load_profiles_for_seasons(records, seasons)
    prod_table, prod_boundaries, prod_policy_len = load_policy(prod_policy_path)
    record_by_seed = {record.seed: record for record in records}

    rows: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []

    for holdout_season in seasons[1:]:
        train_seasons = [season for season in seasons if season < holdout_season]
        train_profiles = profiles[profiles["season"].isin(train_seasons)].copy()
        holdout_profiles = profiles[profiles["season"] == holdout_season].copy()

        base_tables = {
            base_spec.name: build_base_candidate_policy(
                base_spec,
                train_profiles,
                train_seasons,
                season_length=season_length,
                late_phase_days=late_phase_days,
                n_bins=n_bins,
            )
            for base_spec in base_specs
        }
        candidate_tables: dict[str, np.ndarray] = {}
        candidate_metadata: dict[str, dict[str, Any]] = {}
        for base_spec in base_specs:
            for segment in segments:
                name = candidate_name(base_spec, segment)
                table = apply_state_segment(
                    base_table=base_tables[base_spec.name],
                    prod_table=prod_table,
                    segment=segment,
                )
                candidate_tables[name] = table
                candidate_metadata[name] = {
                    "base_policy": base_spec.name,
                    "segment": segment.to_json(),
                    "table_diagnostics": table_diagnostics(
                        table=table,
                        prod_table=prod_table,
                        segment=segment,
                    ),
                }

        fold_rows: list[dict[str, Any]] = []
        for seed, record in sorted(record_by_seed.items()):
            seed_holdout = holdout_profiles[holdout_profiles["seed"] == seed].copy()
            early_bins, late_bins, _diagnostics = _bins_for_eval(
                seed_holdout,
                late_phase_days=late_phase_days,
                n_bins=n_bins,
            )
            v_prod = evaluate_mdp_policy(
                prod_table,
                early_bins,
                season_length=season_length,
                late_bins=late_bins,
                late_phase_days=late_phase_days,
            )
            for name, table in candidate_tables.items():
                if candidate_metadata[name]["table_diagnostics"]["n_changed_states"] == 0:
                    v_candidate = v_prod
                else:
                    v_candidate = evaluate_mdp_policy(
                        table,
                        early_bins,
                        season_length=season_length,
                        late_bins=late_bins,
                        late_phase_days=late_phase_days,
                    )
                row = {
                    "holdout_season": int(holdout_season),
                    "train_seasons": train_seasons,
                    "candidate": name,
                    "base_policy": candidate_metadata[name]["base_policy"],
                    "segment": candidate_metadata[name]["segment"]["name"],
                    "seed": int(seed),
                    "provider": record.provider,
                    "v_prod_fixed_reference": float(v_prod),
                    "v_candidate": float(v_candidate),
                    "gap": float(v_candidate - v_prod),
                }
                rows.append(row)
                fold_rows.append(row)

        candidate_summaries = {
            name: summarize_gaps(
                [row for row in fold_rows if row["candidate"] == name],
                n_bootstrap=n_bootstrap,
                seed=1000 + int(holdout_season),
            )
            for name in candidate_tables
        }
        fold_summaries.append({
            "holdout_season": int(holdout_season),
            "train_seasons": train_seasons,
            "candidates": candidate_summaries,
            "table_diagnostics": candidate_metadata,
        })

    overall = {
        name: summarize_gaps(
            [row for row in rows if row["candidate"] == name],
            n_bootstrap=n_bootstrap,
            seed=1000,
        )
        for name in sorted({row["candidate"] for row in rows})
    }
    fdr = fdr_table(overall)
    leaderboard = [
        {"candidate": name, **summary}
        for name, summary in sorted(
            overall.items(),
            key=lambda item: item[1]["mean_gap"],
            reverse=True,
        )
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_role": "state_segment_policy_candidate_screen",
        "production_deploy_claim": False,
        "inputs": {
            "profile_roots": [root.as_posix() for root in roots],
            "prod_policy_path": prod_policy_path.as_posix(),
            "seasons": seasons,
            "expected_seeds": expect_seeds,
        },
        "methodology": {
            "folds": "rolling_origin_train_prior_seasons_holdout_next_season",
            "reference_policy": "Fixed shipped production policy loaded from prod_policy_path.",
            "candidate_construction": (
                "For each base pooled policy, copy candidate actions only inside one "
                "coarse pre-specified state segment and keep shipped production elsewhere."
            ),
            "candidate_warning": (
                "This artifact uses consumed 2021-2025 surfaces for candidate "
                "generation. It cannot produce a deployment-ready verdict."
            ),
            "season_length": season_length,
            "late_phase_days": late_phase_days,
            "n_bins": n_bins,
            "n_bootstrap": n_bootstrap,
            "base_policy_specs": [
                {
                    "name": spec.name,
                    "kind": spec.kind,
                    "half_life": spec.half_life,
                    "phase_scope": spec.phase_scope,
                }
                for spec in base_specs
            ],
            "segments": [segment.to_json() for segment in segments],
        },
        "production_policy": {
            "path": prod_policy_path.as_posix(),
            "season_length": int(prod_policy_len),
            "boundaries": [float(x) for x in prod_boundaries],
        },
        "seed_pool": {
            "n": len(records),
            "providers": {
                provider: int(sum(record.provider == provider for record in records))
                for provider in sorted({record.provider for record in records})
            },
        },
        "fdr": fdr,
        "leaderboard": leaderboard,
        "overall": overall,
        "fold_summaries": fold_summaries,
        "rows": rows,
        "interpretation": {
            "screen_only": True,
            "next_step": (
                "Use only as post-hoc candidate-generation evidence. Any promising "
                "segment needs a fresh lockbox or live-forward audit before deployment."
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-root", action="append", type=Path, dest="profile_roots")
    ap.add_argument("--prod-policy", type=Path, default=DEFAULT_PROD_POLICY_PATH)
    ap.add_argument("--seasons", default="2021,2022,2023,2024,2025")
    ap.add_argument("--expect-seeds", type=int, default=100)
    ap.add_argument("--season-length", type=int, default=180)
    ap.add_argument("--late-phase-days", type=int, default=30)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    roots = args.profile_roots if args.profile_roots else DEFAULT_PROFILE_ROOTS
    report = build_screen(
        roots=roots,
        prod_policy_path=args.prod_policy,
        seasons=parse_seasons(args.seasons),
        expect_seeds=args.expect_seeds,
        base_specs=DEFAULT_BASE_POLICIES,
        segments=default_segments(args.n_bins),
        season_length=args.season_length,
        late_phase_days=args.late_phase_days,
        n_bins=args.n_bins,
        n_bootstrap=args.n_bootstrap,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    for item in report["leaderboard"][:12]:
        print(
            f"  {item['candidate']}: mean_gap={item['mean_gap']:+.6f} "
            f"positive={item['n_positive']}/{item['n']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
