#!/usr/bin/env python3
"""Rolling-origin screen for next BTS policy candidates.

This is the first local follow-up after the Phase D pooled-policy
falsification. It evaluates policy families on rolling temporal folds so a
candidate must generalize forward in calendar time, not only across random
seeds.

The script is a screen only. It uses the consumed 2021-2025 Phase C profile
surface and includes diagnostic candidates. It does not produce a deploy-ready
claim and does not overwrite production policy files.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bts.simulate.mdp import load_policy, solve_mdp
from bts.simulate.pooled_policy import (
    build_pooled_policy,
    evaluate_mdp_policy,
    split_by_phase_pooled,
)
from bts.simulate.quality_bins import QualityBin, QualityBins
from scripts.phase_d_pooled_policy_outer_eval import (
    DEFAULT_PROFILE_ROOTS,
    DEFAULT_PROD_POLICY_PATH,
    _bins_for_eval,
    discover_seed_records,
    load_profiles_for_seasons,
    parse_seasons,
)


DEFAULT_OUTPUT = Path("data/validation/rolling_origin_policy_candidate_screen_2026-05-08.json")
SCHEMA_VERSION = "rolling_origin_policy_candidate_screen_v1"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    kind: str
    half_life: float | None = None
    phase_scope: str = "full"


DEFAULT_CANDIDATES = (
    CandidateSpec("cumulative_pooled", "cumulative"),
    CandidateSpec("last_season_pooled", "last_season"),
    CandidateSpec("decay_half_life_1", "decay", half_life=1.0),
    CandidateSpec("decay_half_life_2", "decay", half_life=2.0),
    CandidateSpec("prod_early_cumulative_late", "cumulative", phase_scope="late_only"),
    CandidateSpec("prod_early_last_season_late", "last_season", phase_scope="late_only"),
    CandidateSpec("prod_early_decay_hl1_late", "decay", half_life=1.0, phase_scope="late_only"),
    CandidateSpec("prod_early_decay_hl2_late", "decay", half_life=2.0, phase_scope="late_only"),
    CandidateSpec("cumulative_early_prod_late", "cumulative", phase_scope="early_only"),
    CandidateSpec("last_season_early_prod_late", "last_season", phase_scope="early_only"),
    CandidateSpec("decay_hl1_early_prod_late", "decay", half_life=1.0, phase_scope="early_only"),
    CandidateSpec("decay_hl2_early_prod_late", "decay", half_life=2.0, phase_scope="early_only"),
)


def season_decay_weights(train_seasons: list[int], half_life: float) -> dict[int, float]:
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    max_season = max(train_seasons)
    return {
        int(season): float(0.5 ** ((max_season - season) / half_life))
        for season in train_seasons
    }


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: list[float]) -> list[float]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        raise ValueError("weighted_quantile requires at least one value")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    if float(weights.sum()) <= 0:
        raise ValueError("weights must have positive total mass")
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cdf = np.cumsum(sorted_weights) / sorted_weights.sum()
    return [float(np.interp(q, cdf, sorted_values)) for q in quantiles]


def _weighted_rank_pairs(profiles: pd.DataFrame, season_weights: dict[int, float]) -> pd.DataFrame:
    r1 = profiles[profiles["rank"] == 1].copy()
    r2 = profiles[profiles["rank"] == 2].copy()
    merged = r1[["seed", "season", "date", "p_game_hit", "actual_hit"]].merge(
        r2[["seed", "season", "date", "actual_hit"]].rename(
            columns={"actual_hit": "top2_hit"}
        ),
        on=["seed", "season", "date"],
    )
    merged["weight"] = merged["season"].map(season_weights).astype(float)
    if merged["weight"].isna().any():
        missing = sorted(set(merged.loc[merged["weight"].isna(), "season"].astype(int)))
        raise ValueError(f"missing season weights for {missing}")
    return merged


def compute_weighted_pooled_bins(
    profiles: pd.DataFrame,
    *,
    season_weights: dict[int, float],
    n_bins: int,
) -> QualityBins:
    merged = _weighted_rank_pairs(profiles, season_weights)
    quantiles = [i / n_bins for i in range(1, n_bins)]
    boundaries = weighted_quantile(
        merged["p_game_hit"].to_numpy(),
        merged["weight"].to_numpy(),
        quantiles,
    )
    merged["bin"] = np.digitize(merged["p_game_hit"], boundaries)
    total_weight = float(merged["weight"].sum())
    bins: list[QualityBin] = []
    for i in range(n_bins):
        group = merged[merged["bin"] == i]
        if len(group) == 0:
            lower = float("-inf") if i == 0 else boundaries[i - 1]
            upper = float("inf") if i == len(boundaries) else boundaries[i]
            bins.append(QualityBin(
                index=i,
                p_range=(lower, upper),
                p_hit=0.0,
                p_both=0.0,
                frequency=0.0,
            ))
            continue
        weights = group["weight"].to_numpy(dtype=float)
        top1 = group["actual_hit"].to_numpy(dtype=float)
        both = (
            group["actual_hit"].astype(bool) & group["top2_hit"].astype(bool)
        ).to_numpy(dtype=float)
        bins.append(QualityBin(
            index=i,
            p_range=(float(group["p_game_hit"].min()), float(group["p_game_hit"].max())),
            p_hit=float(np.average(top1, weights=weights)),
            p_both=float(np.average(both, weights=weights)),
            frequency=float(weights.sum() / total_weight),
        ))
    return QualityBins(bins=bins, boundaries=boundaries)


def build_weighted_pooled_policy(
    profiles: pd.DataFrame,
    *,
    season_weights: dict[int, float],
    season_length: int,
    late_phase_days: int,
    n_bins: int,
):
    early_df, late_df = split_by_phase_pooled(profiles, late_phase_days)
    early_bins = compute_weighted_pooled_bins(
        early_df,
        season_weights=season_weights,
        n_bins=n_bins,
    )
    late_bins = None
    if late_phase_days > 0 and len(late_df) > 0:
        late_bins = compute_weighted_pooled_bins(
            late_df,
            season_weights=season_weights,
            n_bins=n_bins,
        )
    return solve_mdp(
        early_bins,
        season_length=season_length,
        late_bins=late_bins,
        late_phase_days=late_phase_days,
    )


def build_base_candidate_policy(
    spec: CandidateSpec,
    train_profiles: pd.DataFrame,
    train_seasons: list[int],
    *,
    season_length: int,
    late_phase_days: int,
    n_bins: int,
) -> np.ndarray:
    if spec.kind == "cumulative":
        return build_pooled_policy(
            train_profiles,
            season_length=season_length,
            late_phase_days=late_phase_days,
            n_bins=n_bins,
        ).policy_table
    if spec.kind == "last_season":
        last = max(train_seasons)
        last_profiles = train_profiles[train_profiles["season"] == last].copy()
        return build_pooled_policy(
            last_profiles,
            season_length=season_length,
            late_phase_days=late_phase_days,
            n_bins=n_bins,
        ).policy_table
    if spec.kind == "decay":
        if spec.half_life is None:
            raise ValueError(f"{spec.name} missing half_life")
        return build_weighted_pooled_policy(
            train_profiles,
            season_weights=season_decay_weights(train_seasons, spec.half_life),
            season_length=season_length,
            late_phase_days=late_phase_days,
            n_bins=n_bins,
        ).policy_table
    raise ValueError(f"unknown candidate kind: {spec.kind}")


def apply_phase_scope(
    *,
    spec: CandidateSpec,
    base_table: np.ndarray,
    prod_table: np.ndarray,
    late_phase_days: int,
) -> np.ndarray:
    """Anchor a candidate to production outside its requested phase scope."""
    if spec.phase_scope == "full":
        return base_table
    if base_table.shape != prod_table.shape:
        raise ValueError(f"policy shape mismatch: {base_table.shape} vs {prod_table.shape}")
    if late_phase_days <= 0:
        raise ValueError("phase-scoped candidates require late_phase_days > 0")

    scoped = prod_table.copy()
    late_end = min(late_phase_days, scoped.shape[1] - 1)
    if spec.phase_scope == "late_only":
        scoped[:, 1:late_end + 1, :, :] = base_table[:, 1:late_end + 1, :, :]
        return scoped
    if spec.phase_scope == "early_only":
        scoped[:, late_end + 1:, :, :] = base_table[:, late_end + 1:, :, :]
        return scoped
    raise ValueError(f"unknown phase_scope: {spec.phase_scope}")


def summarize_gaps(
    rows: list[dict[str, Any]],
    *,
    n_bootstrap: int = 0,
    seed: int = 42,
) -> dict[str, Any]:
    gaps = np.asarray([float(row["gap"]) for row in rows], dtype=float)
    if gaps.size == 0:
        raise ValueError("cannot summarize empty rows")
    n_positive = int(np.sum(gaps > 0))
    n_negative = int(np.sum(gaps < 0))
    n_zero = int(np.sum(gaps == 0))
    out = {
        "n": int(gaps.size),
        "mean_gap": float(gaps.mean()),
        "std_gap": float(gaps.std(ddof=1)) if gaps.size > 1 else 0.0,
        "se_gap": float(gaps.std(ddof=1) / math.sqrt(gaps.size)) if gaps.size > 1 else 0.0,
        "min_gap": float(gaps.min()),
        "max_gap": float(gaps.max()),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_zero": n_zero,
    }
    if n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, gaps.size, size=(n_bootstrap, gaps.size))
        means = gaps[idx].mean(axis=1)
        out["bootstrap"] = {
            "kind": "iid_seed_fold_bootstrap",
            "n_bootstrap": int(n_bootstrap),
            "seed": int(seed),
            "ci_lower": float(np.quantile(means, 0.025)),
            "ci_upper": float(np.quantile(means, 0.975)),
            "prob_mean_gt_zero": float(np.mean(means > 0)),
            "p_one_sided_positive": float((np.sum(means <= 0) + 1) / (n_bootstrap + 1)),
        }
    return out


def build_screen(
    *,
    roots: list[Path],
    prod_policy_path: Path,
    seasons: list[int],
    expect_seeds: int | None,
    candidate_specs: tuple[CandidateSpec, ...],
    season_length: int,
    late_phase_days: int,
    n_bins: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    if len(seasons) < 2:
        raise ValueError("rolling-origin screen requires at least two seasons")
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

    row_results: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    for holdout_season in seasons[1:]:
        train_seasons = [season for season in seasons if season < holdout_season]
        train_profiles = profiles[profiles["season"].isin(train_seasons)].copy()
        holdout_profiles = profiles[profiles["season"] == holdout_season].copy()
        base_tables = {}
        for spec in candidate_specs:
            key = (spec.kind, spec.half_life)
            if key not in base_tables:
                base_tables[key] = build_base_candidate_policy(
                    spec,
                    train_profiles,
                    train_seasons,
                    season_length=season_length,
                    late_phase_days=late_phase_days,
                    n_bins=n_bins,
                )
        candidate_tables = {
            spec.name: apply_phase_scope(
                spec=spec,
                base_table=base_tables[(spec.kind, spec.half_life)],
                prod_table=prod_table,
                late_phase_days=late_phase_days,
            )
            for spec in candidate_specs
        }

        fold_rows = []
        for seed, record in sorted(record_by_seed.items()):
            seed_holdout = holdout_profiles[holdout_profiles["seed"] == seed].copy()
            early_bins, late_bins, diagnostics = _bins_for_eval(
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
            for candidate_name, table in candidate_tables.items():
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
                    "candidate": candidate_name,
                    "seed": int(seed),
                    "provider": record.provider,
                    "v_prod_fixed_reference": float(v_prod),
                    "v_candidate": float(v_candidate),
                    "gap": float(v_candidate - v_prod),
                    "eval_diagnostics": diagnostics,
                }
                row_results.append(row)
                fold_rows.append(row)

        fold_summaries.append({
            "holdout_season": int(holdout_season),
            "train_seasons": train_seasons,
            "candidates": {
                spec.name: summarize_gaps([
                    row for row in fold_rows if row["candidate"] == spec.name
                ], n_bootstrap=n_bootstrap, seed=42 + int(holdout_season))
                for spec in candidate_specs
            },
        })

    overall = {
        spec.name: summarize_gaps([
            row for row in row_results if row["candidate"] == spec.name
        ], n_bootstrap=n_bootstrap, seed=42)
        for spec in candidate_specs
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_role": "rolling_origin_policy_candidate_screen",
        "production_deploy_claim": False,
        "inputs": {
            "profile_roots": [root.as_posix() for root in roots],
            "prod_policy_path": prod_policy_path.as_posix(),
            "seasons": seasons,
            "expected_seeds": expect_seeds,
        },
        "methodology": {
            "folds": "rolling_origin_train_prior_seasons_holdout_next_season",
            "reference_policy": (
                "Fixed shipped production policy. This is today's production "
                "reference, not a leak-free historical baseline for older folds."
            ),
            "candidate_warning": (
                "This artifact uses consumed 2021-2025 surfaces for candidate "
                "generation. It cannot produce a deployment-ready verdict."
            ),
            "season_length": season_length,
            "late_phase_days": late_phase_days,
            "n_bins": n_bins,
            "n_bootstrap": n_bootstrap,
            "candidate_specs": [
                {
                    "name": spec.name,
                    "kind": spec.kind,
                    "half_life": spec.half_life,
                    "phase_scope": spec.phase_scope,
                }
                for spec in candidate_specs
            ],
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
        "overall": overall,
        "fold_summaries": fold_summaries,
        "rows": row_results,
        "interpretation": {
            "screen_only": True,
            "next_step": (
                "Use this to choose a narrow recency/drift candidate family for a "
                "fresh pre-registered lockbox/live-forward audit."
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
    ap.add_argument("--n-bootstrap", type=int, default=20000)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    roots = args.profile_roots if args.profile_roots else DEFAULT_PROFILE_ROOTS
    report = build_screen(
        roots=roots,
        prod_policy_path=args.prod_policy,
        seasons=parse_seasons(args.seasons),
        expect_seeds=args.expect_seeds,
        candidate_specs=DEFAULT_CANDIDATES,
        season_length=args.season_length,
        late_phase_days=args.late_phase_days,
        n_bins=args.n_bins,
        n_bootstrap=args.n_bootstrap,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    leaderboard = sorted(
        report["overall"].items(),
        key=lambda item: item[1]["mean_gap"],
        reverse=True,
    )
    print(f"wrote {args.out}")
    for name, summary in leaderboard:
        print(
            f"  {name}: mean_gap={summary['mean_gap']:+.6f} "
            f"positive={summary['n_positive']}/{summary['n']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
