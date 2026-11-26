#!/usr/bin/env python3
"""Measure an exploratory Gate B PA-basis re-bin screen.

This diagnostic transforms historical backtest profile probabilities from
actual PA volume onto a production-like estimated PA volume, then re-bins and
re-solves the MDP on that transformed surface.

It is intentionally not a deployable policy gate. The preferred Gate B path is
to regenerate historical profiles with a real pre-game PA estimator comparable
to production. This script is only the quick screen from the PA-basis memo.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from bts.simulate.exact import exact_p57_policy_table
from bts.simulate.mdp import load_policy, solve_mdp
from bts.simulate.quality_bins import QualityBin, QualityBins


DEFAULT_N_BINS = (2, 3, 4, 5)
DEFAULT_MIN_PER_BIN = 30
DEFAULT_SEASON_LENGTH = 180
DEFAULT_RANK1_TARGET_PAS = 4.429310344827586
DEFAULT_RANK2_TARGET_PAS = 4.4222222222222225
DEFAULT_BOOTSTRAP_REPS = 0
DEFAULT_BOOTSTRAP_SEED = 42
PROFILE_COLUMNS = ["date", "rank", "p_game_hit", "actual_hit", "n_pas"]


def implied_pa_probability(p_game_hit: float, n_pas: float) -> float:
    """Invert product aggregation: p_game = 1 - (1 - p_pa) ** n_pas."""
    if n_pas <= 0:
        raise ValueError(f"n_pas must be positive, got {n_pas!r}")
    p = min(max(float(p_game_hit), 0.0), 1.0 - 1e-12)
    return float(1.0 - (1.0 - p) ** (1.0 / float(n_pas)))


def aggregate_pa_probability(p_hit_pa: float, target_pas: float) -> float:
    if target_pas <= 0:
        raise ValueError(f"target_pas must be positive, got {target_pas!r}")
    p = min(max(float(p_hit_pa), 0.0), 1.0 - 1e-12)
    return float(1.0 - (1.0 - p) ** float(target_pas))


def load_backtest_profiles(profiles_dir: Path) -> pd.DataFrame:
    files = sorted(profiles_dir.glob("backtest_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no backtest_*.parquet files in {profiles_dir}")

    frames = []
    for path in files:
        frame = pd.read_parquet(path, columns=PROFILE_COLUMNS)
        frame["source_file"] = path.name
        frames.append(frame)
    profiles = pd.concat(frames, ignore_index=True)
    profiles["rank"] = profiles["rank"].astype(int)
    return profiles[profiles["rank"].isin([1, 2])].copy()


def transform_profiles_to_target_pas(
    profiles: pd.DataFrame,
    *,
    rank1_target_pas: float,
    rank2_target_pas: float,
) -> pd.DataFrame:
    """Return rank-1/rank-2 profiles with p_game_hit on target PA volumes."""
    required = {"rank", "p_game_hit", "n_pas"}
    missing = required - set(profiles.columns)
    if missing:
        raise ValueError(f"profiles missing required columns: {sorted(missing)}")

    rank_target_pas = {
        1: float(rank1_target_pas),
        2: float(rank2_target_pas),
    }
    transformed = profiles[profiles["rank"].isin(rank_target_pas)].copy()
    transformed["source_p_game_hit"] = transformed["p_game_hit"].astype(float)
    transformed["source_n_pas"] = transformed["n_pas"].astype(float)
    transformed["pa_basis_target_pas"] = transformed["rank"].map(rank_target_pas)
    transformed["p_hit_pa_implied"] = [
        implied_pa_probability(p, n)
        for p, n in zip(
            transformed["source_p_game_hit"],
            transformed["source_n_pas"],
        )
    ]
    transformed["p_game_hit"] = [
        aggregate_pa_probability(p, n)
        for p, n in zip(
            transformed["p_hit_pa_implied"],
            transformed["pa_basis_target_pas"],
        )
    ]
    return transformed


def _series_summary(series: pd.Series) -> dict[str, Any]:
    values = series.astype(float)
    if values.empty:
        return {
            "n": 0,
            "min": None,
            "p20": None,
            "median": None,
            "mean": None,
            "p80": None,
            "max": None,
        }
    return {
        "n": int(values.shape[0]),
        "min": float(values.min()),
        "p20": float(values.quantile(0.20)),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "p80": float(values.quantile(0.80)),
        "max": float(values.max()),
    }


def rank_distribution_summary(profiles: pd.DataFrame, *, value_col: str = "p_game_hit") -> dict[str, Any]:
    return {
        str(rank): _series_summary(group[value_col])
        for rank, group in profiles.groupby("rank")
        if rank in (1, 2)
    }


def _bin_to_dict(bin_: QualityBin, total_n: int) -> dict[str, Any]:
    return {
        "index": int(bin_.index),
        "n": int(round(bin_.frequency * total_n)),
        "p_range": [float(bin_.p_range[0]), float(bin_.p_range[1])],
        "p_hit": float(bin_.p_hit),
        "p_both": float(bin_.p_both),
        "frequency": float(bin_.frequency),
    }


def pair_frame_from_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    r1 = profiles[profiles["rank"] == 1].copy()
    r2 = profiles[profiles["rank"] == 2].copy()

    return r1[["date", "p_game_hit", "actual_hit"]].merge(
        r2[["date", "actual_hit"]].rename(columns={"actual_hit": "top2_hit"}),
        on="date",
    )


def quality_bins_from_pairs(pairs: pd.DataFrame, n_bins: int) -> QualityBins:
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if pairs.empty:
        raise ValueError("cannot compute bins from zero rank-pair days")

    boundaries = [
        float(pairs["p_game_hit"].quantile(i / n_bins))
        for i in range(1, n_bins)
    ]
    assignments = np.digitize(pairs["p_game_hit"], boundaries)

    bins: list[QualityBin] = []
    for i in range(n_bins):
        group = pairs[assignments == i]
        if group.empty:
            continue
        bins.append(QualityBin(
            index=i,
            p_range=(float(group["p_game_hit"].min()), float(group["p_game_hit"].max())),
            p_hit=float(group["actual_hit"].mean()),
            p_both=float((group["actual_hit"] & group["top2_hit"]).mean()),
            frequency=float(len(group) / len(pairs)),
        ))
    return QualityBins(bins=bins, boundaries=boundaries)


def _classify(p_game_hit: float, boundaries: Sequence[float]) -> int:
    q = 0
    for boundary in boundaries:
        if p_game_hit >= boundary:
            q += 1
    return q


def _representative_p(bin_: QualityBin) -> float:
    lo, hi = bin_.p_range
    if np.isneginf(lo):
        return float(hi)
    if np.isposinf(hi):
        return float(lo)
    return float((lo + hi) / 2.0)


def project_policy_to_candidate_bins(
    policy_table: np.ndarray,
    policy_boundaries: Sequence[float],
    candidate_bins: QualityBins,
) -> tuple[np.ndarray, list[int]]:
    out = np.empty(
        (
            policy_table.shape[0],
            policy_table.shape[1],
            policy_table.shape[2],
            len(candidate_bins.bins),
        ),
        dtype=policy_table.dtype,
    )
    mapping: list[int] = []
    for bin_ in candidate_bins.bins:
        old_q = _classify(_representative_p(bin_), policy_boundaries)
        mapping.append(old_q)
        out[:, :, :, bin_.index] = policy_table[:, :, :, old_q]
    return out, mapping


def _p57_gap_for_pairs(
    pairs: pd.DataFrame,
    *,
    n_bins: int,
    policy_table: np.ndarray,
    policy_boundaries: Sequence[float],
    season_length: int,
) -> tuple[QualityBins, list[int], float, float, float]:
    candidate_bins = quality_bins_from_pairs(pairs, n_bins=n_bins)
    projected_policy, old_q_mapping = project_policy_to_candidate_bins(
        policy_table,
        policy_boundaries,
        candidate_bins,
    )
    candidate_solution = solve_mdp(candidate_bins, season_length=season_length)
    projected_baseline_p57 = exact_p57_policy_table(
        projected_policy,
        candidate_bins,
        season_length=season_length,
    )
    candidate_p57 = exact_p57_policy_table(
        candidate_solution.policy_table,
        candidate_bins,
        season_length=season_length,
    )
    return (
        candidate_bins,
        old_q_mapping,
        float(projected_baseline_p57),
        float(candidate_p57),
        float(candidate_p57 - projected_baseline_p57),
    )


def bootstrap_gap_interval(
    pairs: pd.DataFrame,
    *,
    n_bins: int,
    policy_table: np.ndarray,
    policy_boundaries: Sequence[float],
    season_length: int,
    n_bootstrap: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, Any] | None:
    if n_bootstrap <= 0:
        return None

    rng = np.random.default_rng(seed)
    gaps = np.zeros(n_bootstrap, dtype=float)
    n_pairs = len(pairs)
    for i in range(n_bootstrap):
        sample = pairs.iloc[rng.integers(n_pairs, size=n_pairs)].reset_index(drop=True)
        *_unused, gap = _p57_gap_for_pairs(
            sample,
            n_bins=n_bins,
            policy_table=policy_table,
            policy_boundaries=policy_boundaries,
            season_length=season_length,
        )
        gaps[i] = gap

    q_low = alpha / 2.0
    q_high = 1.0 - alpha / 2.0
    return {
        "kind": "rank_pair_day_iid_bootstrap_refit_bins_and_mdp",
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "alpha": float(alpha),
        "mean_gap": float(gaps.mean()),
        "ci95": [
            float(np.quantile(gaps, q_low)),
            float(np.quantile(gaps, q_high)),
        ],
        "p_gap_gt_zero": float((gaps > 0).mean()),
        "scope": (
            "uncertainty screen only; bootstrap resamples transformed rank-pair "
            "days and refits bins/MDP inside the same in-sample surface"
        ),
    }


def evaluate_pa_basis_rebin_candidates(
    transformed_profiles: pd.DataFrame,
    *,
    policy_path: Path,
    n_bins_values: Sequence[int] = DEFAULT_N_BINS,
    min_per_bin: int = DEFAULT_MIN_PER_BIN,
    season_length: int = DEFAULT_SEASON_LENGTH,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    policy_table, policy_boundaries, policy_length = load_policy(policy_path)
    evaluations = []
    pairs = pair_frame_from_profiles(transformed_profiles)

    for n_bins in n_bins_values:
        (
            candidate_bins,
            old_q_mapping,
            projected_baseline_p57,
            candidate_p57,
            gap,
        ) = _p57_gap_for_pairs(
            pairs,
            n_bins=n_bins,
            policy_table=policy_table,
            policy_boundaries=policy_boundaries,
            season_length=season_length,
        )
        bootstrap = bootstrap_gap_interval(
            pairs,
            n_bins=n_bins,
            policy_table=policy_table,
            policy_boundaries=policy_boundaries,
            season_length=season_length,
            n_bootstrap=bootstrap_reps,
            seed=bootstrap_seed + int(n_bins),
        )
        bin_counts = [int(round(bin_.frequency * len(pairs))) for bin_ in candidate_bins.bins]
        evaluations.append({
            "n_bins": int(n_bins),
            "boundaries": [float(x) for x in candidate_bins.boundaries],
            "projected_policy_old_q_mapping": old_q_mapping,
            "min_bin_n": min(bin_counts) if bin_counts else 0,
            "bins": [_bin_to_dict(bin_, len(pairs)) for bin_ in candidate_bins.bins],
            "projected_baseline_p57": float(projected_baseline_p57),
            "candidate_optimal_p57": float(candidate_p57),
            "gap": float(gap),
            "bootstrap_gap": bootstrap,
        })

    if any(item["min_bin_n"] < min_per_bin for item in evaluations):
        decision = "INSUFFICIENT_SUPPORT"
        reason = f"at least one evaluated binning has min_bin_n below {min_per_bin}"
    elif max(item["gap"] for item in evaluations) <= 0:
        decision = "NO_POINT_IMPROVEMENT"
        reason = "no PA-basis re-bin candidate improves point P(57)"
    else:
        decision = "SCREEN_SIGNAL_REQUIRES_POLICY_FILE_BACKTEST"
        reason = (
            "point P(57) improved on the transformed surface; run a proper "
            "production-PA-consistent walk-forward policy-file harness"
        )

    return {
        "policy_path": str(policy_path),
        "policy_boundaries": [float(x) for x in policy_boundaries],
        "policy_season_length": int(policy_length),
        "season_length": int(season_length),
        "min_per_bin": int(min_per_bin),
        "n_rank_pair_days": int(len(pairs)),
        "bootstrap_reps": int(bootstrap_reps),
        "bootstrap_seed": int(bootstrap_seed),
        "decision": decision,
        "reason": reason,
        "evaluations": evaluations,
    }


def run_measurement(
    *,
    profiles_dir: Path,
    policy_path: Path,
    rank1_target_pas: float = DEFAULT_RANK1_TARGET_PAS,
    rank2_target_pas: float = DEFAULT_RANK2_TARGET_PAS,
    n_bins_values: Sequence[int] = DEFAULT_N_BINS,
    min_per_bin: int = DEFAULT_MIN_PER_BIN,
    season_length: int = DEFAULT_SEASON_LENGTH,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    today: date | None = None,
) -> dict[str, Any]:
    raw_profiles = load_backtest_profiles(profiles_dir)
    transformed = transform_profiles_to_target_pas(
        raw_profiles,
        rank1_target_pas=rank1_target_pas,
        rank2_target_pas=rank2_target_pas,
    )
    today = today or date.today()

    return {
        "schema_version": "pa_basis_rebin_gate_measure_v1",
        "artifact_role": "gate_b_pa_basis_rebin_screen",
        "production_deploy_claim": False,
        "heavy_compute": False,
        "date": today.isoformat(),
        "inputs": {
            "profiles_dir": str(profiles_dir),
            "policy_path": str(policy_path),
            "n_bins_values": [int(x) for x in n_bins_values],
            "bootstrap_reps": int(bootstrap_reps),
            "bootstrap_seed": int(bootstrap_seed),
            "rank_target_pas": {
                "1": float(rank1_target_pas),
                "2": float(rank2_target_pas),
            },
        },
        "raw_distribution": rank_distribution_summary(raw_profiles),
        "pa_basis_distribution": rank_distribution_summary(transformed),
        "source_n_pas_distribution": rank_distribution_summary(raw_profiles, value_col="n_pas"),
        "gate_b_screen": evaluate_pa_basis_rebin_candidates(
            transformed,
            policy_path=policy_path,
            n_bins_values=n_bins_values,
            min_per_bin=min_per_bin,
            season_length=season_length,
            bootstrap_reps=bootstrap_reps,
            bootstrap_seed=bootstrap_seed,
        ),
        "methodology": {
            "candidate": (
                "actual-PA backtest p_game_hit values transformed to production-like "
                "estimated PA volumes by inverting the product aggregation"
            ),
            "not_full_gate": (
                "this is an exploratory support and point-measurement screen; the "
                "reported P(57) values are in-sample/optimistic because bins, MDP "
                "solve, and evaluation use the same transformed 2021-2025 rows; a "
                "deployable Gate B claim still needs historical profiles generated "
                "with a pre-game PA estimator and a walk-forward policy-file P(57) "
                "evaluation"
            ),
            "preferred_next_step": (
                "regenerate historical profiles with a production-comparable "
                "pre-game PA estimator rather than relying on this actual-to-target "
                "PA transform"
            ),
            "baseline_projection": (
                "saved production policy table projected onto transformed candidate "
                "bins by representative primary p; exact only when bins do not cross "
                "saved policy boundaries"
            ),
        },
    }


def _parse_n_bins(value: str) -> tuple[int, ...]:
    out = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not out:
        raise argparse.ArgumentTypeError("expected comma-separated n_bins values")
    if any(x < 1 for x in out):
        raise argparse.ArgumentTypeError("n_bins values must be >= 1")
    return out


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles-dir", type=Path, default=Path("data/simulation"))
    parser.add_argument("--policy-path", type=Path, default=Path("data/models/mdp_policy.npz"))
    parser.add_argument("--rank1-target-pas", type=float, default=DEFAULT_RANK1_TARGET_PAS)
    parser.add_argument("--rank2-target-pas", type=float, default=DEFAULT_RANK2_TARGET_PAS)
    parser.add_argument("--n-bins", type=_parse_n_bins, default=DEFAULT_N_BINS)
    parser.add_argument("--min-per-bin", type=int, default=DEFAULT_MIN_PER_BIN)
    parser.add_argument("--season-length", type=int, default=DEFAULT_SEASON_LENGTH)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    today = date.fromisoformat(args.date)
    if not args.policy_path.exists():
        raise SystemExit(f"missing policy artifact: {args.policy_path}")
    output = args.output or Path(f"data/validation/pa_basis_rebin_gate_{today.isoformat()}.json")
    result = run_measurement(
        profiles_dir=args.profiles_dir,
        policy_path=args.policy_path,
        rank1_target_pas=args.rank1_target_pas,
        rank2_target_pas=args.rank2_target_pas,
        n_bins_values=args.n_bins,
        min_per_bin=args.min_per_bin,
        season_length=args.season_length,
        bootstrap_reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
        today=today,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))

    gate = result["gate_b_screen"]
    rank1 = result["pa_basis_distribution"]["1"]
    print(
        f"Loaded {rank1['n']} rank-1 backtest days transformed to "
        f"{args.rank1_target_pas:.3f} target PAs"
    )
    print(f"Gate B PA-basis decision={gate['decision']} reason={gate['reason']}")
    for item in gate["evaluations"]:
        print(
            f"  n_bins={item['n_bins']} min_bin_n={item['min_bin_n']} "
            f"gap={item['gap']:.10f}"
        )
        bootstrap = item.get("bootstrap_gap")
        if bootstrap is not None:
            ci_low, ci_high = bootstrap["ci95"]
            print(
                f"    bootstrap_gap_ci95=[{ci_low:.10f}, {ci_high:.10f}] "
                f"p_gap_gt_zero={bootstrap['p_gap_gt_zero']:.3f}"
            )
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
