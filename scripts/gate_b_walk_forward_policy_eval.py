#!/usr/bin/env python3
"""Gate B production-PA-basis walk-forward policy evaluation.

This is an evidence-only harness. It consumes profiles whose ``p_game_hit`` was
generated on the production-comparable estimated-PA basis, fits candidate bins
and an MDP policy on prior seasons only, and evaluates both that candidate and
the deployed policy on the same held-out future season stream.

It never writes or swaps a production policy artifact.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from bts.simulate.mdp import load_policy, solve_mdp
from bts.simulate.pooled_policy import evaluate_mdp_policy
from bts.simulate.quality_bins import QualityBins, compute_bins, compute_bins_with_boundaries


DEFAULT_SEASONS = (2021, 2022, 2023, 2024, 2025)
DEFAULT_PROFILES_DIR = Path("data/simulation_estimated_pa")
DEFAULT_PROD_POLICY_PATH = Path("data/models/mdp_policy.npz")
DEFAULT_OUTPUT = Path("data/validation/gate_b_walk_forward_policy_eval_2026-05-24.json")
DEFAULT_N_BINS = 5
DEFAULT_SEASON_LENGTH = 180
REQUIRED_PROFILE_COLUMNS = {"date", "rank", "p_game_hit", "actual_hit"}
EXPECTED_BASIS = "estimated_pa"


def parse_seasons(raw: str) -> list[int]:
    seasons = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if len(seasons) < 2:
        raise ValueError("walk-forward evaluation requires at least two seasons")
    return seasons


def _season_from_path(path: Path) -> int:
    try:
        return int(path.stem.replace("backtest_", ""))
    except ValueError as exc:
        raise ValueError(f"cannot parse season from {path}") from exc


def load_profiles(
    profiles_dir: Path,
    seasons: Sequence[int],
    *,
    require_estimated_basis: bool = True,
) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = profiles_dir / f"backtest_{season}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing profile parquet: {path}")
        frame = pd.read_parquet(path)
        missing = sorted(REQUIRED_PROFILE_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        if "season" not in frame.columns:
            frame["season"] = _season_from_path(path)
        if require_estimated_basis:
            if "p_game_hit_basis" not in frame.columns:
                raise ValueError(f"{path} missing p_game_hit_basis for estimated-PA gate")
            bad = frame["p_game_hit_basis"].dropna().astype(str) != EXPECTED_BASIS
            if bool(bad.any()):
                raise ValueError(f"{path} contains non-{EXPECTED_BASIS} profile rows")
        frames.append(frame)
    profiles = pd.concat(frames, ignore_index=True)
    profiles["season"] = profiles["season"].astype(int)
    return profiles[profiles["rank"].isin([1, 2])].copy()


def _bin_counts(bins: QualityBins, n_rank1: int) -> list[int]:
    return [int(round(bin_.frequency * n_rank1)) for bin_ in bins.bins]


def _bins_summary(bins: QualityBins, n_rank1: int) -> dict[str, Any]:
    counts = _bin_counts(bins, n_rank1)
    return {
        "n_bins": int(len(bins.bins)),
        "boundaries": [float(x) for x in bins.boundaries],
        "bin_counts": counts,
        "min_bin_n": min(counts) if counts else 0,
        "bins": [
            {
                "index": int(bin_.index),
                "n": int(count),
                "p_range": [float(bin_.p_range[0]), float(bin_.p_range[1])],
                "p_hit": float(bin_.p_hit),
                "p_both": float(bin_.p_both),
                "frequency": float(bin_.frequency),
            }
            for bin_, count in zip(bins.bins, counts)
        ],
    }


def _summarize_gaps(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gaps = np.asarray([row["gap"] for row in rows], dtype=float)
    if gaps.size == 0:
        raise ValueError("cannot summarize zero folds")
    return {
        "n_folds": int(gaps.size),
        "mean_gap": float(gaps.mean()),
        "std_gap": float(gaps.std(ddof=1)) if gaps.size > 1 else 0.0,
        "min_gap": float(gaps.min()),
        "max_gap": float(gaps.max()),
        "n_nonnegative": int((gaps >= 0).sum()),
        "n_negative": int((gaps < 0).sum()),
    }


def _dropped_starter_matchup_summary(profiles: pd.DataFrame) -> dict[str, Any]:
    required = {
        "dropped_no_starter_matchup",
        "total_batter_games",
        "starter_matchup_batter_games",
    }
    if not required.issubset(profiles.columns):
        return {
            "available": False,
            "reason": "estimated-PA drop accounting columns not present",
        }

    daily = (
        profiles.groupby(["season", "date"], dropna=False)
        .agg(
            dropped_no_starter_matchup=("dropped_no_starter_matchup", "max"),
            total_batter_games=("total_batter_games", "max"),
            starter_matchup_batter_games=("starter_matchup_batter_games", "max"),
        )
        .reset_index()
    )
    by_season = []
    for season, group in daily.groupby("season", sort=True):
        dropped = int(group["dropped_no_starter_matchup"].sum())
        total = int(group["total_batter_games"].sum())
        by_season.append({
            "season": int(season),
            "profile_days": int(group["date"].nunique()),
            "dropped_no_starter_matchup": dropped,
            "total_batter_games": total,
            "starter_matchup_batter_games": int(group["starter_matchup_batter_games"].sum()),
            "dropped_fraction": None if total == 0 else float(dropped / total),
            "mean_dropped_per_day": float(group["dropped_no_starter_matchup"].mean()),
        })

    total_dropped = int(daily["dropped_no_starter_matchup"].sum())
    total_batter_games = int(daily["total_batter_games"].sum())
    return {
        "available": True,
        "overall": {
            "profile_days": int(daily["date"].nunique()),
            "dropped_no_starter_matchup": total_dropped,
            "total_batter_games": total_batter_games,
            "starter_matchup_batter_games": int(daily["starter_matchup_batter_games"].sum()),
            "dropped_fraction": (
                None if total_batter_games == 0 else float(total_dropped / total_batter_games)
            ),
            "mean_dropped_per_day": float(daily["dropped_no_starter_matchup"].mean()),
        },
        "by_season": by_season,
    }


def _decision(overall: dict[str, Any]) -> str:
    if overall["mean_gap"] > 0 and overall["n_negative"] == 0:
        return "WALK_FORWARD_SIGNAL_POSITIVE_REQUIRES_REBASELINE"
    if overall["mean_gap"] > 0:
        return "MIXED_FOLD_SIGNAL_REQUIRES_REVIEW"
    return "NO_WALK_FORWARD_IMPROVEMENT"


def run_evaluation(
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
    prod_table, prod_boundaries, prod_policy_length = load_policy(prod_policy_path)
    rows = []

    for holdout_season in seasons[1:]:
        train_seasons = [season for season in seasons if season < holdout_season]
        train_profiles = profiles[profiles["season"].isin(train_seasons)].copy()
        holdout_profiles = profiles[profiles["season"] == holdout_season].copy()
        if train_profiles.empty or holdout_profiles.empty:
            raise ValueError(f"empty train/holdout fold for {holdout_season}")

        candidate_train_bins = compute_bins(train_profiles, n_bins=n_bins)
        candidate_solution = solve_mdp(candidate_train_bins, season_length=season_length)

        candidate_holdout_bins = compute_bins_with_boundaries(
            holdout_profiles,
            candidate_train_bins.boundaries,
        )
        prod_holdout_bins = compute_bins_with_boundaries(
            holdout_profiles,
            prod_boundaries,
        )

        v_candidate = evaluate_mdp_policy(
            candidate_solution.policy_table,
            candidate_holdout_bins,
            season_length=season_length,
        )
        v_prod = evaluate_mdp_policy(
            prod_table,
            prod_holdout_bins,
            season_length=season_length,
        )
        n_train_rank1 = int((train_profiles["rank"] == 1).sum())
        n_holdout_rank1 = int((holdout_profiles["rank"] == 1).sum())
        rows.append({
            "holdout_season": int(holdout_season),
            "train_seasons": [int(season) for season in train_seasons],
            "n_train_rank1": n_train_rank1,
            "n_holdout_rank1": n_holdout_rank1,
            "candidate_p57": float(v_candidate),
            "deployed_baseline_p57": float(v_prod),
            "gap": float(v_candidate - v_prod),
            "candidate_train_bins": _bins_summary(candidate_train_bins, n_train_rank1),
            "candidate_holdout_bins": _bins_summary(candidate_holdout_bins, n_holdout_rank1),
            "deployed_holdout_bins": _bins_summary(prod_holdout_bins, n_holdout_rank1),
        })

    overall = _summarize_gaps(rows)
    decision = _decision(overall)
    return {
        "schema_version": "gate_b_walk_forward_policy_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date": (today or date.today()).isoformat(),
        "artifact_role": "gate_b_production_pa_basis_walk_forward_policy_eval",
        "production_deploy_claim": False,
        "writes_policy_artifact": False,
        "decision": decision,
        "decision_rule": (
            "A positive screen requires candidate P(57) >= deployed baseline "
            "on every reported holdout season and positive aggregate mean gap. "
            "Any production swap still requires a separate re-baseline, leakage "
            "audit, nuclear test, reversible policy artifact, and deploy gate."
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
            "baseline": (
                "deployed policy table evaluated on the same holdout rows using "
                "the deployed saved probability boundaries"
            ),
            "candidate": (
                "candidate bins and policy fit only on prior estimated-PA profile "
                "seasons; holdout rows are classified by candidate training boundaries"
            ),
            "remaining_caveat": (
                "estimated-PA profiles may still use actual historical lineup "
                "slot and batter universe; this isolates the PA-basis and pitcher "
                "exposure question, not projected-lineup availability"
            ),
        },
        "production_policy": {
            "path": str(prod_policy_path),
            "season_length": int(prod_policy_length),
            "boundaries": [float(x) for x in prod_boundaries],
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
    result = run_evaluation(
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
            f"candidate={fold['candidate_p57']:.10f} "
            f"baseline={fold['deployed_baseline_p57']:.10f} "
            f"gap={fold['gap']:+.10f}"
        )
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
