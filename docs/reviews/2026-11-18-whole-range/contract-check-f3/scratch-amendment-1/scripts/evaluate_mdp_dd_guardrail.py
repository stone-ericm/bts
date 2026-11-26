#!/usr/bin/env python3
"""Evaluate the pre-registered MDP double-down guardrail.

This is an evidence-only row-stream evaluator. It reads ranked historical
profiles plus a saved MDP policy, computes exact first-passage streak metrics
for CURRENT and GUARDRAIL(floor), and emits a JSON artifact. It does not write
policy artifacts or claim a deploy path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from bts.simulate.mdp import ACTIONS, load_policy, lookup_action, transition_outcomes


DEFAULT_SEASONS = (2021, 2022, 2023, 2024, 2025)
DEFAULT_FLOORS = (0.40, 0.50, 0.55, 0.60)
SUPPORT_TARGETS = (10, 20, 30)
DIAGNOSTIC_TARGETS = (40, 57)
MAX_STREAK_CAP = 57
REQUIRED_PRIMARY_COLUMNS = {
    "date",
    "season",
    "rank",
    "p_game_hit",
    "actual_hit",
    "game_pk",
}
REQUIRED_PROXY_COLUMNS = {
    "date",
    "season",
    "rank",
    "p_game_hit",
    "actual_hit",
}


@dataclass(frozen=True)
class RowDay:
    date: str
    season: int
    primary_p: float
    primary_hit: int
    primary_game_pk: Any | None
    double_p: float | None
    double_hit: int | None
    double_game_pk: Any | None
    double_rank: int | None

    @property
    def p_both(self) -> float | None:
        if self.double_p is None:
            return None
        return float(self.primary_p * self.double_p)


@dataclass(frozen=True)
class ReachEvaluation:
    probability: float
    changed_state_count: int
    changed_date_count: int
    changed_state_mass: float
    double_state_count: int
    double_date_count: int
    no_double_candidate_state_count: int
    changed_qbin_counts: dict[str, int]
    changed_qbin_mass: dict[str, float]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip()


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if pd.isna(value):
        return None
    return value


def _parse_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _parse_floats(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def _classify_quality_bin(p_game_hit: float, boundaries: Sequence[float]) -> int:
    q = 0
    for boundary in boundaries:
        if p_game_hit >= boundary:
            q += 1
    return q


def profile_paths(profiles_dir: Path, seasons: Sequence[int]) -> dict[int, Path]:
    return {int(season): profiles_dir / f"backtest_{int(season)}.parquet" for season in seasons}


def load_profiles(
    profiles_dir: Path,
    seasons: Sequence[int],
    *,
    allow_rank2_proxy: bool = False,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Load per-season ranked profile parquets and report schema validity."""
    required = REQUIRED_PROXY_COLUMNS if allow_rank2_proxy else REQUIRED_PRIMARY_COLUMNS
    paths = profile_paths(profiles_dir, seasons)
    missing_paths = [str(path) for path in paths.values() if not path.exists()]
    profile_hashes = {
        str(season): _sha256(path)
        for season, path in paths.items()
        if path.exists()
    }
    if missing_paths:
        return None, {
            "valid": False,
            "allow_rank2_proxy": allow_rank2_proxy,
            "missing_paths": missing_paths,
            "profile_sha256": profile_hashes,
            "issues": ["missing profile parquet"],
        }

    frames = []
    column_sets: dict[str, list[str]] = {}
    missing_columns_by_season: dict[str, list[str]] = {}
    null_counts_by_season: dict[str, dict[str, int]] = {}
    row_counts_by_season: dict[str, int] = {}
    day_counts_by_season: dict[str, int] = {}
    duplicate_date_batter_rows_by_season: dict[str, int | None] = {}
    inferred_season_by_season: dict[str, bool] = {}

    for season, path in paths.items():
        frame = pd.read_parquet(path)
        original_columns = list(frame.columns)
        if "date" in frame.columns:
            frame = frame.copy()
            frame["date"] = pd.to_datetime(frame["date"]).dt.date
        inferred_season = False
        if allow_rank2_proxy and "season" not in frame.columns:
            frame = frame.copy()
            frame["season"] = int(season)
            inferred_season = True
        if "season" in frame.columns:
            frame["season"] = frame["season"].astype(int)
        inferred_season_by_season[str(season)] = inferred_season
        column_sets[str(season)] = original_columns
        missing = sorted(required.difference(frame.columns))
        if missing:
            missing_columns_by_season[str(season)] = missing
        row_counts_by_season[str(season)] = int(len(frame))
        day_counts_by_season[str(season)] = (
            int(frame["date"].nunique()) if "date" in frame.columns else 0
        )
        null_counts_by_season[str(season)] = {
            col: int(frame[col].isna().sum())
            for col in sorted(required.intersection(frame.columns))
        }
        if {"date", "batter_id"}.issubset(frame.columns):
            duplicate_date_batter_rows_by_season[str(season)] = int(
                frame.duplicated(["date", "batter_id"], keep=False).sum()
            )
        else:
            duplicate_date_batter_rows_by_season[str(season)] = None
        frames.append(frame)

    issues: list[str] = []
    if missing_columns_by_season:
        issues.append("missing required columns")
    null_required = {
        season: counts
        for season, counts in null_counts_by_season.items()
        if any(value > 0 for value in counts.values())
    }
    if null_required:
        issues.append("nulls in required columns")

    profiles = pd.concat(frames, ignore_index=True) if frames else None
    return profiles, {
        "valid": not issues,
        "allow_rank2_proxy": allow_rank2_proxy,
        "required_columns": sorted(required),
        "missing_paths": missing_paths,
        "missing_columns_by_season": missing_columns_by_season,
        "null_counts_by_season": null_counts_by_season,
        "row_counts_by_season": row_counts_by_season,
        "day_counts_by_season": day_counts_by_season,
        "duplicate_date_batter_rows_by_season": duplicate_date_batter_rows_by_season,
        "inferred_season_by_season": inferred_season_by_season,
        "columns_by_season": column_sets,
        "profile_sha256": profile_hashes,
        "issues": issues,
    }


def build_row_days(
    profiles: pd.DataFrame,
    season: int,
    *,
    allow_rank2_proxy: bool = False,
) -> list[RowDay]:
    """Build one ordered primary/DD row per profile date."""
    season_profiles = profiles[profiles["season"].astype(int) == int(season)].copy()
    if season_profiles.empty:
        return []
    season_profiles["date"] = pd.to_datetime(season_profiles["date"]).dt.date
    row_days: list[RowDay] = []

    for day, group in season_profiles.groupby("date", sort=True):
        ranked = group.sort_values("rank")
        primary_rows = ranked[ranked["rank"] == 1]
        if primary_rows.empty:
            continue
        primary = primary_rows.iloc[0]
        primary_game_pk = _jsonable(primary.get("game_pk")) if "game_pk" in ranked.columns else None

        candidates = ranked[ranked["rank"] > 1]
        if allow_rank2_proxy:
            eligible = candidates
        else:
            eligible = candidates[
                candidates["game_pk"].notna()
                & (candidates["game_pk"] != primary["game_pk"])
            ]

        if eligible.empty:
            double_p = None
            double_hit = None
            double_game_pk = None
            double_rank = None
        else:
            double = eligible.iloc[0]
            double_p = float(double["p_game_hit"])
            double_hit = int(double["actual_hit"])
            double_game_pk = _jsonable(double.get("game_pk")) if "game_pk" in ranked.columns else None
            double_rank = int(double["rank"])

        row_days.append(RowDay(
            date=day.isoformat(),
            season=int(season),
            primary_p=float(primary["p_game_hit"]),
            primary_hit=int(primary["actual_hit"]),
            primary_game_pk=primary_game_pk,
            double_p=double_p,
            double_hit=double_hit,
            double_game_pk=double_game_pk,
            double_rank=double_rank,
        ))
    return row_days


def _current_executable_action(raw_action: str, day: RowDay) -> str:
    if raw_action == "double" and day.double_p is None:
        return "single"
    return raw_action


def _guardrailed_action(raw_action: str, current_action: str, day: RowDay, floor: float | None) -> str:
    if (
        floor is not None
        and raw_action == "double"
        and day.p_both is not None
        and day.p_both < floor
    ):
        return "single"
    return current_action


def evaluate_reach_probability_row_stream(
    days: Sequence[RowDay],
    policy_table: np.ndarray,
    boundaries: Sequence[float],
    policy_length: int,
    *,
    target: int,
    floor: float | None = None,
    initial_streak: int = 0,
    initial_saver_available: bool = True,
) -> ReachEvaluation:
    """Exact first-passage probability for reaching target on an ordered row stream."""
    if target < 1 or target > MAX_STREAK_CAP:
        raise ValueError(f"target must be between 1 and {MAX_STREAK_CAP}")

    mass: dict[tuple[int, bool], float] = {
        (min(initial_streak, target - 1), bool(initial_saver_available)): 1.0
    }
    reached = 0.0
    changed_state_count = 0
    changed_state_mass = 0.0
    changed_dates: set[str] = set()
    double_state_count = 0
    double_dates: set[str] = set()
    no_double_candidate_state_count = 0
    changed_qbin_counts: Counter[str] = Counter()
    changed_qbin_mass: defaultdict[str, float] = defaultdict(float)

    n_days = len(days)
    for idx, day in enumerate(days):
        if not mass:
            break
        days_remaining = min(policy_length, n_days - idx)
        next_mass: defaultdict[tuple[int, bool], float] = defaultdict(float)

        for (streak, saver), state_prob in mass.items():
            raw_action = lookup_action(
                policy_table,
                list(boundaries),
                streak,
                days_remaining,
                saver,
                day.primary_p,
                policy_length,
            )
            current_action = _current_executable_action(raw_action, day)
            action = _guardrailed_action(raw_action, current_action, day, floor)

            if raw_action == "double":
                double_state_count += 1
                double_dates.add(day.date)
                if day.double_p is None:
                    no_double_candidate_state_count += 1
            if action != current_action:
                qbin = str(_classify_quality_bin(day.primary_p, boundaries))
                changed_state_count += 1
                changed_state_mass += state_prob
                changed_dates.add(day.date)
                changed_qbin_counts[qbin] += 1
                changed_qbin_mass[qbin] += state_prob

            p_both = day.p_both if day.p_both is not None else 0.0
            for branch in transition_outcomes(
                action,
                streak,
                saver,
                p_hit=day.primary_p,
                p_both=p_both,
                target=target,
            ):
                branch_mass = state_prob * branch.probability
                if branch.next_streak >= target:
                    reached += branch_mass
                else:
                    next_mass[(branch.next_streak, branch.saver_available)] += branch_mass
        mass = dict(next_mass)

    return ReachEvaluation(
        probability=float(reached),
        changed_state_count=int(changed_state_count),
        changed_date_count=len(changed_dates),
        changed_state_mass=float(changed_state_mass),
        double_state_count=int(double_state_count),
        double_date_count=len(double_dates),
        no_double_candidate_state_count=int(no_double_candidate_state_count),
        changed_qbin_counts=dict(changed_qbin_counts),
        changed_qbin_mass={k: float(v) for k, v in changed_qbin_mass.items()},
    )


def evaluate_arm(
    days: Sequence[RowDay],
    policy_table: np.ndarray,
    boundaries: Sequence[float],
    policy_length: int,
    *,
    floor: float | None = None,
) -> dict[str, Any]:
    reach_by_target: dict[str, float] = {}
    for target in range(1, MAX_STREAK_CAP + 1):
        evaluation = evaluate_reach_probability_row_stream(
            days,
            policy_table,
            boundaries,
            policy_length,
            target=target,
            floor=floor,
        )
        reach_by_target[str(target)] = evaluation.probability

    support_targets = {
        str(target): reach_by_target[str(target)]
        for target in (*SUPPORT_TARGETS, *DIAGNOSTIC_TARGETS)
    }
    support_evaluation = evaluate_reach_probability_row_stream(
        days,
        policy_table,
        boundaries,
        policy_length,
        target=MAX_STREAK_CAP,
        floor=floor,
    )
    return {
        "expected_max_streak": float(sum(reach_by_target.values())),
        "reach_probability": support_targets,
        "support_diagnostics": asdict(support_evaluation),
    }


def trigger_overlap(days_by_season: dict[int, list[RowDay]], floor: float) -> dict[str, Any]:
    by_season = {}
    total_eligible = 0
    total_triggers = 0
    p_both_values = []

    for season, days in sorted(days_by_season.items()):
        eligible = [day for day in days if day.p_both is not None]
        triggers = [day for day in eligible if day.p_both is not None and day.p_both < floor]
        total_eligible += len(eligible)
        total_triggers += len(triggers)
        p_both_values.extend(day.p_both for day in eligible if day.p_both is not None)
        by_season[str(season)] = {
            "eligible_days": len(eligible),
            "trigger_days": len(triggers),
            "trigger_fraction": (len(triggers) / len(eligible)) if eligible else None,
        }

    seasons_ge5 = sum(1 for item in by_season.values() if item["trigger_days"] >= 5)
    return {
        "floor": float(floor),
        "eligible_days": int(total_eligible),
        "trigger_days": int(total_triggers),
        "trigger_fraction": (total_triggers / total_eligible) if total_eligible else None,
        "seasons_with_ge5_triggers": int(seasons_ge5),
        "by_season": by_season,
        "p_both": {
            "n": len(p_both_values),
            "mean": mean(p_both_values) if p_both_values else None,
            "min": min(p_both_values) if p_both_values else None,
            "max": max(p_both_values) if p_both_values else None,
        },
        "sufficient_trigger_overlap": bool(total_triggers >= 30 and seasons_ge5 >= 3),
    }


def realized_replay(
    days: Sequence[RowDay],
    policy_table: np.ndarray,
    boundaries: Sequence[float],
    policy_length: int,
    *,
    floor: float | None = None,
) -> dict[str, Any]:
    streak = 0
    saver = True
    max_streak = 0
    changed_dates: set[str] = set()
    double_dates: set[str] = set()
    for idx, day in enumerate(days):
        days_remaining = min(policy_length, len(days) - idx)
        raw_action = lookup_action(
            policy_table,
            list(boundaries),
            streak,
            days_remaining,
            saver,
            day.primary_p,
            policy_length,
        )
        current_action = _current_executable_action(raw_action, day)
        action = _guardrailed_action(raw_action, current_action, day, floor)
        if raw_action == "double":
            double_dates.add(day.date)
        if action != current_action:
            changed_dates.add(day.date)

        hit = False
        increment = 0
        if action == "single":
            hit = bool(day.primary_hit)
            increment = 1
        elif action == "double":
            hit = bool(day.primary_hit and day.double_hit)
            increment = 2
        elif action == "skip":
            max_streak = max(max_streak, streak)
            continue

        if hit:
            streak = min(MAX_STREAK_CAP, streak + increment)
        elif saver and 10 <= streak <= 15:
            saver = False
        else:
            streak = 0
        max_streak = max(max_streak, streak)

    return {
        "longest_streak": int(max_streak),
        "final_streak": int(streak),
        "saver_available_final": bool(saver),
        "double_dates": len(double_dates),
        "changed_dates": len(changed_dates),
    }


def _gap_tables(current: dict[int, dict[str, Any]], guardrail: dict[int, dict[str, Any]]) -> dict[str, Any]:
    per_season = {}
    for season in sorted(current):
        c = current[season]
        g = guardrail[season]
        reach_gaps = {
            target: float(g["reach_probability"][target] - c["reach_probability"][target])
            for target in c["reach_probability"]
        }
        per_season[str(season)] = {
            "current_expected_max_streak": c["expected_max_streak"],
            "guardrail_expected_max_streak": g["expected_max_streak"],
            "expected_max_gap": float(g["expected_max_streak"] - c["expected_max_streak"]),
            "reach_gap": reach_gaps,
            "changed_decision_dates": g["support_diagnostics"]["changed_date_count"],
            "changed_state_count": g["support_diagnostics"]["changed_state_count"],
            "changed_state_mass": g["support_diagnostics"]["changed_state_mass"],
            "changed_qbin_counts": g["support_diagnostics"]["changed_qbin_counts"],
            "changed_qbin_mass": g["support_diagnostics"]["changed_qbin_mass"],
        }

    expected_gaps = [row["expected_max_gap"] for row in per_season.values()]
    reach_aggregate_gaps = {}
    for target in map(str, (*SUPPORT_TARGETS, *DIAGNOSTIC_TARGETS)):
        reach_aggregate_gaps[target] = mean(
            row["reach_gap"][target] for row in per_season.values()
        )
    return {
        "per_season": per_season,
        "aggregate": {
            "mean_expected_max_gap": mean(expected_gaps) if expected_gaps else None,
            "negative_expected_max_seasons": sum(1 for gap in expected_gaps if gap < 0),
            "catastrophic_regression_seasons": [
                season
                for season, row in per_season.items()
                if row["guardrail_expected_max_streak"] <= row["current_expected_max_streak"] - 0.25
            ],
            "reach_aggregate_gaps": reach_aggregate_gaps,
            "total_changed_decision_dates": sum(
                row["changed_decision_dates"] for row in per_season.values()
            ),
            "changed_decision_seasons": sum(
                1 for row in per_season.values() if row["changed_decision_dates"] > 0
            ),
        },
    }


def _active_support_summary(
    current: dict[int, dict[str, Any]],
    gaps: dict[str, Any],
) -> dict[str, Any]:
    out = {}
    for target in map(str, SUPPORT_TARGETS):
        active_seasons = [
            str(season)
            for season, metrics in current.items()
            if metrics["reach_probability"][target] >= 1e-3
        ]
        active_gaps = [
            gaps["per_season"][season]["reach_gap"][target]
            for season in active_seasons
        ]
        active_rel_gaps = []
        for season in active_seasons:
            current_value = current[int(season)]["reach_probability"][target]
            gap = gaps["per_season"][season]["reach_gap"][target]
            active_rel_gaps.append(gap / current_value if current_value > 0 else None)
        out[target] = {
            "active_seasons": active_seasons,
            "aggregate_gap": mean(active_gaps) if active_gaps else None,
            "negative_active_seasons": sum(1 for gap in active_gaps if gap < 0),
            "min_relative_gap": min(
                value for value in active_rel_gaps if value is not None
            ) if any(value is not None for value in active_rel_gaps) else None,
        }
    return out


def label_floor(
    *,
    schema_valid: bool,
    overlap: dict[str, Any],
    current: dict[int, dict[str, Any]],
    gaps: dict[str, Any],
) -> tuple[str, list[str], dict[str, Any]]:
    if not schema_valid:
        return "INVALID_PRIMARY_SURFACE", ["primary surface failed schema validation"], {}

    aggregate = gaps["aggregate"]
    active_support = _active_support_summary(current, gaps)
    total_changed = aggregate["total_changed_decision_dates"]
    changed_seasons = aggregate["changed_decision_seasons"]
    catastrophic = bool(aggregate["catastrophic_regression_seasons"])
    expected_gap = aggregate["mean_expected_max_gap"]
    negative_expected_seasons = aggregate["negative_expected_max_seasons"]
    support_aggregate_negative = any(
        item["aggregate_gap"] is not None and item["aggregate_gap"] < 0
        for item in active_support.values()
    )
    support_negative_gt_one = any(
        item["negative_active_seasons"] > 1
        for item in active_support.values()
    )
    support_no_harm = all(
        (
            item["aggregate_gap"] is None
            or (
                item["aggregate_gap"] >= -0.005
                and (
                    item["min_relative_gap"] is None
                    or item["min_relative_gap"] >= -0.20
                )
            )
        )
        for item in active_support.values()
    )
    details = {"active_support": active_support}

    if not overlap["sufficient_trigger_overlap"]:
        if (
            total_changed >= 10
            and changed_seasons >= 2
            and not catastrophic
            and expected_gap is not None
            and expected_gap >= -0.10
            and negative_expected_seasons <= 1
            and support_no_harm
        ):
            return "NO_HARM_SCREEN", ["underpowered trigger overlap but no-harm criteria passed"], details
        return "UNDERPOWERED_TRIGGER_OVERLAP", ["floor lacks pre-registered trigger support"], details

    if total_changed < 10 or changed_seasons < 2:
        return "NO_EFFECT", ["guardrail changes too few decisions on valid primary surface"], details

    if catastrophic:
        return "REJECT", ["catastrophic expected-max regression in at least one season"], details
    if expected_gap is not None and expected_gap < 0:
        return "REJECT", ["aggregate expected-max gap is negative"], details

    if (
        expected_gap is not None
        and expected_gap > 0
        and negative_expected_seasons <= 1
        and not support_aggregate_negative
        and not support_negative_gt_one
    ):
        near_boundary = total_changed < 15 or changed_seasons == 2
        if near_boundary:
            return "MIXED", ["positive floor is near the no-effect support boundary"], details
        return "POSITIVE_SCREEN", ["pre-registered positive benefit screen passed"], details

    if expected_gap is not None and expected_gap > 0:
        return "MIXED", ["headline metric positive but support-ladder or season-spread screen failed"], details
    return "MIXED", ["no pre-registered positive, no-harm, no-effect, or reject condition cleanly applied"], details


def _distribution(values: Iterable[float | None]) -> dict[str, Any]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {
        "n": len(clean),
        "mean": mean(clean),
        "min": min(clean),
        "max": max(clean),
        "q10": float(np.quantile(clean, 0.10)),
        "q50": float(np.quantile(clean, 0.50)),
        "q90": float(np.quantile(clean, 0.90)),
    }


def _load_production_p_both_summary(summary_path: Path | None) -> dict[str, Any] | None:
    if summary_path is None:
        return None
    return json.loads(summary_path.read_text())


def _load_production_p_both(picks_dir: Path | None) -> dict[str, Any] | None:
    if picks_dir is None or not picks_dir.exists():
        return None
    values = []
    for path in sorted(picks_dir.glob("*.json")):
        if "." in path.stem:
            continue
        try:
            body = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        pick = body.get("pick") or {}
        dd = body.get("double_down") or {}
        if pick.get("p_game_hit") is not None and dd.get("p_game_hit") is not None:
            values.append(float(pick["p_game_hit"]) * float(dd["p_game_hit"]))
    return _distribution(values)


def run_evaluation(
    *,
    profiles_dir: Path,
    policy_path: Path,
    seasons: Sequence[int] = DEFAULT_SEASONS,
    floors: Sequence[float] = DEFAULT_FLOORS,
    allow_rank2_proxy: bool = False,
    production_picks_dir: Path | None = None,
    production_p_both_summary_path: Path | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    policy_table, boundaries, policy_length = load_policy(policy_path)
    profiles, schema = load_profiles(
        profiles_dir,
        seasons,
        allow_rank2_proxy=allow_rank2_proxy,
    )

    policy_sha = _sha256(policy_path) if policy_path.exists() else None
    base = {
        "production_deploy_claim": False,
        "writes_policy_artifact": False,
        "methodology": {
            "pre_registration": "docs/sota_audit/2026-05-29-mdp-dd-guardrail-prereg.md",
            "profiles_dir": str(profiles_dir),
            "policy_path": str(policy_path),
            "seasons": [int(season) for season in seasons],
            "floors": [float(floor) for floor in floors],
            "allow_rank2_proxy": bool(allow_rank2_proxy),
            "evaluator": "exact first-passage row-stream DP",
            "transition_kernel": "bts.simulate.mdp.transition_outcomes",
            "p_both": "primary_p * double_down_p",
            "command": command,
            "git_sha": _git_sha(),
        },
        "policy": {
            "sha256": policy_sha,
            "boundaries": [float(x) for x in boundaries],
            "season_length": int(policy_length),
        },
        "profile_schema": schema,
    }

    production_p_both_distribution = (
        _load_production_p_both_summary(production_p_both_summary_path)
        or _load_production_p_both(production_picks_dir)
    )

    if profiles is None or not schema["valid"]:
        invalid_floor_results = {
            f"{float(floor):.2f}": {
                "label": "INVALID_PRIMARY_SURFACE",
                "reasons": ["primary surface failed schema validation"],
            }
            for floor in floors
        }
        return {
            **base,
            "backtest_p_both_distribution": None,
            "production_p_both_distribution": production_p_both_distribution,
            "trigger_overlap": {},
            "current": {},
            "floors": invalid_floor_results,
            "selection": {"selected_floor": None, "selected_label": None},
        }

    days_by_season = {
        int(season): build_row_days(
            profiles,
            int(season),
            allow_rank2_proxy=allow_rank2_proxy,
        )
        for season in seasons
    }
    all_p_both = [
        day.p_both
        for days in days_by_season.values()
        for day in days
        if day.p_both is not None
    ]

    current = {
        season: evaluate_arm(days, policy_table, boundaries, policy_length, floor=None)
        for season, days in days_by_season.items()
    }
    current_realized = {
        str(season): realized_replay(days, policy_table, boundaries, policy_length, floor=None)
        for season, days in days_by_season.items()
    }

    overlap_by_floor: dict[str, Any] = {}
    floor_results: dict[str, Any] = {}
    for floor in floors:
        floor_key = f"{float(floor):.2f}"
        overlap = trigger_overlap(days_by_season, float(floor))
        overlap_by_floor[floor_key] = overlap
        guardrail = {
            season: evaluate_arm(days, policy_table, boundaries, policy_length, floor=float(floor))
            for season, days in days_by_season.items()
        }
        gaps = _gap_tables(current, guardrail)
        label, reasons, decision_details = label_floor(
            schema_valid=schema["valid"] and not allow_rank2_proxy,
            overlap=overlap,
            current=current,
            gaps=gaps,
        )
        floor_results[floor_key] = {
            "label": label,
            "reasons": reasons,
            "trigger_overlap": overlap,
            "gaps": gaps,
            "decision_details": decision_details,
            "realized_replay": {
                str(season): realized_replay(
                    days_by_season[season],
                    policy_table,
                    boundaries,
                    policy_length,
                    floor=float(floor),
                )
                for season in sorted(days_by_season)
            },
        }

    passing_positive = [
        (float(floor_key), item["label"])
        for floor_key, item in floor_results.items()
        if item["label"] == "POSITIVE_SCREEN"
    ]
    passing_no_harm = [
        (float(floor_key), item["label"])
        for floor_key, item in floor_results.items()
        if item["label"] == "NO_HARM_SCREEN"
    ]
    if passing_positive:
        selected_floor, selected_label = min(passing_positive)
    elif passing_no_harm:
        selected_floor, selected_label = min(passing_no_harm)
    else:
        selected_floor, selected_label = None, None

    return {
        **base,
        "row_stream": {
            "days_by_season": {
                str(season): len(days) for season, days in sorted(days_by_season.items())
            },
            "double_eligible_days_by_season": {
                str(season): sum(1 for day in days if day.p_both is not None)
                for season, days in sorted(days_by_season.items())
            },
        },
        "backtest_p_both_distribution": _distribution(all_p_both),
        "production_p_both_distribution": production_p_both_distribution,
        "trigger_overlap": overlap_by_floor,
        "current": {
            str(season): {
                "expected_max_streak": metrics["expected_max_streak"],
                "reach_probability": metrics["reach_probability"],
                "support_diagnostics": metrics["support_diagnostics"],
                "realized_replay": current_realized[str(season)],
            }
            for season, metrics in sorted(current.items())
        },
        "floors": floor_results,
        "selection": {
            "selected_floor": selected_floor,
            "selected_label": selected_label,
            "auto_enable_authorized": False,
            "note": (
                "A positive or no-harm screen permits only a later default-off "
                "implementation PR for explicit review; it does not authorize deploy."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles-dir", type=Path, default=Path("data/simulation"))
    parser.add_argument("--policy-path", type=Path, default=Path("data/models/mdp_policy.npz"))
    parser.add_argument("--seasons", type=_parse_ints, default=list(DEFAULT_SEASONS))
    parser.add_argument("--floors", type=_parse_floats, default=list(DEFAULT_FLOORS))
    parser.add_argument("--allow-rank2-proxy", action="store_true")
    parser.add_argument("--production-picks-dir", type=Path)
    parser.add_argument("--production-p-both-summary", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    result = run_evaluation(
        profiles_dir=args.profiles_dir,
        policy_path=args.policy_path,
        seasons=args.seasons,
        floors=args.floors,
        allow_rank2_proxy=args.allow_rank2_proxy,
        production_picks_dir=args.production_picks_dir,
        production_p_both_summary_path=args.production_p_both_summary,
        command=" ".join(sys.argv),
    )
    text = json.dumps(result, indent=2 if args.pretty else None, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
