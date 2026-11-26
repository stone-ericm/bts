#!/usr/bin/env python3
"""Gate B production-metric re-baseline.

Evidence-only harness for the pre-registered boundary-only Gate B follow-up.
It compares:

* CURRENT: deployed action table + deployed boundaries.
* CANDIDATE: deployed action table + estimated-PA boundaries.

It never writes or swaps a production policy artifact.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bts.health.mdp_policy_alignment import DEFAULT_THRESHOLDS
from bts.picks import active_streak_results
from bts.simulate.mdp import ACTIONS, load_policy, lookup_action
from bts.simulate.quality_bins import QualityBins, compute_bins, compute_bins_with_boundaries
from bts.strategy import SEASON_END_DATE
from scripts.gate_b_walk_forward_policy_eval import (
    DEFAULT_N_BINS,
    DEFAULT_PROD_POLICY_PATH,
    DEFAULT_PROFILES_DIR,
    DEFAULT_SEASON_LENGTH,
    DEFAULT_SEASONS,
    _bins_summary,
    _summarize_gaps,
    load_profiles,
    parse_seasons,
)


DEFAULT_PICKS_DIR = Path("data/picks")
DEFAULT_OUTPUT = Path("data/validation/gate_b_production_metric_rebaseline_2026-05-24.json")
DEFAULT_LADDER_TARGETS = (10, 20, 30, 40)
MAX_STREAK_TARGET = 57
FLOOR_GUARD = 1e-3
SCALE_MEAN_OR_MEDIAN_DELTA_WARN = 0.03
SCALE_ANCHOR_QUANTILE_DELTA_WARN = 0.05


def fit_candidate_bins(
    profiles: pd.DataFrame,
    train_seasons: Sequence[int],
    *,
    n_bins: int,
) -> QualityBins:
    """Fit the candidate estimated-PA bins with one shared code path."""
    train_profiles = profiles[profiles["season"].isin(train_seasons)].copy()
    if train_profiles.empty:
        raise ValueError(f"no profiles for candidate boundary seasons: {list(train_seasons)}")
    return compute_bins(train_profiles, n_bins=n_bins)


def _classify(p_game_hit: float, boundaries: Sequence[float]) -> int:
    q = 0
    for boundary in boundaries:
        if p_game_hit >= boundary:
            q += 1
    return q


def _days_remaining(date_iso: str, *, season_end: str = SEASON_END_DATE) -> int:
    end = datetime.strptime(season_end, "%Y-%m-%d")
    current = datetime.strptime(date_iso, "%Y-%m-%d")
    return max(0, (end - current).days)


def _bin_metrics(values: Sequence[float], boundaries: Sequence[float]) -> dict[str, Any]:
    n_bins = len(boundaries) + 1
    counts = {str(i): 0 for i in range(n_bins)}
    for value in values:
        counts[str(_classify(float(value), boundaries))] += 1
    n = len(values)
    if n == 0:
        return {
            "n": 0,
            "counts": counts,
            "dominant_bin": None,
            "dominant_count": 0,
            "dominant_fraction": None,
            "p_min": None,
            "p_max": None,
        }
    dominant_bin, dominant_count = max(counts.items(), key=lambda item: item[1])
    return {
        "n": int(n),
        "counts": counts,
        "dominant_bin": int(dominant_bin),
        "dominant_count": int(dominant_count),
        "dominant_fraction": float(dominant_count / n),
        "p_min": float(min(values)),
        "p_max": float(max(values)),
    }


def _dominance_alerts(metrics: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    frac = metrics.get("dominant_fraction")
    return (
        metrics["n"] >= int(thresholds["min_recent_days"])
        and frac is not None
        and float(frac) >= float(thresholds["dominant_warn_frac"])
    )


def _distribution_summary(values: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"n": 0}
    quantiles = {
        f"q{int(q * 100):02d}": float(np.quantile(arr, q))
        for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    }
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        **quantiles,
        "max": float(arr.max()),
    }


def _scale_parity(production_values: Sequence[float], historical_values: Sequence[float]) -> dict[str, Any]:
    production = _distribution_summary(production_values)
    historical = _distribution_summary(historical_values)
    if production["n"] == 0 or historical["n"] == 0:
        return {
            "available": False,
            "production": production,
            "historical_estimated_pa_rank1": historical,
            "material_divergence": True,
            "reason": "empty production or historical distribution",
        }

    mean_delta = float(production["mean"] - historical["mean"])
    median_delta = float(production["q50"] - historical["q50"])
    anchor_deltas = {
        key: float(production[key] - historical[key])
        for key in ("q10", "q50", "q90")
    }
    max_anchor_abs_delta = max(abs(v) for v in anchor_deltas.values())
    material = (
        abs(mean_delta) >= SCALE_MEAN_OR_MEDIAN_DELTA_WARN
        or abs(median_delta) >= SCALE_MEAN_OR_MEDIAN_DELTA_WARN
        or max_anchor_abs_delta >= SCALE_ANCHOR_QUANTILE_DELTA_WARN
    )
    return {
        "available": True,
        "production": production,
        "historical_estimated_pa_rank1": historical,
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "anchor_quantile_deltas": anchor_deltas,
        "max_anchor_abs_delta": float(max_anchor_abs_delta),
        "thresholds": {
            "mean_or_median_abs_delta_warn": SCALE_MEAN_OR_MEDIAN_DELTA_WARN,
            "anchor_quantile_abs_delta_warn": SCALE_ANCHOR_QUANTILE_DELTA_WARN,
        },
        "material_divergence": bool(material),
        "reason": (
            "production and historical estimated-PA scales diverge materially"
            if material
            else "no material scale divergence by pre-set thresholds"
        ),
    }


def _transition_streak(
    streak: int,
    saver_available: bool,
    active_results: list[bool],
) -> tuple[int, bool]:
    if not active_results:
        return streak, saver_available
    if all(active_results):
        return streak + len(active_results), saver_available
    if saver_available and 10 <= streak <= 15:
        return streak, False
    return 0, saver_available


def _legacy_active_results(body: dict[str, Any]) -> list[bool] | None:
    result = body.get("result")
    if result in (None, "unresolved", "suspended"):
        return None
    if result == "void":
        return []
    if result == "hit":
        return [True] * (2 if body.get("double_down") else 1)
    if result == "miss":
        return [False]
    return None


def _load_production_rows(
    picks_dir: Path,
    *,
    today: date,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not picks_dir.exists():
        raise FileNotFoundError(f"picks_dir does not exist: {picks_dir}")

    streak = 0
    saver_available = True
    state_known = True
    for path in sorted(picks_dir.glob("*.json")):
        if "." in path.stem:
            continue
        try:
            pick_date = date.fromisoformat(path.stem)
        except ValueError:
            continue
        if pick_date.year != today.year or pick_date > today:
            continue
        body = json.loads(path.read_text())
        pick = body.get("pick") or {}
        p_game_hit = pick.get("p_game_hit")
        if p_game_hit is None:
            continue

        dd = body.get("double_down") or None
        row = {
            "date": pick_date.isoformat(),
            "p_game_hit": float(p_game_hit),
            "double_down_p_game_hit": (
                None if not dd or dd.get("p_game_hit") is None else float(dd["p_game_hit"])
            ),
            "observed_action": "double" if dd else "single",
            "pre_streak": int(streak) if state_known else None,
            "pre_saver_available": bool(saver_available) if state_known else None,
            "state_known": bool(state_known),
            "result": body.get("result"),
            "slot_results": body.get("slot_results"),
        }
        rows.append(row)

        if not state_known:
            continue
        if body.get("slot_results") is not None:
            active = active_streak_results(body["slot_results"])
        else:
            active = _legacy_active_results(body)
        if active is None:
            state_known = False
            continue
        streak, saver_available = _transition_streak(streak, saver_available, active)

    return rows


def _mechanism_check(
    *,
    picks_dir: Path,
    profiles: pd.DataFrame,
    deployed_table: np.ndarray,
    deployed_boundaries: Sequence[float],
    deployed_policy_length: int,
    candidate_boundaries: Sequence[float],
    today: date,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    rows = _load_production_rows(picks_dir, today=today)
    historical_rank1 = profiles[profiles["rank"] == 1]["p_game_hit"].astype(float).tolist()
    production_primary = [row["p_game_hit"] for row in rows]
    scale_parity = _scale_parity(production_primary, historical_rank1)

    for row in rows:
        row["current_bin"] = _classify(row["p_game_hit"], deployed_boundaries)
        row["candidate_bin"] = _classify(row["p_game_hit"], candidate_boundaries)
        if row["state_known"]:
            streak = int(row["pre_streak"])
            saver = bool(row["pre_saver_available"])
            days_remaining = _days_remaining(row["date"])
            row["current_action"] = lookup_action(
                deployed_table,
                list(deployed_boundaries),
                streak,
                days_remaining,
                saver,
                row["p_game_hit"],
                deployed_policy_length,
            )
            row["candidate_action"] = lookup_action(
                deployed_table,
                list(candidate_boundaries),
                streak,
                days_remaining,
                saver,
                row["p_game_hit"],
                deployed_policy_length,
            )
            row["decision_changed"] = row["current_action"] != row["candidate_action"]
            row["decision_change"] = (
                f"{row['current_action']}->{row['candidate_action']}"
                if row["decision_changed"]
                else None
            )
        else:
            row["current_action"] = None
            row["candidate_action"] = None
            row["decision_changed"] = None
            row["decision_change"] = None

    cutoff = today.toordinal() - int(t["lookback_days"])
    recent_pool = [
        row for row in rows
        if date.fromisoformat(row["date"]).toordinal() >= cutoff
    ]
    recent = recent_pool[-int(t["recent_days"]):]
    recent_primary = [row["p_game_hit"] for row in recent]
    recent_dd = [
        row["double_down_p_game_hit"] for row in recent
        if row["double_down_p_game_hit"] is not None
    ]
    current_primary_metrics = _bin_metrics(recent_primary, deployed_boundaries)
    candidate_primary_metrics = _bin_metrics(recent_primary, candidate_boundaries)
    current_dd_metrics = _bin_metrics(recent_dd, deployed_boundaries)
    candidate_dd_metrics = _bin_metrics(recent_dd, candidate_boundaries)

    comparable_rows = [row for row in rows if row["state_known"]]
    changed_rows = [row for row in comparable_rows if row["decision_changed"]]
    change_counts: dict[str, int] = {}
    for row in changed_rows:
        key = str(row["decision_change"])
        change_counts[key] = change_counts.get(key, 0) + 1

    current_alert = _dominance_alerts(current_primary_metrics, t)
    candidate_alert = _dominance_alerts(candidate_primary_metrics, t)
    if scale_parity["material_divergence"]:
        decision = "MECHANISM_DOWNGRADED_SCALE_PARITY_DIVERGENCE"
    elif candidate_alert:
        decision = "MECHANISM_FAILS_CANDIDATE_BIN_DOMINANCE"
    elif candidate_primary_metrics["n"] < int(t["min_recent_days"]):
        decision = "MECHANISM_INSUFFICIENT_RECENT_SUPPORT"
    elif not changed_rows:
        decision = "MECHANISM_NO_DECISION_CHANGE"
    else:
        decision = "MECHANISM_PASSES_NOT_SWAP_JUSTIFYING"

    return {
        "decision": decision,
        "production_pick_count": len(rows),
        "state_known_count": len(comparable_rows),
        "thresholds": t,
        "candidate_training_window": "all_available_historical_estimated_pa_profile_seasons",
        "scale_parity": scale_parity,
        "recent_window": {
            "lookback_days": int(t["lookback_days"]),
            "recent_days": int(t["recent_days"]),
            "n_recent_primary": int(candidate_primary_metrics["n"]),
            "n_recent_double_down": int(candidate_dd_metrics["n"]),
        },
        "bin_occupancy": {
            "current_primary": current_primary_metrics,
            "candidate_primary": candidate_primary_metrics,
            "current_double_down": current_dd_metrics,
            "candidate_double_down": candidate_dd_metrics,
            "current_primary_alerts": bool(current_alert),
            "candidate_primary_alerts": bool(candidate_alert),
        },
        "actions": {
            "changed_decision_count": len(changed_rows),
            "change_counts": change_counts,
            "examples": changed_rows[:10],
        },
    }


def evaluate_policy_reach_probability(
    policy_table: np.ndarray,
    bins: QualityBins,
    *,
    target: int,
    season_length: int,
) -> float:
    """Exact P(reaching at least target streak) for a fixed policy table."""
    if target < 1 or target > MAX_STREAK_TARGET:
        raise ValueError(f"target must be between 1 and {MAX_STREAK_TARGET}, got {target}")
    if policy_table.shape[3] != len(bins.bins):
        raise ValueError(
            f"policy has {policy_table.shape[3]} bins but bins has {len(bins.bins)}"
        )

    n_days = season_length + 1
    n_bins = len(bins.bins)
    freq = np.array([b.frequency for b in bins.bins])
    p_hit = np.array([b.p_hit for b in bins.bins])
    p_both = np.array([b.p_both for b in bins.bins])

    V = np.zeros((target + 1, n_days, 2, n_bins))
    V[target, :, :, :] = 1.0

    for d in range(1, n_days):
        d_policy = min(d, policy_table.shape[1] - 1)

        def ev(next_s: int, next_saver: int) -> float:
            return float(np.dot(freq, V[next_s, d - 1, next_saver, :]))

        for s in range(target):
            for saver in range(2):
                ev_stay = ev(s, saver)
                ev_reset = ev(0, saver)
                ev_hold_saver_off = ev(s, 0)
                saver_active = bool(saver) and 10 <= s <= 15
                for q in range(n_bins):
                    action = int(policy_table[min(s, policy_table.shape[0] - 1), d_policy, saver, q])
                    if action == ACTIONS.index("skip"):
                        V[s, d, saver, q] = ev_stay
                    elif action == ACTIONS.index("single"):
                        next_hit = min(s + 1, target)
                        if saver_active:
                            V[s, d, saver, q] = (
                                p_hit[q] * ev(next_hit, saver)
                                + (1 - p_hit[q]) * ev_hold_saver_off
                            )
                        else:
                            V[s, d, saver, q] = (
                                p_hit[q] * ev(next_hit, saver)
                                + (1 - p_hit[q]) * ev_reset
                            )
                    elif action == ACTIONS.index("double"):
                        next_double = min(s + 2, target)
                        if saver_active:
                            V[s, d, saver, q] = (
                                p_both[q] * ev(next_double, saver)
                                + (1 - p_both[q]) * ev_hold_saver_off
                            )
                        else:
                            V[s, d, saver, q] = (
                                p_both[q] * ev(next_double, saver)
                                + (1 - p_both[q]) * ev_reset
                            )
                    else:
                        raise ValueError(f"invalid action index in policy table: {action}")

    return float(np.dot(freq, V[0, season_length, 1, :]))


def exact_policy_metrics(
    policy_table: np.ndarray,
    bins: QualityBins,
    *,
    season_length: int,
    ladder_targets: Sequence[int] = DEFAULT_LADDER_TARGETS,
) -> dict[str, Any]:
    reach_all = {
        str(target): evaluate_policy_reach_probability(
            policy_table,
            bins,
            target=target,
            season_length=season_length,
        )
        for target in range(1, MAX_STREAK_TARGET + 1)
    }
    return {
        "expected_max_streak": float(sum(reach_all.values())),
        "reach_probabilities": {
            str(target): float(reach_all[str(target)])
            for target in ladder_targets
        },
        "p57_diagnostic": float(reach_all[str(MAX_STREAK_TARGET)]),
    }


def _metric_gap(candidate: dict[str, Any], current: dict[str, Any], metric: str) -> float:
    if metric == "expected_max_streak":
        return float(candidate[metric] - current[metric])
    return float(candidate["reach_probabilities"][metric] - current["reach_probabilities"][metric])


def _summarize_metric_rows(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    gaps = np.asarray([_metric_gap(row["candidate_metrics"], row["current_metrics"], metric) for row in rows])
    current_values = np.asarray([
        row["current_metrics"]["expected_max_streak"]
        if metric == "expected_max_streak"
        else row["current_metrics"]["reach_probabilities"][metric]
        for row in rows
    ])
    summary = _summarize_gaps([{"gap": float(gap)} for gap in gaps])
    summary["current_mean"] = float(current_values.mean())
    summary["current_min"] = float(current_values.min())
    summary["current_max"] = float(current_values.max())
    summary["active_after_floor_guard"] = (
        True if metric == "expected_max_streak" else bool(summary["current_min"] >= FLOOR_GUARD)
    )
    return summary


def _outcome_decision(metric_summaries: dict[str, dict[str, Any]]) -> str:
    headline = metric_summaries["expected_max_streak"]
    active_ladder = {
        key: value for key, value in metric_summaries.items()
        if key != "expected_max_streak" and value["active_after_floor_guard"]
    }

    if headline["mean_gap"] < 0:
        return "OUTCOME_REJECT_NEGATIVE_HEADLINE_GAP"
    if any(summary["mean_gap"] < 0 for summary in active_ladder.values()):
        return "OUTCOME_REJECT_NEGATIVE_ACTIVE_LADDER_GAP"

    headline_stable = (
        headline["std_gap"] == 0.0 and headline["mean_gap"] > 0
    ) or headline["mean_gap"] > headline["std_gap"]
    if headline["mean_gap"] <= 0 or not headline_stable:
        return "OUTCOME_MIXED_HEADLINE_FAILS_STABILITY_BAR"
    if headline["n_negative"] > 1:
        return "OUTCOME_MIXED_HEADLINE_HAS_MULTIPLE_NEGATIVE_FOLDS"
    if any(summary["mean_gap"] <= 0 for summary in active_ladder.values()):
        return "OUTCOME_MIXED_ACTIVE_LADDER_NOT_ALL_POSITIVE"
    if any(summary["n_negative"] > 1 for summary in active_ladder.values()):
        return "OUTCOME_MIXED_ACTIVE_LADDER_HAS_MULTIPLE_NEGATIVE_FOLDS"
    return "OUTCOME_POSITIVE_REQUIRES_FULL_GATE"


def _outcome_quality_eval(
    *,
    profiles: pd.DataFrame,
    deployed_table: np.ndarray,
    deployed_boundaries: Sequence[float],
    seasons: Sequence[int],
    n_bins: int,
    season_length: int,
    ladder_targets: Sequence[int],
) -> dict[str, Any]:
    rows = []
    for holdout_season in seasons[1:]:
        train_seasons = [season for season in seasons if season < holdout_season]
        holdout_profiles = profiles[profiles["season"] == holdout_season].copy()
        if holdout_profiles.empty:
            raise ValueError(f"empty holdout fold for {holdout_season}")

        candidate_train_bins = fit_candidate_bins(
            profiles,
            train_seasons,
            n_bins=n_bins,
        )
        candidate_holdout_bins = compute_bins_with_boundaries(
            holdout_profiles,
            candidate_train_bins.boundaries,
        )
        current_holdout_bins = compute_bins_with_boundaries(
            holdout_profiles,
            list(deployed_boundaries),
        )
        current_metrics = exact_policy_metrics(
            deployed_table,
            current_holdout_bins,
            season_length=season_length,
            ladder_targets=ladder_targets,
        )
        candidate_metrics = exact_policy_metrics(
            deployed_table,
            candidate_holdout_bins,
            season_length=season_length,
            ladder_targets=ladder_targets,
        )

        n_holdout_rank1 = int((holdout_profiles["rank"] == 1).sum())
        rows.append({
            "holdout_season": int(holdout_season),
            "train_seasons": [int(season) for season in train_seasons],
            "candidate_training_window": "prior_estimated_pa_profile_seasons_only",
            "n_holdout_rank1": n_holdout_rank1,
            "current_metrics": current_metrics,
            "candidate_metrics": candidate_metrics,
            "gaps": {
                "expected_max_streak": _metric_gap(
                    candidate_metrics,
                    current_metrics,
                    "expected_max_streak",
                ),
                **{
                    f"p_reach_{target}": _metric_gap(
                        candidate_metrics,
                        current_metrics,
                        str(target),
                    )
                    for target in ladder_targets
                },
                "p57_diagnostic": float(
                    candidate_metrics["p57_diagnostic"] - current_metrics["p57_diagnostic"]
                ),
            },
            "current_holdout_bins": _bins_summary(current_holdout_bins, n_holdout_rank1),
            "candidate_holdout_bins": _bins_summary(candidate_holdout_bins, n_holdout_rank1),
        })

    metric_summaries = {
        "expected_max_streak": _summarize_metric_rows(rows, "expected_max_streak"),
        **{
            f"p_reach_{target}": _summarize_metric_rows(rows, str(target))
            for target in ladder_targets
        },
    }
    decision = _outcome_decision(metric_summaries)
    return {
        "decision": decision,
        "decision_rule": (
            "Positive requires headline E[max streak] mean gap > 0 and above "
            "fold-to-fold std, no active outcome metric with more than one "
            "negative fold, and every active non-floor ladder rung positive. "
            "Support rungs with current mean < 1e-3 are demoted to diagnostic."
        ),
        "floor_guard": FLOOR_GUARD,
        "ladder_targets": [int(target) for target in ladder_targets],
        "metric_summaries": metric_summaries,
        "folds": rows,
    }


def run_measurement(
    *,
    profiles_dir: Path,
    picks_dir: Path,
    prod_policy_path: Path,
    seasons: Sequence[int] = DEFAULT_SEASONS,
    n_bins: int = DEFAULT_N_BINS,
    season_length: int = DEFAULT_SEASON_LENGTH,
    ladder_targets: Sequence[int] = DEFAULT_LADDER_TARGETS,
    today: date | None = None,
) -> dict[str, Any]:
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if today is None:
        today = date.today()

    profiles = load_profiles(profiles_dir, seasons, require_estimated_basis=True)
    deployed_table, deployed_boundaries, deployed_policy_length = load_policy(prod_policy_path)
    if deployed_table.shape[3] != n_bins:
        raise ValueError(
            f"deployed policy has {deployed_table.shape[3]} bins; requested n_bins={n_bins}"
        )

    mechanism_candidate_bins = fit_candidate_bins(
        profiles,
        seasons,
        n_bins=n_bins,
    )
    mechanism = _mechanism_check(
        picks_dir=picks_dir,
        profiles=profiles,
        deployed_table=deployed_table,
        deployed_boundaries=deployed_boundaries,
        deployed_policy_length=deployed_policy_length,
        candidate_boundaries=mechanism_candidate_bins.boundaries,
        today=today,
    )
    outcome = _outcome_quality_eval(
        profiles=profiles,
        deployed_table=deployed_table,
        deployed_boundaries=deployed_boundaries,
        seasons=seasons,
        n_bins=n_bins,
        season_length=season_length,
        ladder_targets=ladder_targets,
    )

    return {
        "schema_version": "gate_b_production_metric_rebaseline_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date": today.isoformat(),
        "artifact_role": "gate_b_production_metric_rebaseline",
        "production_deploy_claim": False,
        "writes_policy_artifact": False,
        "inputs": {
            "profiles_dir": str(profiles_dir),
            "picks_dir": str(picks_dir),
            "prod_policy_path": str(prod_policy_path),
            "seasons": [int(season) for season in seasons],
            "n_bins": int(n_bins),
            "season_length": int(season_length),
            "deployed_policy_season_length": int(deployed_policy_length),
            "ladder_targets": [int(target) for target in ladder_targets],
        },
        "methodology": {
            "comparator": "boundary_only_deployed_action_table_fixed",
            "current": "deployed action table plus deployed saved boundaries",
            "candidate": "deployed action table plus estimated-PA boundaries",
            "mechanism_candidate_training_window": (
                "all available historical estimated-PA profile seasons"
            ),
            "outcome_candidate_training_window": (
                "expanding-origin prior estimated-PA profile seasons only"
            ),
            "ladder_metric_type": "exact_dynamic_programming_no_monte_carlo",
            "selection_caveat": (
                "Decision layer only: both arms evaluate the fixed selected-pick "
                "stream or historical rank-1/rank-2 profile surface."
            ),
        },
        "mechanism": mechanism,
        "outcome_quality": outcome,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES_DIR)
    parser.add_argument("--picks-dir", type=Path, default=DEFAULT_PICKS_DIR)
    parser.add_argument("--prod-policy-path", type=Path, default=DEFAULT_PROD_POLICY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seasons", default=",".join(str(s) for s in DEFAULT_SEASONS))
    parser.add_argument("--n-bins", type=int, default=DEFAULT_N_BINS)
    parser.add_argument("--season-length", type=int, default=DEFAULT_SEASON_LENGTH)
    parser.add_argument(
        "--ladder-targets",
        default=",".join(str(t) for t in DEFAULT_LADDER_TARGETS),
        help="Comma-separated reach-probability targets for the support ladder.",
    )
    parser.add_argument("--date", dest="today", default=None)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    today = date.fromisoformat(args.today) if args.today else date.today()
    result = run_measurement(
        profiles_dir=args.profiles_dir,
        picks_dir=args.picks_dir,
        prod_policy_path=args.prod_policy_path,
        seasons=parse_seasons(args.seasons),
        n_bins=args.n_bins,
        season_length=args.season_length,
        ladder_targets=parse_seasons(args.ladder_targets),
        today=today,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
    print(json.dumps({
        "output": str(args.output),
        "mechanism_decision": result["mechanism"]["decision"],
        "outcome_decision": result["outcome_quality"]["decision"],
        "production_deploy_claim": result["production_deploy_claim"],
        "writes_policy_artifact": result["writes_policy_artifact"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
