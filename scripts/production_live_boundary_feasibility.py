#!/usr/bin/env python3
"""Inventory production-live p_game_hit support for future MDP boundaries.

This evidence-only tool does not derive boundaries, reconcile probability
scales, write policy artifacts, or make deploy claims. It inventories the
production-live probabilities the MDP actually acts on, then assesses whether
there is enough config-stable live-scale support to justify a future
pre-registered boundary derivation or reconciliation slice.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCHEMA_VERSION = "production_live_boundary_feasibility_v1"
DEFAULT_PICKS_DIR = Path("data/picks")
DEFAULT_OUTPUT = Path("data/validation/production_live_boundary_feasibility_2026-05-25.json")
DEFAULT_DIRECT_MIN_RANK1 = 250
DEFAULT_DIRECT_HOLDOUT_RANK1 = 100
DEFAULT_RECONCILE_MIN_RANK1 = 50
DEFAULT_N_BINS = 5
SCALE_MEAN_OR_MEDIAN_DELTA_WARN = 0.03
SCALE_ANCHOR_QUANTILE_DELTA_WARN = 0.05

KNOWN_SCALE_EVENTS = [
    {
        "date": "2026-04-14",
        "event": "pitcher_hr_30g_min_periods_default_7_and_rookie_gate_live_era",
        "reason": (
            "CLAUDE.md documents scale-affecting production feature defaults "
            "around this date; pick JSON does not persist env-level feature "
            "configuration, so this is an external cutpoint for inventory."
        ),
    },
]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _date_from_path(path: Path) -> date | None:
    try:
        return date.fromisoformat(path.stem)
    except ValueError:
        return None


def _slot_row(
    *,
    body: dict[str, Any],
    slot: str,
    rank: int,
    payload: dict[str, Any],
    source_file: Path,
) -> dict[str, Any] | None:
    p_game_hit = payload.get("p_game_hit")
    if p_game_hit is None:
        return None
    return {
        "source": "pick_json",
        "source_file": str(source_file),
        "date": body["date"],
        "run_time": body.get("run_time"),
        "slot": slot,
        "rank": int(rank),
        "p_game_hit": float(p_game_hit),
        "batter_id": payload.get("batter_id"),
        "batter_name": payload.get("batter_name"),
        "projected_lineup": payload.get("projected_lineup"),
        "game_pk": payload.get("game_pk"),
        "model_git_sha": body.get("model_git_sha"),
        "model_pickle_sha256": body.get("model_pickle_sha256"),
        "policy_npz_sha256": body.get("policy_npz_sha256"),
        "feature_env_hash": body.get("feature_env_hash"),
        "feature_env_schema_version": body.get("feature_env_schema_version"),
        "notification_sent": body.get("notification_sent"),
        "notification_channel": body.get("notification_channel"),
        "result": body.get("result"),
        "slot_results_present": body.get("slot_results") is not None,
    }


def load_pick_slot_rows(
    picks_dir: Path,
    *,
    today: date | None = None,
) -> list[dict[str, Any]]:
    if not picks_dir.exists():
        raise FileNotFoundError(f"picks_dir does not exist: {picks_dir}")
    rows: list[dict[str, Any]] = []
    for path in sorted(picks_dir.glob("*.json")):
        if "." in path.stem:
            continue
        pick_date = _date_from_path(path)
        if pick_date is None or (today is not None and pick_date > today):
            continue
        body = json.loads(path.read_text())
        primary = body.get("pick") or {}
        primary_row = _slot_row(
            body=body,
            slot="primary",
            rank=1,
            payload=primary,
            source_file=path,
        )
        if primary_row is not None:
            rows.append(primary_row)
        double_down = body.get("double_down") or None
        if double_down is not None:
            dd_row = _slot_row(
                body=body,
                slot="double_down",
                rank=2,
                payload=double_down,
                source_file=path,
            )
            if dd_row is not None:
                rows.append(dd_row)
    return rows


def load_lineup_evolution_rows(
    picks_dir: Path,
    *,
    today: date | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not picks_dir.exists():
        return rows
    for path in sorted(picks_dir.glob("lineup_evolution_*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            body = json.loads(line)
            try:
                row_date = date.fromisoformat(str(body["date"]))
            except (KeyError, ValueError):
                continue
            if today is not None and row_date > today:
                continue
            for slot, rank in (("primary", 1), ("double_down", 2)):
                payload = body.get(slot)
                if not isinstance(payload, dict) or payload.get("p_game_hit") is None:
                    continue
                rows.append({
                    "source": "lineup_evolution",
                    "source_file": str(path),
                    "date": body["date"],
                    "captured_at": body.get("captured_at"),
                    "run_time": body.get("run_time"),
                    "slot": slot,
                    "rank": rank,
                    "p_game_hit": float(payload["p_game_hit"]),
                    "batter_id": payload.get("batter_id"),
                    "batter_name": payload.get("batter_name"),
                    "projected_lineup": payload.get("projected_lineup"),
                    "game_pk": payload.get("game_pk"),
                })
    return rows


def _values(rows: Iterable[dict[str, Any]], *, rank: int | None = None) -> list[float]:
    out = []
    for row in rows:
        if rank is not None and int(row["rank"]) != rank:
            continue
        out.append(float(row["p_game_hit"]))
    return out


def distribution_summary(values: Sequence[float]) -> dict[str, Any]:
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


def _first_short(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text[:12]


def _date_key(row: dict[str, Any]) -> date:
    return date.fromisoformat(str(row["date"]))


def _window_rows_for_dates(rows: Sequence[dict[str, Any]], dates: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if row["date"] in dates]


def contiguous_windows(
    rows: Sequence[dict[str, Any]],
    *,
    key_fields: Sequence[str],
) -> list[dict[str, Any]]:
    primary = sorted(
        [row for row in rows if int(row["rank"]) == 1],
        key=_date_key,
    )
    windows: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_key: tuple[Any, ...] | None = None
    for row in primary:
        key = tuple(row.get(field) for field in key_fields)
        if current_key is None or key == current_key:
            current.append(row)
        else:
            windows.append(_summarize_window(rows, current, key_fields, current_key))
            current = [row]
        current_key = key
    if current and current_key is not None:
        windows.append(_summarize_window(rows, current, key_fields, current_key))
    return windows


def _summarize_window(
    all_rows: Sequence[dict[str, Any]],
    primary_rows: Sequence[dict[str, Any]],
    key_fields: Sequence[str],
    key: tuple[Any, ...],
) -> dict[str, Any]:
    dates = {row["date"] for row in primary_rows}
    window_rows = _window_rows_for_dates(all_rows, dates)
    rank1_values = _values(window_rows, rank=1)
    rank2_values = _values(window_rows, rank=2)
    return {
        "start_date": min(dates),
        "end_date": max(dates),
        "days": len(dates),
        "rank1_n": len(rank1_values),
        "rank2_n": len(rank2_values),
        "rank1_distribution": distribution_summary(rank1_values),
        "rank2_distribution": distribution_summary(rank2_values),
        "key_fields": list(key_fields),
        "key": {
            field: key[index]
            for index, field in enumerate(key_fields)
        },
        "key_short": {
            field: _first_short(key[index])
            for index, field in enumerate(key_fields)
        },
    }


def _coverage(rows: Sequence[dict[str, Any]], field: str) -> dict[str, Any]:
    total = len([row for row in rows if int(row["rank"]) == 1])
    present = len([row for row in rows if int(row["rank"]) == 1 and row.get(field)])
    return {
        "rank1_total": total,
        "rank1_present": present,
        "rank1_missing": total - present,
        "rank1_present_fraction": None if total == 0 else float(present / total),
    }


def _hist_profile_paths(profiles_dir: Path) -> list[Path]:
    if not profiles_dir.exists():
        return []
    return sorted(profiles_dir.glob("backtest_*.parquet"))


def load_historical_rank1_values(profiles_dir: Path | None) -> list[float]:
    if profiles_dir is None:
        return []
    frames = []
    for path in _hist_profile_paths(profiles_dir):
        frame = pd.read_parquet(path, columns=["rank", "p_game_hit"])
        frames.append(frame[frame["rank"] == 1])
    if not frames:
        return []
    return pd.concat(frames, ignore_index=True)["p_game_hit"].astype(float).tolist()


def scale_parity(
    live_values: Sequence[float],
    historical_values: Sequence[float],
) -> dict[str, Any]:
    live = distribution_summary(live_values)
    historical = distribution_summary(historical_values)
    if live["n"] == 0 or historical["n"] == 0:
        return {
            "available": False,
            "live": live,
            "historical": historical,
            "material_divergence": None,
            "reason": "missing live or historical rank-1 distribution",
        }
    mean_delta = float(live["mean"] - historical["mean"])
    median_delta = float(live["q50"] - historical["q50"])
    anchor_deltas = {
        key: float(live[key] - historical[key])
        for key in ("q10", "q50", "q90")
    }
    max_anchor_abs_delta = max(abs(value) for value in anchor_deltas.values())
    material = (
        abs(mean_delta) >= SCALE_MEAN_OR_MEDIAN_DELTA_WARN
        or abs(median_delta) >= SCALE_MEAN_OR_MEDIAN_DELTA_WARN
        or max_anchor_abs_delta >= SCALE_ANCHOR_QUANTILE_DELTA_WARN
    )
    return {
        "available": True,
        "live": live,
        "historical": historical,
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
            "live and historical probability scales diverge materially"
            if material
            else "no material scale divergence by pre-set thresholds"
        ),
    }


def _best_window(windows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not windows:
        return None
    return max(windows, key=lambda row: (row["rank1_n"], row["rank2_n"], row["end_date"]))


def _recent_non_null_policy_window(windows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    non_null = [
        row for row in windows
        if row["key"].get("policy_npz_sha256") is not None
    ]
    if not non_null:
        return None
    return max(non_null, key=lambda row: (row["end_date"], row["rank1_n"]))


def _best_non_null_window(
    windows: Sequence[dict[str, Any]],
    *,
    required_fields: Sequence[str],
) -> dict[str, Any] | None:
    eligible = [
        row for row in windows
        if all(row["key"].get(field) is not None for field in required_fields)
    ]
    return _best_window(eligible)


def feasibility_decision(
    *,
    best_policy_window: dict[str, Any] | None,
    best_strict_window: dict[str, Any] | None,
    direct_min_rank1: int,
    direct_holdout_rank1: int,
    reconcile_min_rank1: int,
) -> dict[str, Any]:
    policy_n = 0 if best_policy_window is None else int(best_policy_window["rank1_n"])
    strict_n = 0 if best_strict_window is None else int(best_strict_window["rank1_n"])
    direct_required = int(direct_min_rank1 + direct_holdout_rank1)
    direct_feasible = strict_n >= direct_required
    reconciliation_feasible = policy_n >= reconcile_min_rank1
    if direct_feasible:
        decision = "DIRECT_DERIVATION_FEASIBLE_REQUIRES_PREREG"
    elif reconciliation_feasible:
        decision = "DIRECT_NOT_FEASIBLE_RECONCILIATION_CANDIDATE_REQUIRES_PREREG"
    else:
        decision = "NOT_FEASIBLE_DIRECT_OR_RECONCILIATION_NEEDS_MORE_LIVE_N"
    return {
        "decision": decision,
        "direct_derivation_feasible": bool(direct_feasible),
        "reconciliation_feasible": bool(reconciliation_feasible),
        "policy_stable_rank1_n": policy_n,
        "strict_git_policy_rank1_n": strict_n,
        "thresholds": {
            "direct_min_rank1_fit": int(direct_min_rank1),
            "direct_min_rank1_holdout": int(direct_holdout_rank1),
            "direct_required_rank1_total": int(direct_required),
            "reconcile_min_rank1": int(reconcile_min_rank1),
        },
        "reason": (
            "direct live-boundary derivation needs enough live points for both "
            "boundary fit and holdout; reconciliation still needs enough "
            "policy-stable live rank-1 points to estimate live/backtest scale."
        ),
    }


def build_inventory(
    *,
    picks_dir: Path,
    historical_profiles_dir: Path | None = None,
    today: date | None = None,
    direct_min_rank1: int = DEFAULT_DIRECT_MIN_RANK1,
    direct_holdout_rank1: int = DEFAULT_DIRECT_HOLDOUT_RANK1,
    reconcile_min_rank1: int = DEFAULT_RECONCILE_MIN_RANK1,
) -> dict[str, Any]:
    pick_rows = load_pick_slot_rows(picks_dir, today=today)
    lineup_rows = load_lineup_evolution_rows(picks_dir, today=today)
    primary_values = _values(pick_rows, rank=1)
    rank2_values = _values(pick_rows, rank=2)

    policy_windows = contiguous_windows(pick_rows, key_fields=["policy_npz_sha256"])
    strict_windows = contiguous_windows(
        pick_rows,
        key_fields=["model_git_sha", "policy_npz_sha256"],
    )
    complete_scale_windows = contiguous_windows(
        pick_rows,
        key_fields=["model_pickle_sha256", "feature_env_hash"],
    )
    best_policy_window = _best_window(policy_windows)
    best_strict_window = _best_window(strict_windows)
    best_non_null_policy_window = _best_non_null_window(
        policy_windows,
        required_fields=["policy_npz_sha256"],
    )
    best_non_null_strict_window = _best_non_null_window(
        strict_windows,
        required_fields=["model_git_sha", "policy_npz_sha256"],
    )
    best_non_null_complete_scale_window = _best_non_null_window(
        complete_scale_windows,
        required_fields=["model_pickle_sha256", "feature_env_hash"],
    )
    recent_policy_window = _recent_non_null_policy_window(policy_windows)

    historical_values = load_historical_rank1_values(historical_profiles_dir)
    all_scale_parity = scale_parity(primary_values, historical_values)
    recent_scale_parity = (
        scale_parity(
            _values(
                _window_rows_for_dates(
                    pick_rows,
                    set() if recent_policy_window is None else {
                        row["date"]
                        for row in pick_rows
                        if recent_policy_window["start_date"] <= row["date"] <= recent_policy_window["end_date"]
                    },
                ),
                rank=1,
            ),
            historical_values,
        )
        if recent_policy_window is not None
        else {"available": False, "reason": "no non-null policy window"}
    )

    decision = feasibility_decision(
        best_policy_window=best_non_null_policy_window,
        best_strict_window=best_non_null_strict_window,
        direct_min_rank1=direct_min_rank1,
        direct_holdout_rank1=direct_holdout_rank1,
        reconcile_min_rank1=reconcile_min_rank1,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date": (today or date.today()).isoformat(),
        "artifact_role": "production_live_boundary_feasibility",
        "production_deploy_claim": False,
        "writes_policy_artifact": False,
        "derives_boundaries": False,
        "builds_reconciliation_map": False,
        "inputs": {
            "picks_dir": str(picks_dir),
            "historical_profiles_dir": str(historical_profiles_dir)
            if historical_profiles_dir is not None
            else None,
            "n_bins": DEFAULT_N_BINS,
        },
        "known_scale_events": KNOWN_SCALE_EVENTS,
        "surface_inventory": {
            "pick_json": {
                "rank1_n": len(primary_values),
                "rank2_n": len(rank2_values),
                "date_min": min((row["date"] for row in pick_rows), default=None),
                "date_max": max((row["date"] for row in pick_rows), default=None),
                "rank1_distribution": distribution_summary(primary_values),
                "rank2_distribution": distribution_summary(rank2_values),
                "model_git_sha_coverage": _coverage(pick_rows, "model_git_sha"),
                "policy_npz_sha256_coverage": _coverage(pick_rows, "policy_npz_sha256"),
                "model_pickle_sha256_coverage": _coverage(pick_rows, "model_pickle_sha256"),
                "feature_env_hash_coverage": _coverage(pick_rows, "feature_env_hash"),
            },
            "lineup_evolution": {
                "row_count": len(lineup_rows),
                "rank1_n": len(_values(lineup_rows, rank=1)),
                "rank2_n": len(_values(lineup_rows, rank=2)),
                "date_min": min((row["date"] for row in lineup_rows), default=None),
                "date_max": max((row["date"] for row in lineup_rows), default=None),
                "rank1_distribution": distribution_summary(_values(lineup_rows, rank=1)),
                "rank2_distribution": distribution_summary(_values(lineup_rows, rank=2)),
                "role": (
                    "audit trail of saved primary/double-down decisions; not a "
                    "full ranked candidate slate and not sufficient for boundary "
                    "derivation by itself"
                ),
            },
            "full_ranked_live_surface": {
                "available_in_picks_snapshot": False,
                "reason": (
                    "data/picks JSON and lineup_evolution JSONL contain selected "
                    "primary/double-down slots, not full ranked daily slates"
                ),
            },
        },
        "windows": {
            "policy_hash_windows": policy_windows,
            "strict_model_git_policy_windows": strict_windows,
            "complete_scale_windows": complete_scale_windows,
            "best_policy_hash_window": best_policy_window,
            "best_strict_model_git_policy_window": best_strict_window,
            "best_complete_scale_window": _best_window(complete_scale_windows),
            "best_non_null_policy_hash_window": best_non_null_policy_window,
            "best_non_null_strict_model_git_policy_window": best_non_null_strict_window,
            "best_non_null_complete_scale_window": best_non_null_complete_scale_window,
            "recent_non_null_policy_hash_window": recent_policy_window,
            "stability_caveat": (
                "model_pickle_sha256 plus feature_env_hash is the preferred "
                "future live-scale key. Older pick JSON lacks feature_env_hash, "
                "so policy hash and git SHA remain observed provenance proxies "
                "for pre-instrumentation rows; daily git changes can reflect "
                "docs/tooling rather than model scale changes. Do not backfill "
                "feature_env_hash; complete-fingerprint live-N starts only after "
                "the instrumentation is deployed."
            ),
        },
        "scale_parity": {
            "all_pick_json_rank1_vs_historical_estimated_pa": all_scale_parity,
            "recent_non_null_policy_window_vs_historical_estimated_pa": recent_scale_parity,
        },
        "feasibility": decision,
        "recommended_next_step": (
            "WAIT_FOR_MORE_LIVE_N"
            if decision["decision"] == "NOT_FEASIBLE_DIRECT_OR_RECONCILIATION_NEEDS_MORE_LIVE_N"
            else "PREREGISTER_BACKTEST_TO_LIVE_RECONCILIATION"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--picks-dir", type=Path, default=DEFAULT_PICKS_DIR)
    parser.add_argument("--historical-profiles-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--date", dest="today", default=None)
    parser.add_argument("--direct-min-rank1", type=int, default=DEFAULT_DIRECT_MIN_RANK1)
    parser.add_argument("--direct-holdout-rank1", type=int, default=DEFAULT_DIRECT_HOLDOUT_RANK1)
    parser.add_argument("--reconcile-min-rank1", type=int, default=DEFAULT_RECONCILE_MIN_RANK1)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    result = build_inventory(
        picks_dir=args.picks_dir,
        historical_profiles_dir=args.historical_profiles_dir,
        today=date.fromisoformat(args.today) if args.today else None,
        direct_min_rank1=args.direct_min_rank1,
        direct_holdout_rank1=args.direct_holdout_rank1,
        reconcile_min_rank1=args.reconcile_min_rank1,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2 if args.pretty else None, default=_json_default, sort_keys=True)
    )
    print(json.dumps({
        "output": str(args.output),
        "decision": result["feasibility"]["decision"],
        "rank1_n": result["surface_inventory"]["pick_json"]["rank1_n"],
        "rank2_n": result["surface_inventory"]["pick_json"]["rank2_n"],
        "production_deploy_claim": result["production_deploy_claim"],
        "writes_policy_artifact": result["writes_policy_artifact"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
