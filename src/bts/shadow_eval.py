"""Shadow pick result backfill, status, and quality evaluation helpers.

The context-stack shadow model uses semantic local versioning
(``context_stack_shadow_v1`` -> ``v2`` when the shadow feature stack or
selection code changes). This is intentionally different from #16's
frozen-launch-SHA candidate-cycle discipline: shadow status is an operational
monitor for an ongoing sidecar model, not a deployment claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from bts.picks import API_BASE, grade_pick_in_feed
from bts.picks import (
    DailyPick, Pick, load_pick, load_shadow_pick,
    resolve_pick_slot_result,
)

HitChecker = Callable[[int | None, int, str | None, str | None, str | None], bool | str | None]

RESULT_VALUES = {"hit", "miss"}
RESOLVED_RESULT_VALUES = {"hit", "miss", "void"}
VOID_DETAILED_STATES = {"postponed", "cancelled", "canceled"}
SHADOW_MODEL_NAME = "context_stack_shadow_v2"  # v2 2026-07-07: +park_drag_delta (5th context col)
SHADOW_STATUS_SCHEMA_VERSION = "bts_shadow_cycle_status_v1"
SHADOW_STATUS_DEFAULT_MIN_DAYS = 30


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _current_git_commit(cwd: str | Path = ".") -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(cwd),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _date_from_shadow_file(path: Path) -> str:
    return path.name.removesuffix(".shadow.json")


def _pick_key(pick: Pick | None) -> str | None:
    if pick is None:
        return None
    if pick.batter_id is not None:
        return f"id:{pick.batter_id}"
    return f"name:{pick.batter_name.lower()}"


def _pick_slots(daily: DailyPick) -> list[tuple[str, Pick]]:
    slots = [("pick", daily.pick)]
    if daily.double_down is not None:
        slots.append(("double_down", daily.double_down))
    return slots


def _hit_to_slot_result(hit: bool | str | None) -> str | None:
    if hit == "void":
        return "void"
    if hit is None:
        return None
    return "hit" if bool(hit) else "miss"


def _slot_result_to_hit(result: str | None) -> bool | None:
    if result == "hit":
        return True
    if result == "miss":
        return False
    return None


def _is_void_status(data: dict) -> bool:
    status = data.get("gameData", {}).get("status", {})
    detailed = (status.get("detailedState") or "").strip().lower()
    coded = (status.get("codedGameState") or status.get("statusCode") or "").strip().upper()
    return detailed in VOID_DETAILED_STATES or coded in {"D", "C", "DR"}


def _slot_summary(role: str, pick: Pick, result: str | None = None) -> dict:
    return {
        "role": role,
        "batter_name": pick.batter_name,
        "batter_id": pick.batter_id,
        "team": pick.team,
        "game_pk": pick.game_pk,
        "p_game_hit": pick.p_game_hit,
        "hit": _slot_result_to_hit(result),
        "slot_result": result,
        "data_source": None,
    }


def _cached_game_path(raw_dir: Path | None, date: str, game_pk: int | None) -> Path | None:
    if raw_dir is None or game_pk is None:
        return None
    return Path(raw_dir) / date[:4] / f"{game_pk}.json"


def _resolve_hit(
    pick: Pick,
    date: str,
    *,
    raw_dir: Path | None,
    hit_checker: HitChecker | None,
) -> dict:
    if hit_checker is not None:
        hit = hit_checker(pick.game_pk, pick.batter_id, pick.batter_name, date, pick.team)
        slot_result = _hit_to_slot_result(hit)
        return {
            "hit": _slot_result_to_hit(slot_result),
            "slot_result": slot_result,
            "data_source": "test",
            "api_calls": [],
            "response_summary": f"test_slot_result={slot_result}",
        }

    cached_path = _cached_game_path(raw_dir, date, pick.game_pk)
    if cached_path is not None and cached_path.exists():
        data = json.loads(cached_path.read_text())
        status = data["gameData"]["status"]["abstractGameCode"]
        slot_result = None
        if _is_void_status(data):
            slot_result = "void"
        elif status == "F":
            # grade_pick_in_feed returns the status string directly ("hit"/"miss"/"void"/None),
            # preserving a suspended-game "void" that the bool _check_hit_in_game would collapse
            # to "miss". (resolve_pick_slot_result on the live-API path below is already correct.)
            slot_result = grade_pick_in_feed(data, pick.batter_id, pick.batter_name)
        return {
            "hit": _slot_result_to_hit(slot_result),
            "slot_result": slot_result,
            "data_source": "cached_game_json",
            "api_calls": [],
            "response_summary": f"cached_status={status}; slot_result={slot_result}",
        }

    checked_at = _now_iso()
    slot_result = resolve_pick_slot_result(pick, date)
    api_calls = [{
        "checked_at": checked_at,
        "endpoint": (
            f"{API_BASE}/api/v1.1/game/{pick.game_pk}/feed/live"
            if pick.game_pk is not None
            else f"{API_BASE}/api/v1/schedule?sportId=1&date={date}"
        ),
        "response_summary": f"slot_result={slot_result}",
    }]
    if pick.game_pk is not None:
        api_calls.append({
            "checked_at": checked_at,
            "endpoint": f"{API_BASE}/api/v1/schedule?sportId=1&date={date}",
            "response_summary": "possible fallback if batter was not found in primary game feed",
        })
    return {
        "hit": _slot_result_to_hit(slot_result),
        "slot_result": slot_result,
        "data_source": "mlb_api",
        "api_calls": api_calls,
        "response_summary": f"slot_result={slot_result}",
    }


def evaluate_daily_pick(
    daily: DailyPick | None,
    date: str,
    *,
    raw_dir: Path | None = None,
    hit_checker: HitChecker | None = None,
) -> dict:
    """Evaluate a DailyPick with the same day-level semantics as BTS scoring."""
    if daily is None:
        return {
            "status": "missing",
            "recorded_result": None,
            "evaluated_result": None,
            "slots": [],
            "api_calls": [],
            "error": "pick file missing",
        }

    slots = []
    api_calls = []
    active_results: list[bool] = []
    slot_results: dict[str, str] = {}
    for role, pick in _pick_slots(daily):
        try:
            evidence = _resolve_hit(
                pick,
                date,
                raw_dir=raw_dir,
                hit_checker=hit_checker,
            )
        except Exception as exc:  # pragma: no cover - message covered in tests
            slots.append(_slot_summary(role, pick, None))
            return {
                "status": "error",
                "recorded_result": daily.result,
                "evaluated_result": None,
                "slot_results": slot_results,
                "slots": slots,
                "api_calls": api_calls,
                "error": str(exc),
            }
        slot_result = evidence["slot_result"]
        slot = _slot_summary(role, pick, slot_result)
        slot["data_source"] = evidence["data_source"]
        slot["response_summary"] = evidence["response_summary"]
        slots.append(slot)
        api_calls.extend(evidence["api_calls"])
        if slot_result is None:
            return {
                "status": "unresolved",
                "recorded_result": daily.result,
                "evaluated_result": None,
                "slot_results": slot_results,
                "slots": slots,
                "api_calls": api_calls,
                "error": None,
            }
        slot_results[role] = slot_result
        if slot_result != "void":
            active_results.append(slot_result == "hit")

    evaluated = (
        "void"
        if not active_results
        else "hit" if all(active_results) else "miss"
    )
    return {
        "status": "resolved",
        "recorded_result": daily.result,
        "evaluated_result": evaluated,
        "slot_results": slot_results,
        "slots": slots,
        "api_calls": api_calls,
        "error": None,
    }


def _daily_decision_summary(daily: DailyPick | None) -> dict:
    if daily is None:
        return {"primary_key": None, "pair_keys_unordered": []}
    keys = [_pick_key(pick) for _, pick in _pick_slots(daily)]
    return {
        "primary_key": keys[0],
        "pair_keys_unordered": sorted(key for key in keys if key is not None),
    }


def _wilson_interval(hits: int, total: int, z: float = 1.959963984540054) -> list[float | None]:
    if total == 0:
        return [None, None]
    p = hits / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def _sign_test_p_two_sided(prod_only: int, shadow_only: int) -> float | None:
    n = prod_only + shadow_only
    if n == 0:
        return None
    k = min(prod_only, shadow_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def _bootstrap_gap_ci(
    pairs: list[tuple[bool, bool]],
    *,
    n_bootstrap: int,
    seed: int,
) -> list[float | None]:
    if not pairs or n_bootstrap <= 0:
        return [None, None]
    rng = random.Random(seed)
    n = len(pairs)
    gaps = []
    for _ in range(n_bootstrap):
        prod_hits = 0
        shadow_hits = 0
        for _ in range(n):
            prod_hit, shadow_hit = pairs[rng.randrange(n)]
            prod_hits += int(prod_hit)
            shadow_hits += int(shadow_hit)
        gaps.append(shadow_hits / n - prod_hits / n)
    gaps.sort()
    lo_idx = int(0.025 * (n_bootstrap - 1))
    hi_idx = int(0.975 * (n_bootstrap - 1))
    return [gaps[lo_idx], gaps[hi_idx]]


def compute_shadow_quality(
    rows: list[dict],
    *,
    n_bootstrap: int = 10_000,
    seed: int = 57,
) -> dict:
    """Compute paired production-vs-shadow quality metrics from manifest rows."""
    pairs: list[tuple[bool, bool]] = []
    both_hit = both_miss = prod_only = shadow_only = 0
    primary_agree = pair_agree = 0
    production_mismatches = []
    production_counts = {"hit": 0, "miss": 0, "void": 0, "unresolved": 0}
    shadow_counts = {"hit": 0, "miss": 0, "void": 0, "unresolved": 0}

    for row in rows:
        prod_eval = row["production"]["evaluated_result"]
        shadow_eval = row["shadow"]["evaluated_result"]
        production_counts[prod_eval if prod_eval in RESOLVED_RESULT_VALUES else "unresolved"] += 1
        shadow_counts[shadow_eval if shadow_eval in RESOLVED_RESULT_VALUES else "unresolved"] += 1
        if row["production"]["recorded_result"] in RESULT_VALUES and prod_eval in RESULT_VALUES:
            if row["production"]["recorded_result"] != prod_eval:
                production_mismatches.append({
                    "date": row["date"],
                    "recorded_result": row["production"]["recorded_result"],
                    "evaluated_result": prod_eval,
                })

        if row["production_decision"]["primary_key"] == row["shadow_decision"]["primary_key"]:
            primary_agree += 1
        if (
            row["production_decision"]["pair_keys_unordered"]
            == row["shadow_decision"]["pair_keys_unordered"]
        ):
            pair_agree += 1

        if prod_eval not in RESULT_VALUES or shadow_eval not in RESULT_VALUES:
            continue

        prod_hit = prod_eval == "hit"
        shadow_hit = shadow_eval == "hit"
        pairs.append((prod_hit, shadow_hit))
        if prod_hit and shadow_hit:
            both_hit += 1
        elif not prod_hit and not shadow_hit:
            both_miss += 1
        elif prod_hit:
            prod_only += 1
        else:
            shadow_only += 1

    total = len(rows)
    evaluable = len(pairs)
    prod_hits = sum(int(prod) for prod, _ in pairs)
    shadow_hits = sum(int(shadow) for _, shadow in pairs)
    prod_rate = prod_hits / evaluable if evaluable else None
    shadow_rate = shadow_hits / evaluable if evaluable else None
    gap = shadow_rate - prod_rate if evaluable else None

    return {
        "n_days": total,
        "n_evaluable_days": evaluable,
        "n_unevaluable_days": total - evaluable,
        "production_day_hit_rate": {
            "hits": prod_hits,
            "total": evaluable,
            "rate": prod_rate,
            "wilson_95": _wilson_interval(prod_hits, evaluable),
        },
        "shadow_day_hit_rate": {
            "hits": shadow_hits,
            "total": evaluable,
            "rate": shadow_rate,
            "wilson_95": _wilson_interval(shadow_hits, evaluable),
        },
        "shadow_minus_production_hit_rate": {
            "value": gap,
            "paired_bootstrap_95": _bootstrap_gap_ci(pairs, n_bootstrap=n_bootstrap, seed=seed),
            "n_bootstrap": n_bootstrap,
            "seed": seed,
        },
        "paired_outcomes": {
            "both_hit": both_hit,
            "both_miss": both_miss,
            "production_only_hit": prod_only,
            "shadow_only_hit": shadow_only,
            "sign_test_p_two_sided": _sign_test_p_two_sided(prod_only, shadow_only),
        },
        "outcome_counts": {
            "production": production_counts,
            "shadow": shadow_counts,
        },
        "decision_agreement": {
            "primary": {
                "count": primary_agree,
                "total": total,
                "rate": primary_agree / total if total else None,
            },
            "pair_unordered": {
                "count": pair_agree,
                "total": total,
                "rate": pair_agree / total if total else None,
            },
        },
        "production_recorded_mismatches": production_mismatches,
        "sample_size_warning": (
            "Live shadow sample is small; treat this as an operational diagnostic, "
            "not promotion-grade evidence."
        ),
    }


def _recorded_quality_row(
    *,
    date: str,
    production: DailyPick | None,
    shadow: DailyPick | None,
) -> dict:
    return {
        "date": date,
        "production_decision": _daily_decision_summary(production),
        "shadow_decision": _daily_decision_summary(shadow),
        "production": {
            "recorded_result": production.result if production else None,
            "evaluated_result": production.result if production else None,
            "slot_results": production.slot_results if production else None,
        },
        "shadow": {
            "recorded_result": shadow.result if shadow else None,
            "evaluated_result": shadow.result if shadow else None,
            "slot_results": shadow.slot_results if shadow else None,
        },
    }


def _pick_summary(daily: DailyPick | None) -> dict | None:
    if daily is None:
        return None
    return {
        "primary": {
            "batter_name": daily.pick.batter_name,
            "batter_id": daily.pick.batter_id,
            "team": daily.pick.team,
            "game_pk": daily.pick.game_pk,
            "p_game_hit": daily.pick.p_game_hit,
        },
        "double_down": (
            {
                "batter_name": daily.double_down.batter_name,
                "batter_id": daily.double_down.batter_id,
                "team": daily.double_down.team,
                "game_pk": daily.double_down.game_pk,
                "p_game_hit": daily.double_down.p_game_hit,
            }
            if daily.double_down
            else None
        ),
        "result": daily.result,
        "slot_results": daily.slot_results,
        "run_time": daily.run_time,
    }


def build_shadow_cycle_status(
    picks_dir: Path,
    *,
    min_days: int = SHADOW_STATUS_DEFAULT_MIN_DAYS,
    generated_at: str | None = None,
    git_commit: str | None = None,
) -> dict:
    """Build a read-only status artifact for the live context-stack shadow cycle.

    This status uses recorded production/shadow pick files only. It is cheap
    enough for daily cron/safety-net runs and deliberately does not re-query
    MLB boxscores. Use ``shadow-backfill-results`` when a full DD-aware
    recompute/audit manifest is needed.
    """
    picks_dir = Path(picks_dir)
    generated_at = generated_at or _now_iso()
    git_commit = git_commit if git_commit is not None else _current_git_commit()

    rows = []
    quality_rows = []
    for shadow_path in sorted(picks_dir.glob("*.shadow.json")):
        date = _date_from_shadow_file(shadow_path)
        prod_path = picks_dir / f"{date}.json"
        shadow = load_shadow_pick(date, picks_dir)
        production = load_pick(date, picks_dir) if prod_path.exists() else None
        prod_result = production.result if production else None
        shadow_result = shadow.result if shadow else None
        production_decision = _daily_decision_summary(production)
        shadow_decision = _daily_decision_summary(shadow)
        primary_agree = production_decision["primary_key"] == shadow_decision["primary_key"]
        pair_agree = (
            production_decision["pair_keys_unordered"]
            == shadow_decision["pair_keys_unordered"]
        )
        prod_resolved = prod_result in RESOLVED_RESULT_VALUES
        shadow_resolved = shadow_result in RESOLVED_RESULT_VALUES
        prod_evaluable = prod_result in RESULT_VALUES
        shadow_evaluable = shadow_result in RESULT_VALUES
        rows.append({
            "date": date,
            "production_file": str(prod_path) if prod_path.exists() else None,
            "shadow_file": str(shadow_path),
            "shadow_file_sha256": _sha256_file(shadow_path),
            "production": _pick_summary(production),
            "shadow": _pick_summary(shadow),
            "primary_agree": primary_agree,
            "pair_agree": pair_agree,
            "production_result_resolved": prod_resolved,
            "shadow_result_resolved": shadow_resolved,
            "production_result_evaluable": prod_evaluable,
            "shadow_result_evaluable": shadow_evaluable,
        })
        quality_rows.append(_recorded_quality_row(
            date=date,
            production=production,
            shadow=shadow,
        ))

    shadow_files = len(rows)
    paired_files = sum(1 for row in rows if row["production_file"] is not None)
    resolved_shadow = sum(1 for row in rows if row["shadow_result_resolved"])
    void_shadow = sum(
        1 for row in rows
        if row["shadow"] and row["shadow"].get("result") == "void"
    )
    void_production = sum(
        1 for row in rows
        if row["production"] and row["production"].get("result") == "void"
    )
    unresolved_shadow_dates = [
        row["date"] for row in rows if not row["shadow_result_resolved"]
    ]
    missing_production_dates = [
        row["date"] for row in rows if row["production_file"] is None
    ]
    resolved_paired = sum(
        1 for row in rows
        if row["production_result_evaluable"] and row["shadow_result_evaluable"]
    )
    resolved_or_void_paired = sum(
        1 for row in rows
        if row["production_result_resolved"] and row["shadow_result_resolved"]
    )
    primary_agree = sum(1 for row in rows if row["primary_agree"])
    pair_agree = sum(1 for row in rows if row["pair_agree"])

    if shadow_files == 0:
        cycle_state = "no_shadow_files"
    elif missing_production_dates or unresolved_shadow_dates:
        cycle_state = "needs_result_reconciliation"
    elif resolved_paired < min_days:
        cycle_state = "collecting_live_forward"
    else:
        cycle_state = "ready_for_manual_review"

    action_items = []
    if shadow_files == 0:
        action_items.append("Verify scheduler.shadow_model=true and production scheduler logs.")
    if unresolved_shadow_dates:
        action_items.append(
            "Run bts check-results for unresolved dates or use "
            "bts shadow-backfill-results for a reviewed DD-aware recompute."
        )
    if missing_production_dates:
        action_items.append("Investigate shadow files without paired production pick files.")
    if cycle_state == "ready_for_manual_review":
        action_items.append(
            "Review shadow quality under a separate promotion pre-registration before any production change."
        )
    if not action_items:
        action_items.append("Continue collecting live-forward shadow outcomes.")

    return {
        "schema_version": SHADOW_STATUS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "git_commit": git_commit,
        "model": {
            "name": SHADOW_MODEL_NAME,
            "description": "Context-stack shadow model using FEATURE_COLS + CONTEXT_COLS.",
            "activation_config": "scheduler.shadow_model=true",
            "production_deploy_claim": False,
            "artifact_role": "live_shadow_monitoring_status",
            "pick_file_pattern": "*.shadow.json",
            "min_days_for_review": int(min_days),
            "versioning_policy": (
                "Semantic local version; bump when CONTEXT_COLS or shadow "
                "selection code changes. Distinct from #16 frozen-launch-SHA discipline."
            ),
        },
        "history_policy": (
            "This status is a single latest-state artifact intended to be overwritten "
            "by daily monitoring. Snapshot separately if a historical status trail is needed."
        ),
        "picks_dir": str(picks_dir),
        "cycle_state": cycle_state,
        "counts": {
            "shadow_files": shadow_files,
            "paired_production_files": paired_files,
            "resolved_shadow_results": resolved_shadow,
            "unresolved_shadow_results": len(unresolved_shadow_dates),
            "void_shadow_results": void_shadow,
            "void_production_results": void_production,
            "resolved_paired_days": resolved_paired,
            "resolved_or_void_paired_days": resolved_or_void_paired,
            "days_to_review_threshold": max(0, int(min_days) - resolved_paired),
            "primary_agreements": primary_agree,
            "pair_agreements": pair_agree,
        },
        "coverage": {
            "first_shadow_date": rows[0]["date"] if rows else None,
            "latest_shadow_date": rows[-1]["date"] if rows else None,
            "unresolved_shadow_dates": unresolved_shadow_dates,
            "missing_production_dates": missing_production_dates,
        },
        "quality_recorded": compute_shadow_quality(quality_rows, n_bootstrap=0),
        "action_items": action_items,
        "methodology_note": (
            "Recorded status is an operational monitor, not promotion-grade evidence. "
            "Use shadow-backfill-results for reviewed result reconciliation and a "
            "separate pre-registration before promoting CONTEXT_COLS."
        ),
        "rows": rows,
    }


def build_shadow_backfill_manifest(
    picks_dir: Path,
    *,
    raw_dir: Path | None = None,
    n_bootstrap: int = 10_000,
    seed: int = 57,
    hit_checker: HitChecker | None = None,
) -> dict:
    """Dry-run a full shadow-result recompute and return an audit manifest."""
    picks_dir = Path(picks_dir)
    raw_dir = Path(raw_dir) if raw_dir is not None else picks_dir.parent / "raw"
    rows = []
    for shadow_path in sorted(picks_dir.glob("*.shadow.json")):
        date = _date_from_shadow_file(shadow_path)
        prod_path = picks_dir / f"{date}.json"
        shadow = load_shadow_pick(date, picks_dir)
        production = load_pick(date, picks_dir) if prod_path.exists() else None
        prod_eval = evaluate_daily_pick(
            production, date, raw_dir=raw_dir, hit_checker=hit_checker,
        )
        shadow_eval = evaluate_daily_pick(
            shadow, date, raw_dir=raw_dir, hit_checker=hit_checker,
        )

        old_result = shadow.result if shadow else None
        new_result = shadow_eval["evaluated_result"]
        status = shadow_eval["status"]
        apply_eligible = status == "resolved"
        if status == "resolved":
            if old_result is None:
                change_class = "new"
            elif old_result == new_result:
                change_class = "unchanged"
            else:
                change_class = "changed"
        elif status == "unresolved":
            change_class = "skipped"
        else:
            change_class = "error"

        rows.append({
            "date": date,
            "file_path": str(shadow_path.relative_to(picks_dir.parent)),
            "shadow_file": str(shadow_path),
            "shadow_file_sha256_before": _sha256_file(shadow_path),
            "production_file": str(prod_path) if prod_path.exists() else None,
            "production_decision": _daily_decision_summary(production),
            "shadow_decision": _daily_decision_summary(shadow),
            "production": prod_eval,
            "shadow": shadow_eval,
            "api_calls": prod_eval["api_calls"] + shadow_eval["api_calls"],
            "old_shadow_result": old_result,
            "current_result": old_result,
            "new_shadow_result": new_result,
            "recomputed_result": new_result,
            "change_class": change_class,
            "would_change": apply_eligible and old_result != new_result,
            "apply_eligible": apply_eligible,
        })

    counts = {
        "shadow_files": len(rows),
        "resolved": sum(1 for row in rows if row["shadow"]["status"] == "resolved"),
        "unresolved": sum(1 for row in rows if row["shadow"]["status"] == "unresolved"),
        "errors": sum(1 for row in rows if row["shadow"]["status"] == "error"),
        "void": sum(1 for row in rows if row["shadow"]["evaluated_result"] == "void"),
        "would_change": sum(1 for row in rows if row["would_change"]),
        "apply_eligible": sum(1 for row in rows if row["apply_eligible"]),
        "change_class": {
            key: sum(1 for row in rows if row["change_class"] == key)
            for key in ("new", "unchanged", "changed", "skipped", "error")
        },
    }

    return {
        "schema_version": "bts_shadow_result_backfill_manifest_v1",
        "generated_at": _now_iso(),
        "picks_dir": str(picks_dir),
        "raw_dir": str(raw_dir),
        "mode": "dry_run",
        "counts": counts,
        "quality_if_applied": compute_shadow_quality(rows, n_bootstrap=n_bootstrap, seed=seed),
        "rows": rows,
    }


def apply_shadow_backfill_manifest(
    manifest: dict,
    *,
    backup_dir: Path,
) -> dict:
    """Apply a reviewed manifest, preserving pre-backfill shadow files."""
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=False)

    applied = []
    skipped = []
    for row in manifest.get("rows", []):
        if not row.get("would_change") or not row.get("apply_eligible"):
            skipped.append({"date": row.get("date"), "reason": "no_change_or_not_eligible"})
            continue

        shadow_path = Path(row["shadow_file"])
        current_sha = _sha256_file(shadow_path)
        if current_sha != row.get("shadow_file_sha256_before"):
            skipped.append({"date": row.get("date"), "reason": "sha_changed"})
            continue

        backup_path = backup_dir / shadow_path.name
        shutil.copy2(shadow_path, backup_path)

        if not shadow_path.exists():
            skipped.append({"date": row.get("date"), "reason": "shadow_missing_at_apply"})
            continue
        shadow_data = json.loads(shadow_path.read_text())
        shadow_data["result"] = row.get("new_shadow_result")
        shadow_data["slot_results"] = row.get("shadow", {}).get("slot_results")
        shadow_path.write_text(json.dumps(shadow_data, indent=2))
        applied.append({
            "date": row["date"],
            "old_shadow_result": row.get("old_shadow_result"),
            "new_shadow_result": row.get("new_shadow_result"),
            "backup_file": str(backup_path),
            "sha256_before": current_sha,
            "sha256_after": _sha256_file(shadow_path),
        })

    return {
        "applied_at": _now_iso(),
        "backup_dir": str(backup_dir),
        "applied": applied,
        "skipped": skipped,
    }


def write_manifest_json(manifest: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path
