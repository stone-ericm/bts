"""Shadow pick result backfill and quality evaluation helpers."""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from bts.picks import API_BASE, _check_hit_in_game
from bts.picks import (
    DailyPick, Pick, check_hit, load_pick, load_shadow_pick,
)

HitChecker = Callable[[int | None, int, str | None, str | None, str | None], bool | None]

RESULT_VALUES = {"hit", "miss"}


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


def _date_from_shadow_file(path: Path) -> str:
    return path.name.removesuffix(".shadow.json")


def _pick_key(pick: Pick | None) -> str | None:
    if pick is None:
        return None
    if pick.batter_id is not None:
        return f"id:{pick.batter_id}"
    return f"name:{pick.batter_name.lower()}"


def _pick_slots(daily: DailyPick) -> list[tuple[str, Pick]]:
    slots = [("primary", daily.pick)]
    if daily.double_down is not None:
        slots.append(("double_down", daily.double_down))
    return slots


def _slot_summary(role: str, pick: Pick, result: bool | None = None) -> dict:
    return {
        "role": role,
        "batter_name": pick.batter_name,
        "batter_id": pick.batter_id,
        "team": pick.team,
        "game_pk": pick.game_pk,
        "p_game_hit": pick.p_game_hit,
        "hit": result,
        "data_source": None,
    }


def _default_hit_checker(
    game_pk: int | None,
    batter_id: int,
    batter_name: str | None,
    date: str | None,
    team: str | None,
) -> bool | None:
    return check_hit(
        game_pk,
        batter_id,
        batter_name=batter_name,
        date=date,
        team=team,
    )


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
        return {
            "hit": hit,
            "data_source": "test",
            "api_calls": [],
            "response_summary": f"test_hit={hit}",
        }

    cached_path = _cached_game_path(raw_dir, date, pick.game_pk)
    if cached_path is not None and cached_path.exists():
        data = json.loads(cached_path.read_text())
        status = data["gameData"]["status"]["abstractGameCode"]
        hit = None
        if status == "F":
            hit = _check_hit_in_game(data, pick.batter_id, pick.batter_name)
        return {
            "hit": hit,
            "data_source": "cached_game_json",
            "api_calls": [],
            "response_summary": f"cached_status={status}; hit={hit}",
        }

    checked_at = _now_iso()
    hit = _default_hit_checker(
        pick.game_pk,
        pick.batter_id,
        pick.batter_name,
        date,
        pick.team,
    )
    api_calls = [{
        "checked_at": checked_at,
        "endpoint": (
            f"{API_BASE}/api/v1.1/game/{pick.game_pk}/feed/live"
            if pick.game_pk is not None
            else f"{API_BASE}/api/v1/schedule?sportId=1&date={date}"
        ),
        "response_summary": f"check_hit_return={hit}",
    }]
    if pick.game_pk is not None:
        api_calls.append({
            "checked_at": checked_at,
            "endpoint": f"{API_BASE}/api/v1/schedule?sportId=1&date={date}",
            "response_summary": "possible fallback if batter was not found in primary game feed",
        })
    return {
        "hit": hit,
        "data_source": "mlb_api",
        "api_calls": api_calls,
        "response_summary": f"check_hit_return={hit}",
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
    results: list[bool] = []
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
                "slots": slots,
                "api_calls": api_calls,
                "error": str(exc),
            }
        hit = evidence["hit"]
        slot = _slot_summary(role, pick, hit)
        slot["data_source"] = evidence["data_source"]
        slot["response_summary"] = evidence["response_summary"]
        slots.append(slot)
        api_calls.extend(evidence["api_calls"])
        if hit is None:
            return {
                "status": "unresolved",
                "recorded_result": daily.result,
                "evaluated_result": None,
                "slots": slots,
                "api_calls": api_calls,
                "error": None,
            }
        results.append(hit)

    evaluated = "hit" if all(results) else "miss"
    return {
        "status": "resolved",
        "recorded_result": daily.result,
        "evaluated_result": evaluated,
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

    for row in rows:
        prod_eval = row["production"]["evaluated_result"]
        shadow_eval = row["shadow"]["evaluated_result"]
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
