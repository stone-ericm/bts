"""Observability for fallback-defer events.

The fallback-defer scheduler path is expected behavior: it archives an unsafe
fallback candidate when a later lineup-confirmation window can still improve
the pick. This check makes that rare event visible without paging unless the
defer breaks the never-miss guarantee.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from bts.health.alert import Alert
from bts.picks import load_pick, pick_was_delivered

log = logging.getLogger(__name__)

SOURCE = "fallback_defer"
ET = ZoneInfo("America/New_York")
EARLIEST_HOUR_ET = 22


def _health_date(today: date | None, now: datetime | None) -> date:
    if today is not None:
        return today
    if now is not None:
        if now.tzinfo:
            return now.astimezone(ET).date()
        return now.replace(tzinfo=ET).date()
    return datetime.now(ET).date()


def _now_et(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(ET)
    return now.astimezone(ET) if now.tzinfo else now.replace(tzinfo=ET)


def _delivery_window_closed(day: date, now: datetime | None) -> bool:
    """Avoid false never-miss pages before same-day delivery can still occur."""
    now_et = _now_et(now)
    if day < now_et.date():
        return True
    return day == now_et.date() and now_et.hour >= EARLIEST_HOUR_ET


def _load_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("could not parse fallback defer artifact %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def _slot(data: dict | None) -> dict:
    if not isinstance(data, dict):
        return {}
    return {
        "batter_id": data.get("batter_id"),
        "batter_name": data.get("batter_name"),
        "team": data.get("team"),
        "p_game_hit": data.get("p_game_hit"),
        "game_pk": data.get("game_pk"),
    }


def _slot_label(slot: dict) -> str:
    name = slot.get("batter_name") or "unknown"
    team = slot.get("team") or "?"
    p = slot.get("p_game_hit")
    try:
        pct = f"{float(p):.1%}"
    except (TypeError, ValueError):
        pct = "p=n/a"
    return f"{name} ({team}) {pct}"


def _same_pick(a: dict, b: dict) -> bool:
    if a.get("batter_id") is not None and b.get("batter_id") is not None:
        return a.get("batter_id") == b.get("batter_id")
    return (
        a.get("batter_name"),
        a.get("team"),
        a.get("game_pk"),
    ) == (
        b.get("batter_name"),
        b.get("team"),
        b.get("game_pk"),
    )


def _p_delta_pp(delivered: dict, deferred: dict) -> str:
    try:
        delta = (float(delivered["p_game_hit"]) - float(deferred["p_game_hit"])) * 100
    except (KeyError, TypeError, ValueError):
        return "n/a"
    return f"{delta:+.1f}pp"


def _critical(day: date, archives: list[Path], detail: str) -> list[Alert]:
    latest = archives[-1].name if archives else "unknown"
    return [Alert(
        level="CRITICAL",
        source=SOURCE,
        message=(
            f"fallback defer fired for {day.isoformat()} but never-miss validation "
            f"failed: {detail}. deferred_archives={len(archives)}, latest={latest}"
        ),
    )]


def check(
    picks_dir: Path,
    today: date | None = None,
    now: datetime | None = None,
) -> list[Alert]:
    """Return INFO for a healthy defer event, CRITICAL if it lost delivery."""
    day = _health_date(today, now)
    archives = sorted((picks_dir / day.isoformat()).glob("deferred_fallback_*.json"))
    if not archives:
        return []

    payloads: list[tuple[Path, dict]] = []
    for archive in archives:
        payload = _load_json(archive)
        if payload is not None:
            payloads.append((archive, payload))
    if not payloads:
        return []

    try:
        final_daily = load_pick(day.isoformat(), picks_dir)
    except Exception as exc:
        if not _delivery_window_closed(day, now):
            return []
        return _critical(
            day,
            archives,
            f"final pick file missing or unreadable: {day.isoformat()}.json ({exc})",
        )
    if final_daily is None:
        if not _delivery_window_closed(day, now):
            return []
        return _critical(
            day,
            archives,
            f"final pick file missing or unreadable: {day.isoformat()}.json",
        )
    if not pick_was_delivered(final_daily):
        if not _delivery_window_closed(day, now):
            return []
        return _critical(
            day,
            archives,
            (
                "final pick exists but no public post or private notification is recorded "
                f"(bluesky_posted={final_daily.bluesky_posted}, "
                f"bluesky_uri={final_daily.bluesky_uri}, "
                f"notification_sent={final_daily.notification_sent}, "
                f"notification_id={final_daily.notification_id})"
            ),
        )

    latest_path, latest_payload = payloads[-1]
    deferred = _slot(latest_payload.get("pick"))
    delivered = _slot(vars(final_daily.pick))
    # Both slots must match (2026-08-30 Codex review #10): a double-down that
    # changed between the archive and the final pick is a different commit.
    deferred_dd = _slot(latest_payload.get("double_down"))
    delivered_dd = _slot(vars(final_daily.double_down)) if final_daily.double_down else {}
    same_pick = _same_pick(delivered, deferred) and (
        (not deferred_dd and not delivered_dd) or _same_pick(delivered_dd, deferred_dd))
    reason = (latest_payload.get("deferred_fallback") or {}).get("reason", "unknown")
    message = (
        f"fallback defer observed for {day.isoformat()}: "
        f"{len(payloads)} archive(s), latest={day.isoformat()}/{latest_path.name}, "
        f"reason={reason}; deferred={_slot_label(deferred)}; "
        f"delivered={_slot_label(delivered)}; same_pick={str(same_pick).lower()}, "
        f"primary_p_delta={_p_delta_pp(delivered, deferred)}, never_miss=confirmed"
    )
    return [Alert(level="INFO", source=SOURCE, message=message)]
