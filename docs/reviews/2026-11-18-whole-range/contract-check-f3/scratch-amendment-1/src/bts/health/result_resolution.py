"""Health visibility for stranded result resolution — shadow AND production.

The wait loop in `bts check-results --wait-deadline-et` settles the common
same-night cases (West Coast extras not final at the 01:00 ET cron). This
check surfaces the residual stranded ones — box down overnight, multi-day
suspensions, a deadline exhausted on transient API failures — instead of
leaving them silent: nightly crons only ever grade yesterday, and the 02:00
`reconcile` job skips nonterminal picks, so nothing else revisits a stranded
date. The 2026-07-10 shadow sat unresolved for a month with no signal.

Scanning is version-blind and in-memory (Codex r2 #5, #29): a shadow-stack
version bump must not hide still-stranded older-version results, and the
status JSON artifact written by check-results is another process's output —
not trusted here. A horizon caps the window so ancient pre-discipline files
never produce a standing alert.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

from bts.health.alert import Alert

log = logging.getLogger(__name__)

SOURCE = "result_resolution"

DEFAULT_THRESHOLDS = {
    # Yesterday's results may legitimately still be pending until tonight's
    # cron (and its wait loop) has had its chance — alert from age 2.
    "grace_days": 2,
    "critical_age_days": 7,
    # Only look back this far: stranded-incident response window, and a cap
    # so legacy files from before grading discipline never alert forever.
    "horizon_days": 30,
}

TERMINAL_RESULTS = ("hit", "miss", "void")


def _file_result(path: Path):
    """Return (ok, result) — ok False means unreadable (logged, skipped)."""
    try:
        return True, json.loads(path.read_text()).get("result")
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("result_resolution: could not read %s: %s", path, exc)
        return False, None


def check(
    picks_dir: Path | str,
    *,
    today: date | None = None,
    thresholds: dict | None = None,
) -> list[Alert]:
    picks_path = Path(picks_dir)
    if not picks_path.exists():
        return []

    limits = dict(DEFAULT_THRESHOLDS)
    limits.update(thresholds or {})
    today = today or date.today()

    def _age_level(day: date) -> str | None:
        age = (today - day).days
        if age < limits["grace_days"] or age > limits["horizon_days"]:
            return None
        return "CRITICAL" if age >= limits["critical_age_days"] else "WARN"

    alerts: list[Alert] = []

    # --- shadow side: every *.shadow.json regardless of stack version ---
    for path in sorted(picks_path.glob("*.shadow.json")):
        try:
            day = date.fromisoformat(path.name.split(".")[0])
        except ValueError:
            continue
        level = _age_level(day)
        if level is None:
            continue
        ok, result = _file_result(path)
        if not ok or result in TERMINAL_RESULTS:
            continue
        age = (today - day).days
        alerts.append(Alert(
            level=level,
            source=SOURCE,
            message=(
                f"shadow result for {day} unresolved for {age} days — the "
                f"check-results wait loop did not settle it; run "
                f"`bts check-results --date {day}` (the stale-scoring guard "
                f"keeps it shadow-only)"
            ),
            # One incident per stranded date+side: two stranded dates must
            # both reach the operator, not dedup as "already seen".
            incident_key=f"{SOURCE}:shadow:{day}",
        ))

    # --- production side: a scoreable pick nothing will ever score again ---
    from bts.daily_decision import is_scoreable_commit
    from bts.picks import load_pick

    for offset in range(limits["grace_days"], limits["horizon_days"] + 1):
        day = today - timedelta(days=offset)
        date_str = day.isoformat()
        if not (picks_path / f"{date_str}.json").exists():
            continue
        try:
            daily = load_pick(date_str, picks_path)
        except Exception as exc:
            log.warning("result_resolution: could not load pick %s: %s", date_str, exc)
            continue
        if daily is None or daily.result in TERMINAL_RESULTS:
            continue
        try:
            scoreable = is_scoreable_commit(date_str, picks_path, daily)
        except Exception as exc:
            log.warning("result_resolution: scoreable check failed %s: %s", date_str, exc)
            continue
        if not scoreable:
            continue  # stale preview / undelivered / skip day — nothing to score
        level = _age_level(day)
        if level is None:
            continue
        alerts.append(Alert(
            level=level,
            source=SOURCE,
            message=(
                f"scoreable production pick for {date_str} unresolved for "
                f"{offset} days — daemon polling and the nightly cron both "
                f"failed to score it; investigate, then backfill "
                f"chronologically with `bts check-results --date {date_str} "
                f"--allow-stale-scoring` if warranted"
            ),
            incident_key=f"{SOURCE}:production:{date_str}",
        ))

    return alerts
