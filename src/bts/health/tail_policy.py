"""Tier 1 health check: the tail-policy artifact loads and pairs with the base policy.

2026-09-03: once streak 57 is unreachable the pick path switches to
``data/models/mdp_tail_policy.npz`` (exact E[season-best]). The deploy canary
checks only service state + dashboard HTTP, and the strategy loader degrades a
broken tail to a forced single with a log line nobody reads — so this check
loads BOTH artifacts from the deployed paths every health run:

- inside the tail window (0 < days left <= 28: some streak is in the tail regime)
  a missing/invalid/unpaired tail is CRITICAL;
- earlier in the season it is a WARN (the season will reach the window);
- a missing/unreadable base policy is reported at the same level.

Read-only; never mutates artifacts.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

from bts.health.alert import Alert
from bts.simulate.tail_policy import MAX_TAIL_DAYS, TailPolicyError, load_tail_policy, sha256_file

log = logging.getLogger(__name__)

SOURCE = "tail_policy"


def _days_remaining(today: date) -> int:
    from bts.strategy import _days_remaining as strategy_days
    return strategy_days(today.isoformat())


def check(*, base_path: Path, tail_path: Path, today: date | None = None) -> list[Alert]:
    today = today or date.today()
    days = _days_remaining(today)
    in_window = 0 < days <= MAX_TAIL_DAYS
    level = "CRITICAL" if in_window else "WARN"
    window = (f"{days} days left: the tail regime is LIVE for streaks below {57 - 2 * days}"
              if in_window else f"{days} days left: tail window opens at {MAX_TAIL_DAYS}")
    alerts: list[Alert] = []
    base_path, tail_path = Path(base_path), Path(tail_path)
    base_sha = None
    if not base_path.exists():
        alerts.append(Alert(level=level, source=SOURCE,
                            message=f"base policy missing at {base_path} ({window})"))
    else:
        try:
            base_sha = sha256_file(base_path)
        except OSError as exc:
            alerts.append(Alert(level=level, source=SOURCE,
                                message=f"base policy unreadable at {base_path}: {exc} ({window})"))
    if base_sha is None:
        alerts.append(Alert(
            level=level, source=SOURCE,
            message=(f"tail policy unverifiable: base policy unavailable, so the pairing cannot "
                     f"be checked — the pick path is on the forced fallback ({window})"),
        ))
        return alerts
    try:
        load_tail_policy(tail_path, expected_base_sha=base_sha)
    except TailPolicyError as exc:
        alerts.append(Alert(
            level=level, source=SOURCE,
            message=(f"tail policy invalid: {exc} — the pick path is on the forced fallback "
                     f"(skip iff season best unbeatable, else single) ({window}); rebuild with "
                     f"scripts/rebuild_tail_policy.py"),
        ))
    return alerts
