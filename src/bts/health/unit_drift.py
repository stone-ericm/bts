"""Tier 3 health check: systemd unit drift vs repo templates (audit F12).

The production units were hand-maintained snowflakes — the repo's tracked
scheduler unit was the stale Pi5 one (wrong user/path, Type=simple, no
watchdog), so disaster recovery from the repo would have installed a broken
unit, and live edits could drift without review. scripts/systemd/ now holds
the canonical templates; this check is READ-ONLY and flags:

- installed tracked unit != repo template     -> WARN (drift)
- installed tracked unit with no repo template -> WARN (unreproducible)
- installed dir or unit absent                 -> silent (not this box /
                                                  local dev)

Installation remains an explicit operator action
(scripts/install-systemd-hetzner.sh) — a health check must never mutate
service configuration.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from bts.health.alert import Alert

log = logging.getLogger(__name__)

SOURCE = "unit_drift"

TRACKED_UNITS = ("bts-scheduler.service", "bts-dashboard.service")


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def check(*, installed_dir: Path, repo_units_dir: Path) -> list[Alert]:
    installed_dir = Path(installed_dir)
    repo_units_dir = Path(repo_units_dir)
    if not installed_dir.exists():
        return []  # no user units on this machine — local dev

    alerts: list[Alert] = []
    for name in TRACKED_UNITS:
        installed = installed_dir / name
        if not installed.exists():
            continue  # unit not deployed here — silent
        template = repo_units_dir / name
        if not template.exists():
            alerts.append(Alert(
                level="WARN", source=SOURCE,
                message=(f"{name} is installed but has no repo template "
                         f"(expected {template}) — production config is "
                         f"unreproducible; capture it into scripts/systemd/"),
            ))
            continue
        if _sha256(installed) != _sha256(template):
            alerts.append(Alert(
                level="WARN", source=SOURCE,
                message=(f"{name} differs from repo template scripts/systemd/{name} "
                         f"— review the live edit and either update the template "
                         f"(commit) or reinstall via scripts/install-systemd-hetzner.sh"),
            ))
    return alerts
