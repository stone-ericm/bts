"""Tier 3: disk fill check.

Monitors disk usage on the path containing picks_dir (typically /home/bts).
Uses shutil.disk_usage which is cgroup-aware on Linux.

Thresholds chosen for bts-mlb (38GB total, ~1GB picks/state, ~15GB raw data):
  INFO:    80% used (still room, but accumulation worth noticing)
  WARN:    90% used (approaching tight)
  CRITICAL: 95% used (imminent fill — game data ingest could fail)
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from bts.health.alert import Alert

log = logging.getLogger(__name__)

SOURCE = "disk_fill"

DEFAULT_THRESHOLDS = {
    "info_pct": 0.80,
    "warn_pct": 0.90,
    "critical_pct": 0.95,
}


def check(path: Path, thresholds: dict | None = None) -> list[Alert]:
    """Returns alerts based on disk-fill % at the given path."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    try:
        usage = shutil.disk_usage(str(path))
    except (FileNotFoundError, PermissionError, OSError) as e:
        log.warning(f"disk_fill check failed for {path}: {e}")
        return []
    pct = usage.used / usage.total
    if pct < t["info_pct"]:
        return []
    if pct >= t["critical_pct"]:
        level = "CRITICAL"
    elif pct >= t["warn_pct"]:
        level = "WARN"
    else:
        level = "INFO"
    used_gb = usage.used / (1024 ** 3)
    total_gb = usage.total / (1024 ** 3)
    return [Alert(
        level=level,
        source=SOURCE,
        message=f"disk usage {pct:.1%} ({used_gb:.1f}/{total_gb:.1f}GB) on {path}",
    )]
