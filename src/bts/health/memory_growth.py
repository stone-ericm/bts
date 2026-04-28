"""Tier 3: scheduler memory growth check.

Reads /proc/self/status VmRSS for the running process. The scheduler
is a long-lived daemon — observed at 90.4 MB after 24h. Memory leaks
would manifest as monotonic growth.

Thresholds:
  INFO:    >= 200 MB    (3-4× normal — investigate)
  WARN:    >= 500 MB
  CRITICAL: >= 1 GB

This works on Linux. Returns [] on non-Linux (Mac dev box) or if /proc
isn't readable for any reason.
"""

from __future__ import annotations

import logging
from pathlib import Path

from bts.health.alert import Alert

log = logging.getLogger(__name__)

SOURCE = "memory_growth"

DEFAULT_THRESHOLDS = {
    "info_mb": 200,
    "warn_mb": 500,
    "critical_mb": 1024,
}


def _read_vmrss_kb(pid: int) -> int | None:
    """Returns VmRSS in kB from /proc/<pid>/status, or None if unavailable."""
    proc_path = Path(f"/proc/{pid}/status")
    if not proc_path.exists():
        return None
    try:
        for line in proc_path.read_text().splitlines():
            if line.startswith("VmRSS:"):
                # Format: "VmRSS:    92376 kB"
                return int(line.split()[1])
    except (OSError, ValueError) as e:
        log.warning(f"could not read {proc_path}: {e}")
    return None


def check(pid: int, thresholds: dict | None = None) -> list[Alert]:
    """Returns alert if process RSS exceeds threshold."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    rss_kb = _read_vmrss_kb(pid)
    if rss_kb is None:
        return []
    rss_mb = rss_kb / 1024
    if rss_mb < t["info_mb"]:
        return []
    if rss_mb >= t["critical_mb"]:
        level = "CRITICAL"
    elif rss_mb >= t["warn_mb"]:
        level = "WARN"
    else:
        level = "INFO"
    return [Alert(
        level=level,
        source=SOURCE,
        message=f"scheduler RSS {rss_mb:.1f} MB (pid={pid}) — typical baseline ~90 MB",
    )]
