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

Item #5 from 2026-04-28 retro: when called with `history_path`, this
appends a daily JSONL row and on Tuesday EOD additionally emits a
weekly INFO digest with rolling stats. Lets us catch slow week-over-week
RSS creep before it crosses the absolute thresholds. Tuesday picked on
action-window grounds (mid-week, not buried in Monday's alert pile-up,
leaves work days to address before weekend).
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from statistics import median

from bts.health.alert import Alert

log = logging.getLogger(__name__)

SOURCE = "memory_growth"

DEFAULT_THRESHOLDS = {
    # Tuned 2026-04-28 evening after first real CRITICAL fired at 2.9 GB.
    # Previous thresholds (200/500/1024) were calibrated against sleeping-state
    # baseline RSS (~90 MB), but the scheduler's bts-run path loads 1.5M PA ×
    # dozens of features into pandas + trains 12 LightGBM blend models in
    # process. That legitimately allocates 2-3 GB which CPython doesn't return
    # to the OS. Post-prediction baseline is fundamentally different from
    # sleeping baseline. New thresholds:
    #   INFO 1 GB:    notable growth, worth observing
    #   WARN 3 GB:    significant — beyond expected post-prediction RSS
    #   CRITICAL 6 GB: ~40% of bts-mlb's 16 GB, likely real leak
    "info_mb": 1024,
    "warn_mb": 3072,
    "critical_mb": 6144,
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


def _append_history(history_path: Path, today: date, rss_mb: float) -> bool:
    """Append today's RSS row to the JSONL history. Returns True on success."""
    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a") as f:
            f.write(json.dumps({"date": today.isoformat(), "rss_mb": rss_mb}) + "\n")
        return True
    except OSError as e:
        log.warning(f"could not append memory_growth history at {history_path}: {e}")
        return False


def _read_recent_history(history_path: Path, today: date, days: int = 14) -> list[dict]:
    """Return rows from the last `days` days of history, oldest-first."""
    if not history_path.exists():
        return []
    rows = []
    cutoff = today - timedelta(days=days)
    try:
        for line in history_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                row_date = date.fromisoformat(row["date"])
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
            if row_date >= cutoff:
                rows.append(row)
    except OSError as e:
        log.warning(f"could not read memory_growth history at {history_path}: {e}")
    return rows


def _weekly_digest_alert(rows: list[dict]) -> Alert | None:
    """Build the Tuesday-EOD INFO digest alert from recent history rows.

    Reports n, median over last 14d, latest, and 7d trend (median of last
    7d minus median of preceding 7d, or N/A if not enough data).
    """
    if not rows:
        return None
    latest = rows[-1]["rss_mb"]
    n = len(rows)
    med14 = median(r["rss_mb"] for r in rows)
    if n >= 14:
        recent7 = [r["rss_mb"] for r in rows[-7:]]
        prev7 = [r["rss_mb"] for r in rows[-14:-7]]
        trend_str = f"7d trend {(median(recent7) - median(prev7)):+.1f} MB"
    else:
        trend_str = f"7d trend N/A (only {n} day{'s' if n != 1 else ''} of data)"
    msg = (
        f"weekly memory digest: {n} data point{'s' if n != 1 else ''}, "
        f"median {med14:.1f} MB, latest {latest:.1f} MB, {trend_str}"
    )
    return Alert(level="INFO", source=SOURCE, message=msg)


def check(
    pid: int,
    thresholds: dict | None = None,
    history_path: Path | None = None,
    today: date | None = None,
) -> list[Alert]:
    """Returns alerts: threshold-based RSS + (Tuesday only) weekly digest.

    If `history_path` is provided, appends today's RSS as a JSONL row. On
    Tuesdays (weekday() == 1) emits an additional INFO digest summarising
    the last 14 days of history. Backward-compat: when `history_path` is
    None, behavior matches the pre-2026-04-29 check (threshold only).
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    rss_kb = _read_vmrss_kb(pid)
    if rss_kb is None:
        return []
    rss_mb = rss_kb / 1024

    alerts: list[Alert] = []

    # Threshold-based alert (existing behavior)
    if rss_mb >= t["info_mb"]:
        if rss_mb >= t["critical_mb"]:
            level = "CRITICAL"
        elif rss_mb >= t["warn_mb"]:
            level = "WARN"
        else:
            level = "INFO"
        alerts.append(Alert(
            level=level,
            source=SOURCE,
            message=f"scheduler RSS {rss_mb:.1f} MB (pid={pid}) — typical baseline ~90 MB",
        ))

    # History append + Tuesday digest (item #5)
    if history_path is not None:
        if today is None:
            today = date.today()
        _append_history(history_path, today, rss_mb)
        # weekday(): Mon=0, Tue=1, ...
        if today.weekday() == 1:
            rows = _read_recent_history(history_path, today, days=14)
            digest = _weekly_digest_alert(rows)
            if digest is not None:
                alerts.append(digest)

    return alerts
