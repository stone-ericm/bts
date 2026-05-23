"""Tier 3: scheduler memory growth check.

Reads /proc/<pid>/status VmRSS for the running scheduler process. The
scheduler is a long-lived daemon, but its post-prediction resident set
can be much larger than its cold sleeping footprint because model and
feature-frame allocations remain resident in CPython. Memory leaks
should be assessed from sustained growth over the daily history, not from
the cold-start baseline alone.

Thresholds:
  INFO:    >= 4.5 GB
  WARN:    >= 5 GB, or >= 1 GB over recent post-prediction baseline
  CRITICAL: >= 6 GB, or >= 3 GB over recent post-prediction baseline

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
    # Recalibrated 2026-05-23 after real history showed normal post-prediction
    # RSS commonly around 2.8-3.6 GB while cold sleeping RSS is about 140 MB.
    # Absolute 1 GB / 3 GB thresholds mixed those two operating states and made
    # normal post-prediction residency look actionable. Keep a high absolute
    # floor, and use recent post-prediction samples (not cold samples) for
    # growth detection.
    "info_mb": 4608,
    "warn_mb": 5120,
    "critical_mb": 6144,
    "post_prediction_floor_mb": 1024,
    "baseline_ceiling_mb": 5120,
    "warn_delta_mb": 1024,
    "critical_delta_mb": 3072,
    "baseline_days": 14,
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


def _row_date(row: dict) -> date | None:
    try:
        return date.fromisoformat(row["date"])
    except (KeyError, ValueError, TypeError):
        return None


def _daily_latest_rows(rows: list[dict]) -> list[dict]:
    """Collapse repeated same-day samples to the latest row for that date."""
    by_date: dict[date, dict] = {}
    for row in rows:
        row_date = _row_date(row)
        if row_date is None:
            continue
        by_date[row_date] = row
    return [by_date[row_date] for row_date in sorted(by_date)]


def _post_prediction_baseline(
    rows: list[dict],
    *,
    today: date,
    floor_mb: float,
    ceiling_mb: float,
) -> float | None:
    """Median recent post-prediction RSS, excluding today's sample and spikes."""
    samples: list[float] = []
    for row in _daily_latest_rows(rows):
        row_date = _row_date(row)
        if row_date is None or row_date >= today:
            continue
        try:
            rss_mb = float(row["rss_mb"])
        except (KeyError, TypeError, ValueError):
            continue
        if floor_mb <= rss_mb < ceiling_mb:
            samples.append(rss_mb)
    return median(samples) if samples else None


def _weekly_digest_alert(rows: list[dict]) -> Alert | None:
    """Build the Tuesday-EOD INFO digest alert from recent history rows.

    Reports n unique days, median over last 14d, latest, and 7d trend
    (median of last 7d minus median of preceding 7d, or N/A if not enough
    data). The append-only history can contain multiple same-day samples from
    manual checks, so callers should pass rows collapsed by _daily_latest_rows.
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
    if history_path is not None and today is None:
        today = date.today()

    baseline_mb: float | None = None
    if history_path is not None and today is not None:
        recent_rows = _read_recent_history(
            history_path,
            today,
            days=int(t.get("baseline_days", 14)),
        )
        baseline_mb = _post_prediction_baseline(
            recent_rows,
            today=today,
            floor_mb=float(t["post_prediction_floor_mb"]),
            ceiling_mb=float(t["baseline_ceiling_mb"]),
        )

    # Threshold-based alert (existing behavior)
    level: str | None = None
    trigger = ""
    delta_mb: float | None = None
    if baseline_mb is not None:
        delta_mb = rss_mb - baseline_mb

    critical_delta_threshold_mb = (
        baseline_mb + float(t["critical_delta_mb"])
        if baseline_mb is not None
        else None
    )
    warn_delta_threshold_mb = (
        baseline_mb + float(t["warn_delta_mb"])
        if baseline_mb is not None
        else None
    )

    if rss_mb >= t["critical_mb"]:
        level = "CRITICAL"
        trigger = f"absolute critical threshold {t['critical_mb']} MB"
    elif critical_delta_threshold_mb is not None and rss_mb >= critical_delta_threshold_mb:
        level = "CRITICAL"
        trigger = (
            f"delta threshold {critical_delta_threshold_mb:.1f} MB "
            f"(baseline {baseline_mb:.1f} MB + {t['critical_delta_mb']} MB)"
        )
    elif rss_mb >= t["warn_mb"]:
        level = "WARN"
        trigger = f"absolute warn threshold {t['warn_mb']} MB"
    elif warn_delta_threshold_mb is not None and rss_mb >= warn_delta_threshold_mb:
        level = "WARN"
        trigger = (
            f"delta threshold {warn_delta_threshold_mb:.1f} MB "
            f"(baseline {baseline_mb:.1f} MB + {t['warn_delta_mb']} MB)"
        )
    elif rss_mb >= t["info_mb"]:
        level = "INFO"
        trigger = f"absolute info threshold {t['info_mb']} MB"

    if level is not None:
        baseline_msg = (
            f"; recent post-prediction baseline={baseline_mb:.1f} MB "
            f"delta={delta_mb:+.1f} MB"
            if baseline_mb is not None and delta_mb is not None
            else "; recent post-prediction baseline unavailable"
        )
        alerts.append(Alert(
            level=level,
            source=SOURCE,
            message=(
                f"scheduler RSS {rss_mb:.1f} MB (pid={pid}); "
                f"thresholds info={t['info_mb']} MB warn_floor={t['warn_mb']} MB "
                f"critical={t['critical_mb']} MB "
                f"baseline_floor={t['post_prediction_floor_mb']} MB "
                f"baseline_ceiling={t['baseline_ceiling_mb']} MB "
                f"warn_delta={t['warn_delta_mb']} MB "
                f"critical_delta={t['critical_delta_mb']} MB"
                f"{baseline_msg}; trigger={trigger}"
            ),
        ))

    # History append + Tuesday digest (item #5)
    if history_path is not None:
        _append_history(history_path, today, rss_mb)
        # weekday(): Mon=0, Tue=1, ...
        if today.weekday() == 1:
            rows = _daily_latest_rows(_read_recent_history(history_path, today, days=14))
            digest = _weekly_digest_alert(rows)
            if digest is not None:
                alerts.append(digest)

    return alerts
