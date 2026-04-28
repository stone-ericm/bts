"""Calibration drift checking on production picks.

Two layers:
  - compute_drift_metrics(): I/O on picks_dir, returns DriftMetrics
  - evaluate_drift():       pure function (metrics, thresholds) -> list[Alert]

Thresholds are data-grounded against 28 days of 2026-04 picks (n=28):
  Top-1 P(game hit) observed: mean=0.7508 std=0.0273 min=0.7022 max=0.7918
  Single-day floor: 0.70  (= mean - 2σ; never observed historically)
  Sustained floor:  0.71  (= mean - 1.5σ; for 3+ consecutive days)
  Drift vs 14d baseline: 0.02 INFO / 0.04 WARN / 0.06 CRITICAL

The MDP threshold sweep (2026-04-27) showed strategy collapses at k≤0.90
(probabilities scaled to 90% of true). This alerting is the early warning
for drift toward that cliff edge — it observes only, never modifies picks.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

from bts.health.alert import Alert, dispatch_dm_for_critical, log_alerts

log = logging.getLogger(__name__)

SOURCE = "calibration_drift"


@dataclass(frozen=True)
class DriftMetrics:
    daily_top1: dict[str, float]            # date_iso -> top-1 P(game hit)
    daily_dd: dict[str, float]              # date_iso -> DD P (only days with DD)
    rolling_7d_mean: float | None           # mean of last 7 days' top-1 P
    rolling_14d_mean: float | None          # mean of last 14 days' top-1 P (baseline)
    drift: float | None                     # rolling_7d_mean - rolling_14d_mean


DEFAULT_THRESHOLDS: dict = {
    "single_day_floor": 0.70,
    "sustained_floor": 0.71,
    "sustained_days": 3,
    "drift_info": 0.02,
    "drift_warn": 0.04,
    "drift_critical": 0.06,
}


def compute_drift_metrics(
    picks_dir: Path,
    today: date | None = None,
    lookback_days: int = 30,
) -> DriftMetrics:
    """Read picks_dir/{YYYY-MM-DD}.json files, extract top-1 + DD P, compute rolling means.

    Skips *.shadow.json files. Skips files with corrupt JSON. Skips files
    outside the lookback window. `today` defaults to date.today() when None.
    """
    if today is None:
        today = date.today()
    cutoff = today - timedelta(days=lookback_days)

    daily_top1: dict[str, float] = {}
    daily_dd: dict[str, float] = {}

    for p in sorted(picks_dir.glob("*.json")):
        # Skip shadow / non-pick files. Pattern: YYYY-MM-DD.json (no extra dots).
        stem_parts = p.stem.split(".")
        if len(stem_parts) != 1:
            continue
        try:
            file_date = date.fromisoformat(p.stem)
        except ValueError:
            continue
        if file_date < cutoff or file_date > today:
            continue
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        pick = data.get("pick") or {}
        top1 = pick.get("p_game_hit")
        if top1 is None:
            continue
        date_iso = data.get("date") or p.stem
        daily_top1[date_iso] = float(top1)
        dd = data.get("double_down") or {}
        dd_p = dd.get("p_game_hit") if dd else None
        if dd_p is not None:
            daily_dd[date_iso] = float(dd_p)

    # Rolling means: take the most recent N days' values
    sorted_dates = sorted(daily_top1.keys())
    last_7 = sorted_dates[-7:]
    last_14 = sorted_dates[-14:]
    rolling_7d_mean = mean(daily_top1[d] for d in last_7) if len(last_7) >= 7 else None
    rolling_14d_mean = mean(daily_top1[d] for d in last_14) if len(last_14) >= 14 else None
    drift = (
        rolling_7d_mean - rolling_14d_mean
        if (rolling_7d_mean is not None and rolling_14d_mean is not None)
        else None
    )
    return DriftMetrics(
        daily_top1=daily_top1,
        daily_dd=daily_dd,
        rolling_7d_mean=rolling_7d_mean,
        rolling_14d_mean=rolling_14d_mean,
        drift=drift,
    )


def evaluate_drift(
    metrics: DriftMetrics,
    thresholds: dict | None = None,
) -> list[Alert]:
    """Return list of Alerts triggered by the given metrics.

    Pure function — no I/O, no state. The same metrics + thresholds always
    return the same alert list.
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    alerts: list[Alert] = []

    sorted_dates = sorted(metrics.daily_top1.keys())

    # 1) Single-day floor: latest day below single_day_floor → WARN
    if sorted_dates:
        latest = sorted_dates[-1]
        latest_p = metrics.daily_top1[latest]
        if latest_p < t["single_day_floor"]:
            alerts.append(Alert(
                level="WARN",
                source=SOURCE,
                message=f"top-1 P on {latest} = {latest_p:.4f} below floor {t['single_day_floor']:.2f}",
            ))

    # 2) Sustained low: last sustained_days all below sustained_floor → WARN
    n_days = int(t["sustained_days"])
    if len(sorted_dates) >= n_days:
        recent = sorted_dates[-n_days:]
        recent_ps = [metrics.daily_top1[d] for d in recent]
        if all(p < t["sustained_floor"] for p in recent_ps):
            alerts.append(Alert(
                level="WARN",
                source=SOURCE,
                message=(
                    f"{n_days} consecutive days top-1 P < {t['sustained_floor']:.2f} "
                    f"({recent[0]}..{recent[-1]}: {[round(p, 3) for p in recent_ps]})"
                ),
            ))

    # 3) Drift vs 14d baseline: only when both rolling means are available
    if metrics.drift is not None and metrics.rolling_7d_mean is not None and metrics.rolling_14d_mean is not None:
        drop = -metrics.drift  # positive number = drop
        if drop >= t["drift_critical"]:
            level = "CRITICAL"
        elif drop >= t["drift_warn"]:
            level = "WARN"
        elif drop >= t["drift_info"]:
            level = "INFO"
        else:
            level = None
        if level is not None:
            alerts.append(Alert(
                level=level,
                source=SOURCE,
                message=(
                    f"drift {drop:+.4f}: "
                    f"7d mean {metrics.rolling_7d_mean:.4f} vs 14d baseline {metrics.rolling_14d_mean:.4f}"
                ),
            ))

    return alerts


def check(
    picks_dir: Path,
    today: date | None = None,
    thresholds: dict | None = None,
) -> list[Alert]:
    """Top-level entrypoint for the runner. Returns alerts (no I/O beyond reading)."""
    try:
        metrics = compute_drift_metrics(picks_dir, today=today)
        return evaluate_drift(metrics, thresholds=thresholds)
    except Exception as e:
        log.exception(f"calibration check failed to compute metrics: {e}")
        return []


def run_calibration_check(
    picks_dir: Path,
    dm_recipient: str | None,
    today: date | None = None,
    thresholds: dict | None = None,
) -> list[Alert]:
    """Backwards-compatible standalone entrypoint.

    Same flow as the runner does for any check: compute alerts, log them,
    dispatch DM on CRITICAL. Kept for any caller importing this directly.
    """
    alerts = check(picks_dir, today=today, thresholds=thresholds)
    log_alerts(alerts)
    dispatch_dm_for_critical(alerts, dm_recipient)
    return alerts
