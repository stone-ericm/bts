"""Tier 2: predicted vs realized divergence.

Detects acute model degradation: the gap between mean predicted P(hit on top-1)
and mean realized hit rate has widened significantly in the recent 14 days
versus the 28-day baseline.

Historical context: realized-picks analysis 2026-04-25 (n=48) showed a chronic
~7pp overconfidence (predicted 0.74, realized 0.667) — that's PRESENT in
production today. This alert is for ACUTE degradation on top of the chronic
gap, not the chronic gap itself. Drift thresholds compare 14d gap to 28d
baseline so chronic miscalibration cancels out.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

from bts.health.alert import Alert

log = logging.getLogger(__name__)

SOURCE = "predicted_vs_realized"


@dataclass(frozen=True)
class PredRealMetrics:
    daily: dict[str, dict]  # date_iso -> {"predicted": float, "realized": int (0/1), "result": str}
    rolling_14d_gap: float | None  # mean(predicted) - mean(realized) over last 14 days with results
    baseline_28d_gap: float | None  # same over last 28 days
    drift: float | None  # 14d gap - 28d gap (positive = current more overconfident than baseline)


DEFAULT_THRESHOLDS = {
    # n=14 binomial SE ≈ 0.13 at p~0.5; thresholds set above sampling
    # noise so a CRITICAL requires real signal, not chance variance.
    # Tune down once we have larger windows (post Aug 2026 with n≥60d).
    "drift_info": 0.05,
    "drift_warn": 0.08,
    "drift_critical": 0.12,
    "min_days_14d": 10,  # require n≥10 in 14-day window for stat power
    "min_days_28d": 20,  # require n≥20 in 28-day baseline
}


def compute_metrics(picks_dir: Path, today: date | None = None,
                    lookback_days: int = 35) -> PredRealMetrics:
    """Read picks_dir, extract (predicted, realized) per resolved day.

    Only counts days where data["result"] in {"hit","miss"} (resolved games).
    Skips shadow files, unresolved games (result=null), and days with no pick.
    """
    if today is None:
        today = date.today()
    cutoff = today - timedelta(days=lookback_days)

    daily: dict[str, dict] = {}
    for p in sorted(picks_dir.glob("*.json")):
        if "." in p.stem:  # skip shadow.json etc.
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
        predicted = pick.get("p_game_hit")
        result = data.get("result")
        if predicted is None or result not in ("hit", "miss"):
            continue
        daily[data.get("date") or p.stem] = {
            "predicted": float(predicted),
            "realized": 1 if result == "hit" else 0,
            "result": result,
        }

    sorted_dates = sorted(daily.keys())

    def gap_over(window: list[str]) -> float | None:
        if not window:
            return None
        preds = [daily[d]["predicted"] for d in window]
        reals = [daily[d]["realized"] for d in window]
        return mean(preds) - mean(reals)

    last_14 = sorted_dates[-14:]
    last_28 = sorted_dates[-28:]
    rolling_14d_gap = gap_over(last_14) if len(last_14) >= 1 else None
    baseline_28d_gap = gap_over(last_28) if len(last_28) >= 1 else None
    drift = (
        rolling_14d_gap - baseline_28d_gap
        if (rolling_14d_gap is not None and baseline_28d_gap is not None)
        else None
    )
    return PredRealMetrics(
        daily=daily,
        rolling_14d_gap=rolling_14d_gap,
        baseline_28d_gap=baseline_28d_gap,
        drift=drift,
    )


def evaluate(metrics: PredRealMetrics, thresholds: dict | None = None) -> list[Alert]:
    """Pure function: return alerts for acute drift.

    Drift is positive when 14d gap > 28d baseline gap (i.e., model has gotten
    MORE overconfident recently). Negative drift (less overconfident than
    baseline) doesn't alert — that's improvement.
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    alerts: list[Alert] = []
    if metrics.drift is None:
        return alerts
    n14 = sum(1 for _ in [d for d in metrics.daily])
    if n14 < t["min_days_14d"]:
        return alerts
    drift = metrics.drift
    if drift < t["drift_info"]:
        return alerts
    if drift >= t["drift_critical"]:
        level = "CRITICAL"
    elif drift >= t["drift_warn"]:
        level = "WARN"
    else:
        level = "INFO"
    alerts.append(Alert(
        level=level,
        source=SOURCE,
        message=(
            f"acute predicted-vs-realized drift +{drift:.4f}: "
            f"14d gap {metrics.rolling_14d_gap:+.4f} vs 28d baseline {metrics.baseline_28d_gap:+.4f}"
        ),
    ))
    return alerts


def check(picks_dir: Path, today: date | None = None,
          thresholds: dict | None = None) -> list[Alert]:
    """Top-level entrypoint."""
    try:
        m = compute_metrics(picks_dir, today=today)
        return evaluate(m, thresholds=thresholds)
    except Exception as e:
        log.exception(f"predicted_vs_realized check failed: {e}")
        return []
