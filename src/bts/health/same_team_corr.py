"""Tier 2: same-day pick correlation drift.

DD strategy assumes some independence between primary pick and DD. Empirical
P(both hit) is observed via the realized result of pick + DD pair. If the
realized P(both hit) deviates from the predicted naive-independence
P(p1 × pdd) by significant margin, the DD strategy assumption is breaking.

2026-04-25 baseline observation: 11/22 = 0.500 realized vs naive 0.548 expected.
That's ~5pp shortfall, attributed to positive same-day correlation between picks
(good lineup days lift both, bad days drag both). Acute drift here would mean
the correlation pattern is intensifying.

This is a pair-level realized check, separate from the top-1 calibration check.
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

SOURCE = "same_team_corr"


@dataclass(frozen=True)
class CorrMetrics:
    pair_days: list[dict]  # each: {"date", "p1", "pdd", "predicted", "realized"}
    rolling_14d_gap: float | None  # mean(predicted_pair) - mean(realized_pair) recent 14d
    baseline_28d_gap: float | None
    drift: float | None  # 14d gap - 28d gap


DEFAULT_THRESHOLDS = {
    "drift_info": 0.05,
    "drift_warn": 0.10,
    "drift_critical": 0.15,
    "min_days_14d": 8,
}


def compute_metrics(picks_dir: Path, today: date | None = None,
                    lookback_days: int = 35) -> CorrMetrics:
    """Read picks_dir, extract pair-level (predicted_pair, realized_pair) for resolved days."""
    if today is None:
        today = date.today()
    cutoff = today - timedelta(days=lookback_days)

    pair_days: list[dict] = []
    for p in sorted(picks_dir.glob("*.json")):
        if "." in p.stem:
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
        dd = data.get("double_down") or {}
        if not pick or not dd:
            continue  # only days with both picks
        p1 = pick.get("p_game_hit")
        pdd = dd.get("p_game_hit")
        result = data.get("result")
        # Need both picks resolved. Treat pair_realized as 1 only when result=="hit"
        # (BTS DD scoring: streak only advances if BOTH picks hit).
        if p1 is None or pdd is None or result not in ("hit", "miss"):
            continue
        pair_days.append({
            "date": data.get("date") or p.stem,
            "p1": float(p1),
            "pdd": float(pdd),
            "predicted": float(p1) * float(pdd),  # naive independence
            "realized": 1 if result == "hit" else 0,
        })

    pair_days.sort(key=lambda r: r["date"])

    def gap_over(window):
        if not window:
            return None
        return mean(d["predicted"] for d in window) - mean(d["realized"] for d in window)

    last_14 = pair_days[-14:]
    last_28 = pair_days[-28:]
    rolling_14d_gap = gap_over(last_14) if len(last_14) >= 1 else None
    baseline_28d_gap = gap_over(last_28) if len(last_28) >= 1 else None
    drift = (
        rolling_14d_gap - baseline_28d_gap
        if (rolling_14d_gap is not None and baseline_28d_gap is not None)
        else None
    )
    return CorrMetrics(pair_days=pair_days, rolling_14d_gap=rolling_14d_gap,
                       baseline_28d_gap=baseline_28d_gap, drift=drift)


def evaluate(metrics: CorrMetrics, thresholds: dict | None = None) -> list[Alert]:
    """Drift > 0 means current realization is FURTHER below naive prediction
    than baseline — correlation getting stronger. Drift < 0 means improvement.
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    alerts: list[Alert] = []
    if metrics.drift is None:
        return alerts
    n_recent = len(metrics.pair_days[-14:]) if metrics.pair_days else 0
    if n_recent < t["min_days_14d"]:
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
            f"DD pair-correlation drift +{drift:.4f}: "
            f"14d realization-shortfall {metrics.rolling_14d_gap:+.4f} "
            f"vs 28d baseline {metrics.baseline_28d_gap:+.4f}"
        ),
    ))
    return alerts


def check(picks_dir: Path, today: date | None = None,
          thresholds: dict | None = None) -> list[Alert]:
    try:
        m = compute_metrics(picks_dir, today=today)
        return evaluate(m, thresholds=thresholds)
    except Exception as e:
        log.exception(f"same_team_corr check failed: {e}")
        return []
