"""Tier 2: predicted vs realized divergence.

Detects acute model degradation: the gap between mean predicted P(hit) and
mean realized hit rate — pooled over every graded SLOT (primary and, on
double-down days, the DD leg, each against its own p) — has widened
significantly in the recent 14 days versus the 28-day baseline.

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
    # date_iso -> {"slots": [(predicted, realized 0/1), ...], "result": str}
    # One entry per GRADED slot (primary and, on DD days, the DD leg).
    daily: dict[str, dict]
    rolling_14d_gap: float | None  # mean(predicted) - mean(realized), slots pooled over last 14 days
    baseline_28d_gap: float | None  # same over last 28 days
    drift: float | None  # 14d gap - 28d gap (positive = current more overconfident than baseline)


DEFAULT_THRESHOLDS = {
    # The drift statistic is a 14d window MINUS its overlapping 28d baseline:
    # at one obs/day its SE is ≈0.094 (NOT the standalone n=14 Bernoulli
    # SE≈0.13 the original comment used — round-2 review #9), so the old
    # 0.12 CRITICAL was ~1.3σ. DD days pool a second, same-day-correlated
    # slot, which doesn't buy the independence a lower bar would need.
    # WARN stays a cheap attention signal; CRITICAL is reserved for
    # catastrophic/pipeline-scale drift until a day-block bootstrap
    # recalibrates these (TODO, queued with the post-Aug window widening).
    "drift_info": 0.05,
    "drift_warn": 0.08,
    "drift_critical": 0.25,
    "min_days_14d": 10,  # require n≥10 in 14-day window for stat power
    "min_days_28d": 20,  # require n≥20 in 28-day baseline
}


def compute_metrics(picks_dir: Path, today: date | None = None,
                    lookback_days: int = 35) -> PredRealMetrics:
    """Read picks_dir, extract (predicted, realized) per GRADED SLOT per day.

    Per-slot attribution (2026-07-12 incident): the old day-level counting
    compared the PRIMARY's p_game_hit against the day result — but on
    double-down days the day result requires BOTH legs to hit, so a DD-dense
    stretch (what the MDP produces at streak 0) mechanically inflates the gap.
    Live decomposition that night: day-level drift +0.1737 "CRITICAL" while
    primary-only drift was +0.042; the real signal (DD legs 1-for-6) belongs
    to the DD slot's own p, not the primary's. Same attribution lesson as
    realized_calibration's 2026-05-01 fix: grade each delivered leg against
    its own probability, via slot_results when present.

    Legacy files without slot_results: a single-pick day's result IS the
    primary outcome and is kept; a DD day is excluded entirely — its "miss"
    is unattributable, and keeping only the attributable "hit" days would be
    outcome-dependent censoring (round-2 review #7). Void slots are never
    graded; days with no gradable slot are skipped entirely.
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
        result = data.get("result")
        if result not in ("hit", "miss"):
            continue
        dd = data.get("double_down") or None
        slot_results = data.get("slot_results") or {}
        slots: list[tuple[float, int]] = []

        def _grade(slot_key: str, slot_obj: dict | None) -> None:
            predicted = (slot_obj or {}).get("p_game_hit")
            if predicted is None:
                return
            outcome = slot_results.get(slot_key)
            if outcome == "void":
                return
            if outcome in ("hit", "miss"):
                slots.append((float(predicted), 1 if outcome == "hit" else 0))
            elif dd is None:
                # Legacy single-pick file: day result IS the primary outcome.
                slots.append((float(predicted), 1 if result == "hit" else 0))
            # Legacy DD file (no slot_results): excluded ENTIRELY. A day-level
            # "hit" is attributable (both legs hit) but a "miss" is not —
            # including only the attributable hits is outcome-dependent
            # censoring that inflates realized and can MASK degradation
            # (round-2 review #7).

        _grade("pick", data.get("pick") or {})
        _grade("double_down", dd)
        if not slots:
            continue
        daily[data.get("date") or p.stem] = {
            "slots": slots,
            "result": result,
        }

    sorted_dates = sorted(daily.keys())

    def gap_over(window: list[str]) -> float | None:
        if not window:
            return None
        preds = [s[0] for d in window for s in daily[d]["slots"]]
        reals = [s[1] for d in window for s in daily[d]["slots"]]
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
    # Stat-power gates measured over the actual windows, not the full lookback:
    # the 14d window needs >= min_days_14d, AND the 28d baseline needs
    # >= min_days_28d so it is a real baseline distinct from the 14d window
    # (otherwise early-season drift fires on overlapping, near-identical sets).
    sorted_dates = sorted(metrics.daily)
    n14 = len(sorted_dates[-14:])
    n28 = len(sorted_dates[-28:])
    if n14 < t["min_days_14d"] or n28 < t["min_days_28d"]:
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
            " (per-slot: primary + DD legs graded against their own p)"
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
