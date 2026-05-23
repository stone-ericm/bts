"""Tier 2: double-down pair shortfall and residual pair drift.

DD strategy consumes a model-pair probability (`p1 × pdd`) and also has a
separate pair-dependence assumption. These are different failure modes:

1. Model-pair shortfall: realized P(both hit) is below `p1 × pdd`. This can be
   caused by ordinary marginal overconfidence in either slot.
2. Residual pair shortfall: realized P(both hit) is below the product of the
   empirical primary/DD marginal hit rates. This is the part that can justify a
   pair-dependence/correlation investigation.

The first signal stays visible, but it should not be labeled as pair
correlation by itself.
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

SOURCE = "dd_pair_realized_shortfall"
RESIDUAL_SOURCE = "dd_pair_residual_corr"


@dataclass(frozen=True)
class CorrMetrics:
    pair_days: list[dict]  # each: {"date", "p1", "pdd", "predicted", "realized", ...}
    rolling_14d_gap: float | None  # mean(predicted_pair) - mean(realized_pair) recent 14d
    baseline_28d_gap: float | None
    drift: float | None  # 14d gap - 28d gap
    rolling_14d_residual_gap: float | None = None
    baseline_28d_residual_gap: float | None = None
    residual_drift: float | None = None
    n_residual_14d: int = 0


DEFAULT_THRESHOLDS = {
    "drift_info": 0.05,
    "drift_warn": 0.10,
    "drift_critical": 0.15,
    "residual_info": 0.05,
    "residual_warn": 0.10,
    "residual_critical": 0.15,
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
        slot_results = data.get("slot_results") or {}
        # Need both picks resolved. Treat pair_realized as 1 only when result=="hit"
        # (BTS DD scoring: streak only advances if BOTH picks hit).
        if (
            p1 is None
            or pdd is None
            or result not in ("hit", "miss")
            or slot_results.get("pick") == "void"
            or slot_results.get("double_down") == "void"
        ):
            continue
        pick_hit = None
        dd_hit = None
        if slot_results.get("pick") in ("hit", "miss"):
            pick_hit = 1 if slot_results["pick"] == "hit" else 0
        if slot_results.get("double_down") in ("hit", "miss"):
            dd_hit = 1 if slot_results["double_down"] == "hit" else 0
        if result == "hit" and (pick_hit is None or dd_hit is None):
            # Old files without slot_results can still prove both slots hit.
            pick_hit = 1
            dd_hit = 1

        pair_days.append({
            "date": data.get("date") or p.stem,
            "p1": float(p1),
            "pdd": float(pdd),
            "predicted": float(p1) * float(pdd),  # naive independence
            "realized": 1 if result == "hit" else 0,
            "pick_hit": pick_hit,
            "dd_hit": dd_hit,
        })

    pair_days.sort(key=lambda r: r["date"])

    def gap_over(window):
        if not window:
            return None
        return mean(d["predicted"] for d in window) - mean(d["realized"] for d in window)

    def residual_gap_over(window):
        rows = [
            d for d in window
            if d.get("pick_hit") is not None and d.get("dd_hit") is not None
        ]
        if not rows:
            return None
        primary_rate = mean(d["pick_hit"] for d in rows)
        dd_rate = mean(d["dd_hit"] for d in rows)
        realized_pair_rate = mean(d["realized"] for d in rows)
        return (primary_rate * dd_rate) - realized_pair_rate

    last_14 = pair_days[-14:]
    last_28 = pair_days[-28:]
    rolling_14d_gap = gap_over(last_14) if len(last_14) >= 1 else None
    baseline_28d_gap = gap_over(last_28) if len(last_28) >= 1 else None
    drift = (
        rolling_14d_gap - baseline_28d_gap
        if (rolling_14d_gap is not None and baseline_28d_gap is not None)
        else None
    )
    rolling_14d_residual_gap = residual_gap_over(last_14)
    baseline_28d_residual_gap = residual_gap_over(last_28)
    residual_drift = (
        rolling_14d_residual_gap - baseline_28d_residual_gap
        if (
            rolling_14d_residual_gap is not None
            and baseline_28d_residual_gap is not None
        )
        else None
    )
    n_residual_14d = sum(
        1
        for d in last_14
        if d.get("pick_hit") is not None and d.get("dd_hit") is not None
    )
    return CorrMetrics(pair_days=pair_days, rolling_14d_gap=rolling_14d_gap,
                       baseline_28d_gap=baseline_28d_gap, drift=drift,
                       rolling_14d_residual_gap=rolling_14d_residual_gap,
                       baseline_28d_residual_gap=baseline_28d_residual_gap,
                       residual_drift=residual_drift,
                       n_residual_14d=n_residual_14d)


def evaluate(metrics: CorrMetrics, thresholds: dict | None = None) -> list[Alert]:
    """Drift > 0 means recent realized pairs are further below model pairs
    than baseline. Residual gap is checked separately against empirical
    marginal rates.
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
        pass
    else:
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
                f"DD model-pair shortfall drift +{drift:.4f}: "
                f"14d model-vs-realized shortfall {metrics.rolling_14d_gap:+.4f} "
                f"vs 28d baseline {metrics.baseline_28d_gap:+.4f}; "
                "check marginal calibration before pair-correlation attribution"
            ),
        ))

    residual_gap = metrics.rolling_14d_residual_gap
    if (
        residual_gap is not None
        and metrics.n_residual_14d >= t["min_days_14d"]
        and residual_gap >= t["residual_info"]
    ):
        residual_drift = (
            f"{metrics.residual_drift:+.4f}"
            if metrics.residual_drift is not None
            else "n/a"
        )
        if residual_gap >= t["residual_critical"]:
            level = "CRITICAL"
        elif residual_gap >= t["residual_warn"]:
            level = "WARN"
        else:
            level = "INFO"
        alerts.append(Alert(
            level=level,
            source=RESIDUAL_SOURCE,
            message=(
                f"DD residual pair shortfall {residual_gap:+.4f}: "
                "observed both-hit rate below empirical marginal product "
                f"(14d n={metrics.n_residual_14d}, "
                f"residual drift {residual_drift})"
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
