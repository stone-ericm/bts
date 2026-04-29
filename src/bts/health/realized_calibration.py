"""Tier 2: realized calibration check (75-80% predicted-P bucket overconfidence).

The complement to predicted_vs_realized.py. That check detects DRIFT in the
gap between predicted and realized P over time. This check detects the
ABSOLUTE LEVEL of miscalibration in the 75-80% bucket — where most prod
picks land — vs realized hit rates.

F's analysis on 2026-04-29 found +14pp overconfidence in this bucket sitting
unaddressed for weeks because no one ran the analysis manually until then.
This check converts that finding into an automatic alert.

Severity ladder (75-80% bucket only; other buckets ignored):
  predicted - realized < 5pp:    no alert (well-calibrated)
  >= 5pp:                        INFO  (overconfident; worth observing)
  >= 10pp:                       WARN  (significantly overconfident)
  >= 15pp:                       CRITICAL (consider seed switch / pooled mode)

Lookback window: last 30 days. Minimum bucket count: 5 picks (avoids
high-variance estimates triggering false alerts).
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

from bts.health.alert import Alert

log = logging.getLogger(__name__)

SOURCE = "realized_calibration"

DEFAULT_THRESHOLDS = {
    "info_pp": 5.0,
    "warn_pp": 10.0,
    "critical_pp": 15.0,
    "lookback_days": 30,
    "min_bucket_n": 5,
    "bucket_low": 0.75,
    "bucket_high": 0.80,
}


def check(
    picks_dir: Path,
    today: date | None = None,
    thresholds: dict | None = None,
) -> list[Alert]:
    """Returns INFO/WARN/CRITICAL alert when 75-80% bucket is overconfident."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if today is None:
        today = date.today()
    if not picks_dir.exists():
        return []

    cutoff = today - timedelta(days=t["lookback_days"])
    in_bucket: list[tuple[float, int]] = []  # (predicted_p, actual_hit)
    try:
        files = sorted(picks_dir.glob("*.json"))
    except OSError as e:
        log.warning(f"could not list {picks_dir}: {e}")
        return []
    for f in files:
        # Filenames are typically YYYY-MM-DD.json or YYYY-MM-DD.shadow.json
        if ".shadow." in f.name or "scheduler" in f.name or "streak" in f.name:
            continue
        try:
            body = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        try:
            pick_date = date.fromisoformat(body.get("date", ""))
        except (ValueError, TypeError):
            continue
        if pick_date < cutoff or pick_date > today:
            continue
        result = body.get("result")
        if result not in ("hit", "miss"):
            continue
        pick = body.get("pick") or {}
        p = pick.get("p_game_hit")
        if p is None:
            continue
        if t["bucket_low"] <= p < t["bucket_high"]:
            in_bucket.append((float(p), 1 if result == "hit" else 0))

    if len(in_bucket) < t["min_bucket_n"]:
        return []

    mean_predicted = sum(p for p, _ in in_bucket) / len(in_bucket)
    realized_rate = sum(h for _, h in in_bucket) / len(in_bucket)
    overconf_pp = (mean_predicted - realized_rate) * 100

    if overconf_pp < t["info_pp"]:
        return []
    if overconf_pp >= t["critical_pp"]:
        level = "CRITICAL"
    elif overconf_pp >= t["warn_pp"]:
        level = "WARN"
    else:
        level = "INFO"

    msg = (
        f"75-80% bucket overconfident by {overconf_pp:+.1f}pp over last "
        f"{t['lookback_days']}d (n={len(in_bucket)}, predicted {mean_predicted:.3f}, "
        f"realized {realized_rate:.3f})"
    )
    if level == "CRITICAL":
        msg += ". Consider switching production seed or enabling pooled prediction."
    return [Alert(level=level, source=SOURCE, message=msg)]
