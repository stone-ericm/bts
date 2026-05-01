"""Tier 2: realized calibration check (75-80% predicted-P bucket overconfidence).

The complement to predicted_vs_realized.py. That check detects DRIFT in the
gap between predicted and realized P over time. This check detects the
ABSOLUTE LEVEL of miscalibration in the 75-80% bucket — where most prod
picks land — vs realized hit rates.

**Attribution fix 2026-05-01**: previously used streak ``result`` as proxy
for primary-pick hit. That's biased on double-down days because streak
"hit" requires BOTH picks to hit, so a DD pick that did hit gets attributed
as "miss" whenever the primary missed. The fix: when ``data_dir`` is
provided, look up the actual per-pick day-hit from the season's PA frame.
The biased path remains as a safety fallback when pa frame isn't available.

The corrected attribution shows real over-confidence is ~+6.6pp overall
and ~+12.3pp in the [0.75, 0.80) bucket — **less alarming than the
proxy-based "+14pp" finding from 2026-04-29**, which was inflated by the
DD attribution bias. Thresholds are recalibrated accordingly.

Severity ladder (75-80% bucket only; other buckets ignored):
  predicted - realized < 8pp:     no alert (well-calibrated under proper attribution)
  >= 8pp:                         INFO  (worth observing)
  >= 15pp:                        WARN  (significantly overconfident)
  >= 25pp:                        CRITICAL (true distribution-shift signal)

Lookback window: last 30 days. Minimum bucket count: 5 picks.
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
    "info_pp": 8.0,
    "warn_pp": 15.0,
    "critical_pp": 25.0,
    "lookback_days": 30,
    "min_bucket_n": 5,
    "bucket_low": 0.75,
    "bucket_high": 0.80,
}


def _build_day_hit_lookup(data_dir: Path, today: date, lookback_days: int) -> dict:
    """Build (batter_id, date) -> day_had_any_hit lookup from current season's PA frame.

    Returns empty dict if no parquet exists; caller falls back to streak-result proxy.
    """
    try:
        import pandas as pd
    except ImportError:
        return {}
    cutoff = today - timedelta(days=lookback_days)
    candidates = [data_dir / f"pa_{y}.parquet" for y in (today.year, today.year - 1)]
    parts = []
    for p in candidates:
        if p.exists():
            try:
                parts.append(pd.read_parquet(p, columns=["batter_id", "date", "is_hit"]))
            except Exception as e:
                log.warning(f"failed to load {p} for calibration attribution: {e}")
    if not parts:
        return {}
    pa_df = pd.concat(parts, ignore_index=True)
    pa_df["date"] = pd.to_datetime(pa_df["date"]).dt.date
    pa_df = pa_df[(pa_df["date"] >= cutoff) & (pa_df["date"] <= today)]
    daily = (
        pa_df.groupby(["batter_id", "date"])["is_hit"]
        .max()
        .reset_index()
    )
    return {(int(r["batter_id"]), r["date"]): int(r["is_hit"]) for _, r in daily.iterrows()}


def check(
    picks_dir: Path,
    today: date | None = None,
    thresholds: dict | None = None,
    data_dir: Path | None = None,
) -> list[Alert]:
    """Returns INFO/WARN/CRITICAL alert when 75-80% bucket is overconfident.

    When ``data_dir`` is provided AND the season parquet exists, uses true
    per-pick day-hit attribution. Otherwise falls back to streak-result
    proxy (biased on DD days; preserved for backward compatibility).
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if today is None:
        today = date.today()
    if not picks_dir.exists():
        return []

    # Build proper attribution lookup if PA frame is available.
    day_hit_lookup = {}
    if data_dir is not None:
        day_hit_lookup = _build_day_hit_lookup(data_dir, today, t["lookback_days"])
    using_pa_attribution = bool(day_hit_lookup)

    cutoff = today - timedelta(days=t["lookback_days"])
    in_bucket: list[tuple[float, int]] = []
    try:
        files = sorted(picks_dir.glob("*.json"))
    except OSError as e:
        log.warning(f"could not list {picks_dir}: {e}")
        return []
    for f in files:
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
        # Iterate primary + double_down so both picks contribute to the bucket
        # under the proper PA-frame attribution path. The biased fallback path
        # uses primary-only because streak result misattributes DD picks.
        for slot_key in ("pick", "double_down"):
            slot = body.get(slot_key) or {}
            p = slot.get("p_game_hit")
            if p is None:
                continue
            if not (t["bucket_low"] <= p < t["bucket_high"]):
                continue
            if using_pa_attribution:
                bid = slot.get("batter_id")
                if bid is None:
                    continue  # PA-frame join needs batter_id
                day_hit = day_hit_lookup.get((int(bid), pick_date))
                if day_hit is None:
                    continue  # late data; skip rather than guess
                in_bucket.append((float(p), int(day_hit)))
            else:
                # Biased fallback: streak result. Only trustworthy for primary picks
                # (and only on primary-only days at that — but DD-presence isn't
                # checked here; this is the legacy path before the PA-frame fix).
                if slot_key == "pick":
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

    attribution = "pa-frame" if using_pa_attribution else "streak-proxy (biased on DD days)"
    msg = (
        f"75-80% bucket overconfident by {overconf_pp:+.1f}pp over last "
        f"{t['lookback_days']}d (n={len(in_bucket)}, predicted {mean_predicted:.3f}, "
        f"realized {realized_rate:.3f}, attribution={attribution})"
    )
    if level == "CRITICAL":
        msg += ". True distribution-shift signal — investigate model staleness vs new-regime."
    return [Alert(level=level, source=SOURCE, message=msg)]
