"""Tier 1: NRestarts spike check.

The systemd unit's NRestarts counter is cumulative since unit-load. A
sudden delta (≥3 restarts since yesterday's checkpoint) signals a
restart loop or repeated service failure. Prior incidents included
heartbeat-gap regressions, but OOM kills, watchdog exits, and crash loops
can produce the same symptom.

State is kept in a small JSON file at picks_dir/.nrestarts_checkpoint.
On first run (no checkpoint), records baseline and emits no alert. On
each subsequent run, computes delta from the previous checkpoint and
alerts on threshold breach.

The checkpoint is DAY-ANCHORED (2026-07-12 incident): it only advances on
the first run of a new day. The old advance-every-run behavior aliased a
Restart=always loop into invisibility — each ~48s EOD re-walk re-ran this
check and moved the baseline +1 at a time, so a 47-restart storm never
summed past the +3 threshold. Anchoring to the day's first observation
makes same-day churn accumulate against a fixed baseline.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

from bts.health.alert import Alert

log = logging.getLogger(__name__)

SOURCE = "restart_spike"


def check(
    picks_dir: Path,
    current_nrestarts: int,
    spike_threshold: int = 3,
    today: date | None = None,
) -> list[Alert]:
    """Returns CRITICAL alert if NRestarts grew ≥ spike_threshold since the
    day-anchored checkpoint.

    `current_nrestarts` is read by the caller via systemctl. We just track
    the delta. The checkpoint advances only on the first run of a new day —
    same-day re-runs keep the anchor so churn accumulates (a legacy
    checkpoint without a `day` field is treated as a prior day: compared
    against, then rewritten with today's anchor).
    """
    if today is None:
        today = date.today()
    checkpoint_path = picks_dir / ".nrestarts_checkpoint"
    prior = None
    prior_day = None
    if checkpoint_path.exists():
        try:
            prior_data = json.loads(checkpoint_path.read_text())
            prior = int(prior_data.get("nrestarts"))
            prior_day = prior_data.get("day")
        except Exception:
            log.warning(f"could not parse {checkpoint_path}; treating as fresh baseline")

    alerts: list[Alert] = []
    if prior is not None:
        delta = current_nrestarts - prior
        # Budget PLANNED restarts across multi-day gaps (round-2 review #4):
        # the daily lifecycle exits once per day by design (idle → return →
        # Restart=always), and no-games days never run this check — so a
        # 4-day break accumulates +4 planned restarts against a frozen
        # anchor. Allow one per elapsed day beyond the first; a real loop
        # (dozens per day) still clears the raised bar easily.
        days_gap = 0
        if prior_day:
            try:
                days_gap = max(0, (today - date.fromisoformat(prior_day)).days)
            except ValueError:
                days_gap = 0
        effective_threshold = spike_threshold + max(0, days_gap - 1)
        if delta >= effective_threshold:
            # Bar in UNPLANNED terms is constant (= spike_threshold - 1) at
            # every gap: the base threshold already budgets the single
            # planned exit-restart of a normal consecutive day, and each
            # extra elapsed day adds exactly one more planned restart.
            budget_note = (
                f" (threshold {effective_threshold} over a {days_gap}d gap: "
                f"base {spike_threshold} + {days_gap - 1} extra planned "
                "exit-restarts; the base already budgets one)"
            ) if days_gap > 1 else ""
            alerts.append(Alert(
                level="CRITICAL",
                source=SOURCE,
                message=(
                    f"NRestarts spiked +{delta} since last checkpoint "
                    f"({prior} → {current_nrestarts}). Scheduler restart loop "
                    "suspected; inspect journal/OOM/watchdog/crash evidence."
                    + budget_note
                ),
            ))

    # Advance the checkpoint only across a day boundary (or on first run):
    # same-day runs must diff against the day's first observation, not
    # each other (the aliasing that hid the 2026-07-12 loop).
    if prior is None or prior_day != today.isoformat():
        checkpoint_path.write_text(json.dumps({
            "nrestarts": int(current_nrestarts),
            "day": today.isoformat(),
            "checkpointed_at": datetime.now(timezone.utc).isoformat(),
        }))
    return alerts
