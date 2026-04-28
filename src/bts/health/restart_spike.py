"""Tier 1: NRestarts spike check.

The systemd unit's NRestarts counter is cumulative since unit-load. A
sudden delta (≥3 restarts since yesterday's checkpoint) signals a
heartbeat-gap regression — the class of bug we shipped 5 fixes for in
2026-04-22/23.

State is kept in a small JSON file at picks_dir/.nrestarts_checkpoint.
On first run (no checkpoint), records baseline and emits no alert. On
each subsequent run, computes delta from the previous checkpoint and
alerts on threshold breach. Resets checkpoint on each non-alert run.
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
    """Returns CRITICAL alert if NRestarts grew ≥ spike_threshold since last checkpoint.

    `current_nrestarts` is read by the caller via systemctl. We just track
    the delta. Updates the checkpoint file unconditionally (so today's
    snapshot becomes tomorrow's baseline).
    """
    checkpoint_path = picks_dir / ".nrestarts_checkpoint"
    prior = None
    if checkpoint_path.exists():
        try:
            prior_data = json.loads(checkpoint_path.read_text())
            prior = int(prior_data.get("nrestarts"))
        except Exception:
            log.warning(f"could not parse {checkpoint_path}; treating as fresh baseline")

    alerts: list[Alert] = []
    if prior is not None:
        delta = current_nrestarts - prior
        if delta >= spike_threshold:
            alerts.append(Alert(
                level="CRITICAL",
                source=SOURCE,
                message=(
                    f"NRestarts spiked +{delta} since last checkpoint "
                    f"({prior} → {current_nrestarts}). Heartbeat-gap regression suspected."
                ),
            ))

    # Update checkpoint
    checkpoint_path.write_text(json.dumps({
        "nrestarts": int(current_nrestarts),
        "checkpointed_at": datetime.now(timezone.utc).isoformat(),
    }))
    return alerts
