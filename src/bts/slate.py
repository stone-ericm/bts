"""Persist the full ranked daily slate for realized slate-level analysis.

Until 2026-06-11 the candidate-level predictions were computed every cycle
and discarded — only pick/double_down/runner_up survived. That made realized
slate metrics (rolling AUC, sub-top-1 ranking quality, live feature
attribution) impossible to compute after the fact: the M3 serving-staleness
closeout (docs/audit/2026-06-11-m3-serving-staleness.md) could not quantify
bpm's realized live contribution for exactly this reason.

One JSON file per date under {picks_dir}/slates/. Last write wins: re-runs
within a day overwrite with the slate that produced the final pick, which is
the slate of record. Persistence is observability — it must NEVER break the
pick path, so save_slate swallows and logs every failure.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bts.util import atomic_write_text

log = logging.getLogger(__name__)

SCHEMA_VERSION = "bts_slate_v1"

# Persisted per candidate when present in the predictions frame. Feature
# values are deliberately excluded: they are reconstructable from the PA
# parquets (validated to 5.55e-17 by scripts/replay_m3_serving_parity.py),
# while the model outputs below are not.
ROW_COLUMNS = [
    "batter_id", "batter_name", "team", "game_pk", "lineup",
    "pitcher_id", "pitcher_name",
    "p_game_hit", "p_game_blend", "p_hit_vs_starter", "p_hit_vs_reliever",
    "est_pas", "flags", "projected",
]


def save_slate(
    predictions: pd.DataFrame | None,
    date: str,
    picks_dir: Path,
    tier_name: str | None,
) -> Path | None:
    """Write the ranked slate for `date`. Returns the path, or None on any failure."""
    try:
        if predictions is None or predictions.empty:
            return None
        cols = [c for c in ROW_COLUMNS if c in predictions.columns]
        rows = json.loads(
            predictions[cols].to_json(orient="records")
        )  # to_json maps NaN -> null and numpy scalars -> JSON natives
        payload = {
            "schema_version": SCHEMA_VERSION,
            "date": date,
            "tier": tier_name,
            "written_at": datetime.now(timezone.utc).isoformat(),
            "n_rows": len(rows),
            "rows": rows,
        }
        slates_dir = Path(picks_dir) / "slates"
        slates_dir.mkdir(parents=True, exist_ok=True)
        path = slates_dir / f"{date}.json"
        atomic_write_text(path, json.dumps(payload, indent=2))
        return path
    except Exception as e:
        log.warning(f"slate persistence failed for {date} (pick path unaffected): {e}")
        return None
