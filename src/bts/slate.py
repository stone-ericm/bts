"""Persist the full ranked daily slate for realized slate-level analysis.

Until 2026-06-11 the candidate-level predictions were computed every cycle
and discarded — only pick/double_down/runner_up survived. That made realized
slate metrics (rolling AUC, sub-top-1 ranking quality, live feature
attribution) impossible to compute after the fact: the M3 serving-staleness
closeout (docs/audit/2026-06-11-m3-serving-staleness.md) could not quantify
bpm's realized live contribution for exactly this reason.

One JSON file per date under {picks_dir}/slates/. Last write wins: re-runs
within a day overwrite it, so the file is the last slate persisted that day.
It is written before selection, so it is not proof that it produced the final
pick. Persistence is observability — it must NEVER break the pick path, so
save_slate swallows and logs every failure.

v3 (C2 step 2a, design docs/superpowers/specs/2026-10-06-c2-2a-serving-record-design.md):
the envelope's `serving` is the serving witness the local tier attached
(bts.serving_witness: recipe, model, inputs, flags, calibration), or null when
there is none (another tier) or it cannot be serialized; and each local row's
`projected` is an explicit boolean (false = posted lineup). v2 files (deployed
2026-10-06 without `serving`) keep their tag.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bts.util import atomic_write_text

log = logging.getLogger(__name__)

SCHEMA_VERSION = "bts_slate_v3"   # v3 (C2 step 2a): the envelope's serving witness and an explicit projected
                                  # boolean; v2 (C1 4a prerequisite P1): each row's game_time and schedule status

# Persisted per candidate when present in the predictions frame. Feature
# values are deliberately excluded: they are reconstructable from the PA
# parquets (validated to 5.55e-17 by scripts/replay_m3_serving_parity.py),
# while the model outputs below are not.
# v2 adds game_time (the schedule's gameDate as known at this run) and status (the
# schedule's detailedState fetched in this run, before the write): C1 rank 4a's
# eligibility rule needs the run-known start and a pregame-status observation at or
# before the write (docs/sota_audit/2026-10-04-prereg-c1-calibration.md section 2).
ROW_COLUMNS = [
    "batter_id", "batter_name", "team", "game_pk", "lineup",
    "pitcher_id", "pitcher_name",
    "p_game_hit", "p_game_blend", "p_hit_vs_starter", "p_hit_vs_reliever",
    "est_pas", "flags", "projected", "game_time", "status",
]


def _serving(predictions: pd.DataFrame, date: str):
    """The attached serving witness if it serializes as strict JSON, else None. Never raises."""
    try:
        serving = predictions.attrs.get("serving")
        json.dumps(serving, allow_nan=False)
        return serving
    except Exception as e:
        log.warning(f"serving witness not serializable for {date} (slate still written): {e}")
        return None


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
            "serving": _serving(predictions, date),
        }
        slates_dir = Path(picks_dir) / "slates"
        slates_dir.mkdir(parents=True, exist_ok=True)
        path = slates_dir / f"{date}.json"
        atomic_write_text(path, json.dumps(payload, indent=2))
        return path
    except Exception as e:
        log.warning(f"slate persistence failed for {date} (pick path unaffected): {e}")
        return None
