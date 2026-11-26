"""Bronze layer for Statcast per-pitch swing data (campaign Stage 0).

Pulls land WIDE (Codex round-2): ids, event context, count/handedness, zone
geometry, and ALL swing/bat-tracking columns — storage is cheap, re-pulls and
Savant schema drift are not. Per-pitch swing data is NEVER merged into the PA
frame (no stable pitch keys there); the swing_daily layer (bts.features.swing)
aggregates from these files.

pybaseball hygiene (its datasource has no request timeout): callers must set
a project-scoped cache (PYBASEBALL_CACHE), pull serially in date chunks with
bounded retries, and re-pull a rolling recent window for incremental updates
(stale current-season cache). See scripts/backfill_swing_data.py.

Spec: docs/superpowers/specs/2026-06-12-statcast-swing-campaign-design.md
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Wide bronze schema. Absent columns (older dates / Savant drift) are created
# as NA so every season file shares one schema; the manifest records what the
# raw pull actually contained.
BRONZE_COLUMNS = [
    # identity / join keys
    "game_date", "game_pk", "at_bat_number", "pitch_number", "sv_id",
    "batter", "pitcher",
    # event context
    "events", "description", "type", "pitch_type", "game_type",
    "balls", "strikes", "stand", "p_throws",
    # location / zone geometry
    "zone", "plate_x", "plate_z", "sz_top", "sz_bot",
    # swing / bat-tracking payload
    "miss_distance", "swing_length", "attack_angle", "attack_direction",
    "swing_path_tilt",
    "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches",
]


def normalize_bronze(raw: pd.DataFrame) -> pd.DataFrame:
    """Project a raw pybaseball.statcast() frame onto the bronze schema.

    Regular season only; missing bronze columns created as NA; extra raw
    columns dropped (but recorded by the caller in the manifest).
    """
    df = raw.copy()
    if "game_type" in df.columns:
        df = df[df["game_type"] == "R"]
    for col in BRONZE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[BRONZE_COLUMNS].reset_index(drop=True)


def write_bronze_season(
    df: pd.DataFrame,
    season: int,
    out_dir: Path,
    raw_columns: list[str],
) -> Path:
    """Write swing_{season}.parquet + a manifest recording the raw pull."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"swing_{season}.parquet"
    df.to_parquet(path, index=False)
    manifest = {
        "season": season,
        "n_rows": int(len(df)),
        "pulled_at": datetime.now(timezone.utc).isoformat(),
        "bronze_columns": BRONZE_COLUMNS,
        "raw_columns": raw_columns,
        "date_min": str(df["game_date"].min()) if len(df) else None,
        "date_max": str(df["game_date"].max()) if len(df) else None,
    }
    (out_dir / f"swing_{season}.manifest.json").write_text(json.dumps(manifest, indent=2))
    return path


def month_chunks(start: str, end: str) -> list[tuple[str, str]]:
    """Split [start, end] into calendar-month-bounded chunks (inclusive)."""
    chunks = []
    cur = pd.Timestamp(start)
    final = pd.Timestamp(end)
    while cur <= final:
        month_end = (cur + pd.offsets.MonthEnd(0))
        chunk_end = min(month_end, final)
        chunks.append((str(cur.date()), str(chunk_end.date())))
        cur = chunk_end + pd.Timedelta(days=1)
    return chunks
