"""Parquet I/O for leaderboard data.

Writes are atomic-by-write-then-rename. user_picks parquet is append-only:
every observation is preserved (distinguished by captured_at). Dedup happens
on read via the read_user_picks(dedupe=...) parameter.

Schema is enforced via pydantic models (see models.py) — non-conforming
rows raise on construction, before they reach storage.
"""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from bts.leaderboard.models import LeaderboardRow, PickRow, SeasonStats


_LEADERBOARD_SCHEMA = pa.schema([
    ("captured_at", pa.timestamp("ms")),
    ("tab", pa.string()),
    ("rank", pa.int32()),
    ("username", pa.string()),
    ("streak", pa.int32()),
    ("hits_today", pa.int32()),
])

_USER_PICKS_SCHEMA = pa.schema([
    ("captured_at", pa.timestamp("ms")),
    ("round_id", pa.int32()),
    ("pick_date", pa.date32()),
    ("batter_name", pa.string()),
    ("batter_team", pa.string()),
    ("opponent_team", pa.string()),
    ("home_or_away", pa.string()),
    ("at_bats", pa.int32()),
    ("hits", pa.int32()),
    ("streak_after", pa.int32()),
    ("batter_id", pa.int64()),
])

_SEASON_STATS_SCHEMA = pa.schema([
    ("captured_at", pa.timestamp("ms")),
    ("username", pa.string()),
    ("best_streak", pa.int32()),
    ("active_streak", pa.int32()),
    ("pick_accuracy_pct", pa.float64()),
])


def _rows_to_table(rows, schema: pa.Schema) -> pa.Table:
    cols: dict[str, list] = {f.name: [] for f in schema}
    for r in rows:
        d = r.model_dump()
        for f in schema:
            cols[f.name].append(d.get(f.name))
    return pa.table(cols, schema=schema)


def _atomic_write(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    tmp.rename(path)


def write_leaderboard_snapshot(path: Path, rows: list[LeaderboardRow]) -> None:
    """Write leaderboard snapshot rows to parquet."""
    _atomic_write(path, _rows_to_table(rows, _LEADERBOARD_SCHEMA))


def append_user_picks(path: Path, rows: list[PickRow]) -> None:
    """Append-only writer for per-user picks log.

    Reads existing parquet (if any), concatenates new rows, writes back.
    Every observation is preserved — dedup happens on read, not write.
    """
    new_table = _rows_to_table(rows, _USER_PICKS_SCHEMA)
    if path.exists():
        existing = pq.read_table(path)
        combined = pa.concat_tables([existing, new_table])
    else:
        combined = new_table
    _atomic_write(path, combined)


def read_user_picks(path: Path, dedupe: str | None = None) -> pa.Table:
    """Read user picks with optional dedup.

    dedupe=None: return raw appended observations
    dedupe='latest_per_pick_date': return only newest captured_at per pick_date
    """
    if not path.exists():
        return _rows_to_table([], _USER_PICKS_SCHEMA)
    table = pq.read_table(path)
    if dedupe is None:
        return table
    if dedupe == "latest_per_pick_date":
        df = table.to_pandas().sort_values("captured_at").drop_duplicates(
            subset=["pick_date"], keep="last"
        )
        return pa.Table.from_pandas(df, schema=_USER_PICKS_SCHEMA, preserve_index=False)
    raise ValueError(f"unknown dedupe mode: {dedupe!r}")


def write_season_stats(path: Path, rows: list[SeasonStats]) -> None:
    _atomic_write(path, _rows_to_table(rows, _SEASON_STATS_SCHEMA))
