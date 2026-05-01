"""Read-only analysis on the BTS leaderboard data store.

These functions surface views the dashboard / health checks consume.
They never write to the data store — pure queries against parquet files
populated by the scraper.

Two primary analyses (more to come):
  - consensus_pick: modal batter pick across all tracked users for a date
  - percentile_rank: where our active streak ranks among the active-streak
    leaderboard's tracked top-N
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq

from bts.leaderboard.storage import read_user_picks


def consensus_pick(leaderboard_dir: Path, pick_date: date) -> dict | None:
    """Return modal batter pick across all tracked users for `pick_date`.

    Groups by `bts_player_id` (always set), then resolves a display name
    from the most-frequent non-null `batter_name` observed for that
    player_id. Uses 'latest_per_pick_date' dedup on each user's parquet
    so multi-snapshot observations don't double-count.

    Returns dict with keys:
      consensus_bts_player_id: int
      consensus_batter_name:   str | None  (None if all observations had None)
      consensus_share:         float in [0, 1]
      n_users:                 int (total users with a pick on this date)
    Returns None if no picks for that date are recorded.
    """
    pick_dir = leaderboard_dir / "user_picks"
    if not pick_dir.exists():
        return None
    user_files = list(pick_dir.glob("*.parquet"))
    if not user_files:
        return None

    # Per-user: latest pick observation for `pick_date`
    by_player: Counter[int] = Counter()
    names_by_player: dict[int, Counter[str]] = defaultdict(Counter)
    n_users = 0
    for f in user_files:
        try:
            table = read_user_picks(f, dedupe="latest_per_pick_date")
        except Exception:
            continue
        if table.num_rows == 0:
            continue
        df = table.to_pandas()
        match = df[df["pick_date"] == pick_date]
        if match.empty:
            continue
        # Take the row with the latest captured_at (already deduped to one
        # per pick_date by storage layer, so .iloc[0] is fine)
        row = match.iloc[0]
        pid = int(row["bts_player_id"])
        by_player[pid] += 1
        if row.get("batter_name") and isinstance(row["batter_name"], str):
            names_by_player[pid][row["batter_name"]] += 1
        n_users += 1

    if not by_player:
        return None

    pid, count = by_player.most_common(1)[0]
    # Pick the most-frequent non-null name observed for this pid; fall back to None
    name_counts = names_by_player.get(pid)
    consensus_name = name_counts.most_common(1)[0][0] if name_counts else None

    return {
        "consensus_bts_player_id": pid,
        "consensus_batter_name": consensus_name,
        "consensus_share": count / n_users,
        "n_users": n_users,
    }


def percentile_rank(leaderboard_dir: Path, our_streak: int) -> dict:
    """Compute our percentile rank in the latest active_streak leaderboard.

    Returns:
      pct:      our_streak's percentile (1.0 = top); None if no snapshots
      n_above:  count of tracked users with strictly higher streak
      n_total:  total tracked users in active_streak tab on latest snapshot
    """
    snaps_dir = leaderboard_dir / "leaderboard_snapshots"
    if not snaps_dir.exists():
        return {"pct": None, "n_above": 0, "n_total": 0}
    snaps = sorted(snaps_dir.glob("*.parquet"))
    if not snaps:
        return {"pct": None, "n_above": 0, "n_total": 0}

    latest = snaps[-1]
    df = pq.read_table(latest).to_pandas()
    active = df[df["tab"] == "active_streak"]
    if active.empty:
        return {"pct": None, "n_above": 0, "n_total": 0}

    # Coerce streak to numeric (it may be int or pd.Int with NaN)
    streaks = active["streak"].dropna().astype(int)
    n_above = int((streaks > our_streak).sum())
    n_total = int(len(streaks))
    return {
        "pct": 1.0 - (n_above / n_total) if n_total > 0 else None,
        "n_above": n_above,
        "n_total": n_total,
    }
