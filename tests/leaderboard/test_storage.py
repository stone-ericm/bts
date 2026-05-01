"""Tests for parquet I/O in the leaderboard package."""
from __future__ import annotations

from datetime import datetime, date

import pyarrow.parquet as pq
import pytest

from bts.leaderboard.models import LeaderboardRow, PickRow, SeasonStats
from bts.leaderboard.storage import (
    write_leaderboard_snapshot,
    append_user_picks,
    write_season_stats,
    read_user_picks,
)


def _row(rank: int, username: str, streak: int) -> LeaderboardRow:
    return LeaderboardRow(
        captured_at=datetime(2026, 5, 1, 14, 0, 0),
        tab="active_streak",
        rank=rank,
        username=username,
        streak=streak,
        hits_today=None,
    )


def _pick(pick_date_str: str, batter_name_unused: str, captured_iso: str, round_id: int = 859) -> PickRow:
    return PickRow(
        captured_at=datetime.fromisoformat(captured_iso),
        round_id=round_id,
        pick_date=date.fromisoformat(pick_date_str),
        pick_number=1,
        unit_id=465,
        bts_player_id=1419,
        result="hit",
        at_bats=3,
        hits=2,
        streak_after=10,
        batter_name=batter_name_unused,  # for test "different captured_at" assertions still work
    )


class TestWriteLeaderboardSnapshot:
    def test_writes_parquet_with_all_rows(self, tmp_path):
        rows = [_row(1, "alpha", 35), _row(2, "beta", 34)]
        out = tmp_path / "leaderboard_snapshots" / "2026-05-01.parquet"
        write_leaderboard_snapshot(out, rows)
        assert out.exists()
        table = pq.read_table(out)
        assert table.num_rows == 2
        assert "username" in table.column_names

    def test_creates_parent_dir(self, tmp_path):
        out = tmp_path / "deep" / "nested" / "snapshots" / "2026-05-01.parquet"
        write_leaderboard_snapshot(out, [_row(1, "x", 10)])
        assert out.exists()

    def test_empty_rows_writes_empty_parquet(self, tmp_path):
        out = tmp_path / "snapshots" / "empty.parquet"
        write_leaderboard_snapshot(out, [])
        assert out.exists()
        table = pq.read_table(out)
        assert table.num_rows == 0
        assert "username" in table.column_names


class TestAppendUserPicks:
    def test_first_write_creates_file(self, tmp_path):
        path = tmp_path / "user_picks" / "tombrady12.parquet"
        append_user_picks(path, [_pick("2026-04-30", "Soto", "2026-05-01T14:00:00", round_id=859)])
        assert path.exists()
        assert pq.read_table(path).num_rows == 1

    def test_append_preserves_existing(self, tmp_path):
        path = tmp_path / "user_picks" / "tombrady12.parquet"
        append_user_picks(path, [_pick("2026-04-30", "Soto", "2026-05-01T14:00:00", round_id=859)])
        append_user_picks(path, [_pick("2026-05-01", "Vlad", "2026-05-02T14:00:00", round_id=860)])
        table = pq.read_table(path)
        assert table.num_rows == 2
        names = table.column("batter_name").to_pylist()
        assert "Soto" in names and "Vlad" in names

    def test_append_keeps_observations_with_different_captured_at(self, tmp_path):
        path = tmp_path / "user_picks" / "tombrady12.parquet"
        append_user_picks(path, [_pick("2026-04-30", "Soto", "2026-05-01T14:00:00")])
        append_user_picks(path, [_pick("2026-04-30", "Soto", "2026-05-01T20:00:00")])
        assert pq.read_table(path).num_rows == 2


class TestReadUserPicks:
    def test_returns_latest_per_pick_date(self, tmp_path):
        path = tmp_path / "user_picks" / "x.parquet"
        append_user_picks(path, [_pick("2026-04-30", "Old", "2026-05-01T08:00:00")])
        append_user_picks(path, [_pick("2026-04-30", "New", "2026-05-01T20:00:00")])
        latest = read_user_picks(path, dedupe="latest_per_pick_date")
        assert latest.num_rows == 1
        assert latest.column("batter_name").to_pylist() == ["New"]

    def test_no_dedupe_returns_all_observations(self, tmp_path):
        path = tmp_path / "user_picks" / "x.parquet"
        append_user_picks(path, [_pick("2026-04-30", "Old", "2026-05-01T08:00:00")])
        append_user_picks(path, [_pick("2026-04-30", "New", "2026-05-01T20:00:00")])
        all_rows = read_user_picks(path, dedupe=None)
        assert all_rows.num_rows == 2

    def test_returns_empty_table_when_path_missing(self, tmp_path):
        path = tmp_path / "user_picks" / "nonexistent.parquet"
        result = read_user_picks(path, dedupe=None)
        assert result.num_rows == 0


class TestWriteSeasonStats:
    def test_writes_stats_parquet(self, tmp_path):
        out = tmp_path / "season_stats" / "2026-05-01.parquet"
        rows = [SeasonStats(
            captured_at=datetime(2026, 5, 1, 14, 0),
            username="tombrady12",
            best_streak=35,
            active_streak=35,
            pick_accuracy_pct=100.0,
        )]
        write_season_stats(out, rows)
        table = pq.read_table(out)
        assert table.num_rows == 1
        assert table.column("pick_accuracy_pct").to_pylist() == [100.0]

    def test_empty_rows_writes_header_only(self, tmp_path):
        out = tmp_path / "season_stats" / "empty.parquet"
        write_season_stats(out, [])
        table = pq.read_table(out)
        assert table.num_rows == 0
        assert "username" in table.column_names
