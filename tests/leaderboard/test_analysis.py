"""Tests for the analysis layer: consensus_pick + percentile_rank."""
from __future__ import annotations

from datetime import datetime, date
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from bts.leaderboard.models import LeaderboardRow, PickRow, SeasonStats
from bts.leaderboard.storage import (
    append_user_picks, write_leaderboard_snapshot, write_season_stats,
)
from bts.leaderboard.analysis import consensus_pick, percentile_rank


def _pick(pick_date_str: str, batter_name: str | None, bts_player_id: int,
          captured_iso: str = "2026-05-01T14:00:00", round_id: int = 859,
          pick_number: int = 1) -> PickRow:
    return PickRow(
        captured_at=datetime.fromisoformat(captured_iso),
        round_id=round_id,
        pick_date=date.fromisoformat(pick_date_str),
        pick_number=pick_number,
        unit_id=1,
        bts_player_id=bts_player_id,
        result="hit",
        at_bats=3,
        hits=1,
        streak_after=10,
        batter_name=batter_name,
    )


class TestConsensusPick:
    def test_returns_modal_batter_when_majority(self, tmp_path):
        leaderboard = tmp_path / "leaderboard"
        # 3 users picked Soto (player_id=665742), 1 picked Vlad (player_id=665489)
        for u in ["alpha", "beta", "gamma"]:
            append_user_picks(
                leaderboard / "user_picks" / f"{u}.parquet",
                [_pick("2026-05-01", "Juan Soto", 665742)],
            )
        append_user_picks(
            leaderboard / "user_picks" / "delta.parquet",
            [_pick("2026-05-01", "Vladimir Guerrero Jr.", 665489)],
        )
        result = consensus_pick(leaderboard, pick_date=date(2026, 5, 1))
        assert result is not None
        assert result["consensus_bts_player_id"] == 665742
        assert result["consensus_batter_name"] == "Juan Soto"
        assert result["consensus_share"] == 0.75
        assert result["n_users"] == 4

    def test_groups_by_bts_player_id_when_names_inconsistent(self, tmp_path):
        """If two users observed different batter_name strings for the same player_id
        (e.g., one had it None), consensus_pick should still group them together."""
        leaderboard = tmp_path / "leaderboard"
        append_user_picks(
            leaderboard / "user_picks" / "alpha.parquet",
            [_pick("2026-05-01", "Juan Soto", 665742)],
        )
        append_user_picks(
            leaderboard / "user_picks" / "beta.parquet",
            [_pick("2026-05-01", None, 665742)],  # name unresolved
        )
        result = consensus_pick(leaderboard, pick_date=date(2026, 5, 1))
        assert result["consensus_bts_player_id"] == 665742
        assert result["consensus_batter_name"] == "Juan Soto"  # picks the non-null
        assert result["n_users"] == 2

    def test_returns_none_when_no_users_picked_for_date(self, tmp_path):
        leaderboard = tmp_path / "leaderboard"
        # User has a pick for a different date
        append_user_picks(
            leaderboard / "user_picks" / "alpha.parquet",
            [_pick("2026-04-30", "Juan Soto", 665742)],
        )
        result = consensus_pick(leaderboard, pick_date=date(2026, 5, 1))
        assert result is None

    def test_returns_none_when_no_users_at_all(self, tmp_path):
        result = consensus_pick(tmp_path / "leaderboard", pick_date=date(2026, 5, 1))
        assert result is None

    def test_uses_latest_per_pick_date_dedup(self, tmp_path):
        """If same user has two captured_at observations for same pick_date,
        consensus_pick uses the latest one."""
        leaderboard = tmp_path / "leaderboard"
        append_user_picks(
            leaderboard / "user_picks" / "alpha.parquet",
            [
                _pick("2026-05-01", "Old Pick", 1, captured_iso="2026-05-01T08:00:00"),
                _pick("2026-05-01", "New Pick", 2, captured_iso="2026-05-01T20:00:00"),
            ],
        )
        result = consensus_pick(leaderboard, pick_date=date(2026, 5, 1))
        assert result["consensus_bts_player_id"] == 2  # the newer observation
        assert result["consensus_batter_name"] == "New Pick"


class TestPercentileRank:
    def _row(self, rank: int, username: str, streak: int) -> LeaderboardRow:
        return LeaderboardRow(
            captured_at=datetime(2026, 5, 1, 14, 0),
            tab="active_streak",
            rank=rank,
            username=username,
            streak=streak,
            hits_today=None,
        )

    def test_our_streak_ranks_among_active_users(self, tmp_path):
        leaderboard = tmp_path / "leaderboard"
        # 10 users: streaks 35, 34, ..., 26
        rows = [self._row(i + 1, f"u{i}", 35 - i) for i in range(10)]
        write_leaderboard_snapshot(
            leaderboard / "leaderboard_snapshots" / "2026-05-01.parquet", rows,
        )
        # Our streak of 30 -> 5 users have higher (35,34,33,32,31) -> 50th percentile
        rank = percentile_rank(leaderboard, our_streak=30)
        assert rank["n_above"] == 5
        assert rank["n_total"] == 10
        assert 0.4 <= rank["pct"] <= 0.6

    def test_top_rank_when_we_are_highest(self, tmp_path):
        leaderboard = tmp_path / "leaderboard"
        rows = [self._row(i + 1, f"u{i}", 10 - i) for i in range(10)]  # streaks 10..1
        write_leaderboard_snapshot(
            leaderboard / "leaderboard_snapshots" / "2026-05-01.parquet", rows,
        )
        rank = percentile_rank(leaderboard, our_streak=50)
        assert rank["n_above"] == 0
        assert rank["pct"] == 1.0

    def test_uses_latest_snapshot_only(self, tmp_path):
        leaderboard = tmp_path / "leaderboard"
        # Older snapshot: high streaks (we'd rank low)
        old_rows = [self._row(i + 1, f"u{i}", 100 - i) for i in range(5)]
        write_leaderboard_snapshot(
            leaderboard / "leaderboard_snapshots" / "2026-04-01.parquet", old_rows,
        )
        # Newer snapshot: low streaks (we'd rank high)
        new_rows = [self._row(i + 1, f"u{i}", 5 - i) for i in range(5)]
        write_leaderboard_snapshot(
            leaderboard / "leaderboard_snapshots" / "2026-05-01.parquet", new_rows,
        )
        rank = percentile_rank(leaderboard, our_streak=10)
        # Should use the May 1 snapshot, where streaks max out at 5; 10 beats all
        assert rank["n_above"] == 0

    def test_returns_none_pct_when_no_snapshots(self, tmp_path):
        leaderboard = tmp_path / "leaderboard"
        rank = percentile_rank(leaderboard, our_streak=10)
        assert rank["pct"] is None
        assert rank["n_total"] == 0

    def test_only_active_streak_tab_counted(self, tmp_path):
        """Other tabs (all_season, all_time, yesterday) should be excluded."""
        leaderboard = tmp_path / "leaderboard"
        rows = [
            self._row(1, "x", 5),  # active_streak: streak 5
            LeaderboardRow(
                captured_at=datetime(2026, 5, 1, 14, 0),
                tab="all_time", rank=1, username="legend", streak=51, hits_today=None,
            ),
        ]
        write_leaderboard_snapshot(
            leaderboard / "leaderboard_snapshots" / "2026-05-01.parquet", rows,
        )
        # Our streak of 10 vs only the active_streak row (streak 5)
        # 51 in all_time should NOT count as "above"
        rank = percentile_rank(leaderboard, our_streak=10)
        assert rank["n_above"] == 0
        assert rank["n_total"] == 1
