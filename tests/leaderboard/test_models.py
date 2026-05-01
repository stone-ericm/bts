"""Tests for pydantic schemas representing leaderboard data."""
from __future__ import annotations

from datetime import datetime, date

import pytest
from pydantic import ValidationError

from bts.leaderboard.models import LeaderboardRow, PickRow, SeasonStats


class TestLeaderboardRow:
    def test_valid_active_streak_row(self):
        row = LeaderboardRow(
            captured_at=datetime(2026, 5, 1, 14, 0, 0),
            tab="active_streak",
            rank=1,
            username="tombrady12",
            streak=35,
            hits_today=None,
        )
        assert row.username == "tombrady12"
        assert row.tab == "active_streak"

    def test_yesterday_tab_has_hits_today(self):
        row = LeaderboardRow(
            captured_at=datetime(2026, 5, 1, 14, 0, 0),
            tab="yesterday",
            rank=1,
            username="someone",
            streak=None,
            hits_today=2,
        )
        assert row.hits_today == 2

    def test_invalid_tab_rejected(self):
        with pytest.raises(ValidationError):
            LeaderboardRow(
                captured_at=datetime(2026, 5, 1),
                tab="not_a_real_tab",
                rank=1, username="x", streak=10, hits_today=None,
            )

    def test_negative_rank_rejected(self):
        with pytest.raises(ValidationError):
            LeaderboardRow(
                captured_at=datetime(2026, 5, 1),
                tab="active_streak",
                rank=0,  # 0 is invalid (must be > 0)
                username="x", streak=10, hits_today=None,
            )


class TestPickRow:
    def test_valid_pick_with_round_id(self):
        row = PickRow(
            captured_at=datetime(2026, 5, 1, 14, 0),
            round_id=859,
            pick_date=date(2026, 4, 30),
            batter_name="Juan Soto",
            batter_team="NYM",
            opponent_team="WSH",
            home_or_away="home",
            at_bats=3,
            hits=2,
            streak_after=35,
            batter_id=665742,
        )
        assert row.hits == 2
        assert row.round_id == 859
        assert row.streak_after == 35

    def test_batter_id_optional(self):
        row = PickRow(
            captured_at=datetime(2026, 5, 1),
            round_id=860,
            pick_date=date(2026, 5, 1),
            batter_name="Some Player",
            batter_team="ABC", opponent_team="XYZ", home_or_away="away",
            at_bats=3, hits=1, streak_after=5,
        )
        assert row.batter_id is None

    def test_invalid_home_away_rejected(self):
        with pytest.raises(ValidationError):
            PickRow(
                captured_at=datetime(2026, 5, 1),
                round_id=859,
                pick_date=date(2026, 4, 30),
                batter_name="X", batter_team="A", opponent_team="B",
                home_or_away="elsewhere",
                at_bats=3, hits=1, streak_after=5,
            )

    def test_negative_round_id_rejected(self):
        with pytest.raises(ValidationError):
            PickRow(
                captured_at=datetime(2026, 5, 1),
                round_id=-1,
                pick_date=date(2026, 4, 30),
                batter_name="X", batter_team="A", opponent_team="B",
                home_or_away="home",
                at_bats=3, hits=1, streak_after=5,
            )


class TestSeasonStats:
    def test_valid_stats(self):
        s = SeasonStats(
            captured_at=datetime(2026, 5, 1, 14, 0),
            username="tombrady12",
            best_streak=35,
            active_streak=35,
            pick_accuracy_pct=100.0,
        )
        assert s.pick_accuracy_pct == 100.0

    def test_accuracy_out_of_bounds_rejected(self):
        with pytest.raises(ValidationError):
            SeasonStats(
                captured_at=datetime(2026, 5, 1),
                username="x",
                best_streak=10, active_streak=5,
                pick_accuracy_pct=101.0,
            )
