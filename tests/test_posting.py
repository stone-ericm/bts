import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
from zoneinfo import ZoneInfo
from bts.posting import format_post, should_post_now, get_bluesky_password

ET = ZoneInfo("America/New_York")


class TestFormatPost:
    def test_single_pick(self):
        text = format_post(
            batter="Jacob Wilson", team="ATH", pitcher="Jose Suarez",
            p_game=0.763, streak=3,
        )
        assert text == (
            "Today's pick: Jacob Wilson (ATH)\n"
            "vs Jose Suarez | 76.3%\n\n"
            "Streak: 3"
        )

    def test_double_down(self):
        text = format_post(
            batter="Jacob Wilson", team="ATH", pitcher="Jose Suarez",
            p_game=0.763, streak=3,
            double="Shohei Ohtani", double_p_game=0.741,
        )
        assert "Today's picks: Jacob Wilson (ATH) + Shohei Ohtani" in text
        assert "P(both): 56.5%" in text
        assert "Streak: 3" in text

    def test_streak_zero(self):
        text = format_post(
            batter="Mike Trout", team="LAA", pitcher="Max Scherzer",
            p_game=0.875, streak=0,
        )
        assert "Streak: 0" in text


class TestShouldPostNow:
    @patch("bts.posting._now_et")
    def test_game_within_3_hours_posts(self, mock_now):
        mock_now.return_value = datetime(2026, 4, 1, 14, 0, tzinfo=ET)
        # Game at 4:30pm ET (20:30 UTC) = within 2.5 hours
        assert should_post_now("2026-04-01T20:30:00Z", already_posted=False) is True

    @patch("bts.posting._now_et")
    def test_game_far_away_skips(self, mock_now):
        mock_now.return_value = datetime(2026, 4, 1, 11, 0, tzinfo=ET)
        # Game at 7:10pm ET (23:10 UTC) = 8+ hours away
        assert should_post_now("2026-04-01T23:10:00Z", already_posted=False) is False

    @patch("bts.posting._now_et")
    def test_evening_run_always_posts(self, mock_now):
        mock_now.return_value = datetime(2026, 4, 1, 19, 30, tzinfo=ET)
        # Game at 10pm ET — but it's after 7pm so post anyway
        assert should_post_now("2026-04-02T02:00:00Z", already_posted=False) is True

    def test_already_posted_skips(self):
        assert should_post_now("2026-04-01T23:10:00Z", already_posted=True) is False

    @patch("bts.posting._now_et")
    def test_game_already_started_skips(self, mock_now):
        mock_now.return_value = datetime(2026, 4, 1, 17, 0, tzinfo=ET)
        # Game started 3 hours ago (at 2pm ET = 18:00 UTC)
        assert should_post_now("2026-04-01T18:00:00Z", already_posted=False) is False


class TestGetBlueskyPassword:
    @patch("bts.posting.subprocess.run")
    def test_keychain_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="test-password\n")
        assert get_bluesky_password() == "test-password"

    @patch.dict("os.environ", {"BTS_BLUESKY_PASSWORD": "env-password"})
    @patch("bts.posting.subprocess.run")
    def test_env_fallback_when_keychain_fails(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        assert get_bluesky_password() == "env-password"

    @patch.dict("os.environ", {"BTS_BLUESKY_PASSWORD": "env-password"})
    @patch("bts.posting.subprocess.run", side_effect=FileNotFoundError)
    def test_env_fallback_when_not_macos(self, mock_run):
        assert get_bluesky_password() == "env-password"

    @patch.dict("os.environ", {}, clear=True)
    @patch("bts.posting.subprocess.run", side_effect=FileNotFoundError)
    def test_raises_when_no_password_found(self, mock_run):
        with pytest.raises(RuntimeError, match="not found"):
            get_bluesky_password()
