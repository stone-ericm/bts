import json
import pytest
from unittest.mock import patch, MagicMock
from bts.picks import Pick, DailyPick, save_pick, load_pick, load_streak, save_streak, update_streak, load_saver_available
from bts.picks import pick_from_row
from bts.picks import get_game_statuses, get_game_statuses_detailed, check_hit


def _sample_pick(**overrides):
    defaults = dict(
        batter_name="Jacob Wilson",
        batter_id=700363,
        team="ATH",
        lineup_position=1,
        pitcher_name="Jose Suarez",
        pitcher_id=660761,
        p_game_hit=0.763,
        flags=[],
        projected_lineup=False,
        game_pk=778899,
        game_time="2026-04-01T23:10:00Z",
    )
    defaults.update(overrides)
    return Pick(**defaults)


def _sample_daily(pick=None, **overrides):
    defaults = dict(
        date="2026-04-01",
        run_time="2026-04-01T15:00:00+00:00",
        pick=pick or _sample_pick(),
        double_down=None,
        runner_up={"batter_name": "Jake Mangum", "p_game_hit": 0.726},
        bluesky_posted=False,
        bluesky_uri=None,
    )
    defaults.update(overrides)
    return DailyPick(**defaults)


class TestPickFileIO:
    def test_save_and_load_roundtrip(self, tmp_path):
        daily = _sample_daily()
        save_pick(daily, tmp_path)

        loaded = load_pick("2026-04-01", tmp_path)
        assert loaded is not None
        assert loaded.pick.batter_name == "Jacob Wilson"
        assert loaded.pick.p_game_hit == pytest.approx(0.763)
        assert loaded.pick.game_pk == 778899

    def test_load_nonexistent_returns_none(self, tmp_path):
        assert load_pick("2099-01-01", tmp_path) is None

    def test_save_creates_directory(self, tmp_path):
        subdir = tmp_path / "nested" / "picks"
        daily = _sample_daily()
        save_pick(daily, subdir)
        assert (subdir / "2026-04-01.json").exists()

    def test_save_overwrites_existing(self, tmp_path):
        daily = _sample_daily()
        save_pick(daily, tmp_path)

        updated = _sample_daily(bluesky_posted=True, bluesky_uri="at://did:plc:xxx/post/yyy")
        save_pick(updated, tmp_path)

        loaded = load_pick("2026-04-01", tmp_path)
        assert loaded.bluesky_posted is True

    def test_roundtrip_with_double_down(self, tmp_path):
        double = _sample_pick(batter_name="Shohei Ohtani", batter_id=660271, p_game_hit=0.741)
        daily = _sample_daily(double_down=double)
        save_pick(daily, tmp_path)

        loaded = load_pick("2026-04-01", tmp_path)
        assert loaded.double_down is not None
        assert loaded.double_down.batter_name == "Shohei Ohtani"

    def test_roundtrip_preserves_flags(self, tmp_path):
        pick = _sample_pick(flags=["IL? (8d rest)", "PROJECTED lineup"], projected_lineup=True)
        daily = _sample_daily(pick=pick)
        save_pick(daily, tmp_path)

        loaded = load_pick("2026-04-01", tmp_path)
        assert loaded.pick.flags == ["IL? (8d rest)", "PROJECTED lineup"]
        assert loaded.pick.projected_lineup is True

    def test_json_format_matches_spec(self, tmp_path):
        daily = _sample_daily()
        save_pick(daily, tmp_path)

        raw = json.loads((tmp_path / "2026-04-01.json").read_text())
        assert raw["date"] == "2026-04-01"
        assert raw["pick"]["batter_name"] == "Jacob Wilson"
        assert raw["pick"]["game_pk"] == 778899
        assert raw["double_down"] is None
        assert raw["runner_up"]["batter_name"] == "Jake Mangum"

    def test_no_streak_in_pick_file(self, tmp_path):
        """Streak should not be stored in the pick file (Issue 3)."""
        daily = _sample_daily()
        save_pick(daily, tmp_path)

        raw = json.loads((tmp_path / "2026-04-01.json").read_text())
        assert "streak" not in raw


class TestPickFromRow:
    def _row(self, **overrides):
        defaults = {
            "batter_name": "Jacob Wilson",
            "batter_id": 700363,
            "team": "ATH",
            "lineup": 1,
            "pitcher_name": "Jose Suarez",
            "pitcher_id": 660761,
            "p_game_hit": 0.763,
            "flags": "",
            "game_pk": 778899,
            "game_time": "2026-04-01T23:10:00Z",
        }
        defaults.update(overrides)
        return defaults

    def test_basic_row(self):
        pick = pick_from_row(self._row())
        assert pick.batter_name == "Jacob Wilson"
        assert pick.batter_id == 700363
        assert pick.lineup_position == 1
        assert pick.flags == []
        assert pick.projected_lineup is False

    def test_flags_parsed_from_comma_string(self):
        pick = pick_from_row(self._row(flags="IL? (8d rest), PROJECTED lineup"))
        assert pick.flags == ["IL? (8d rest)", "PROJECTED lineup"]
        assert pick.projected_lineup is True

    def test_none_pitcher_id(self):
        pick = pick_from_row(self._row(pitcher_id=None))
        assert pick.pitcher_id is None

    def test_missing_flags_key(self):
        row = self._row()
        del row["flags"]
        pick = pick_from_row(row)
        assert pick.flags == []
        assert pick.projected_lineup is False


class TestStreak:
    def test_load_empty_returns_zero(self, tmp_path):
        assert load_streak(tmp_path) == 0

    def test_save_and_load_roundtrip(self, tmp_path):
        save_streak(5, tmp_path)
        assert load_streak(tmp_path) == 5

    def test_update_single_hit_increments(self, tmp_path):
        save_streak(3, tmp_path)
        new = update_streak([True], tmp_path)
        assert new == 4
        assert load_streak(tmp_path) == 4

    def test_update_single_miss_resets(self, tmp_path):
        save_streak(5, tmp_path)  # streak 5, outside saver range
        new = update_streak([False], tmp_path)
        assert new == 0
        assert load_streak(tmp_path) == 0

    def test_update_double_both_hit_adds_two(self, tmp_path):
        save_streak(5, tmp_path)
        new = update_streak([True, True], tmp_path)
        assert new == 7

    def test_update_double_one_miss_resets(self, tmp_path):
        save_streak(5, tmp_path)
        new = update_streak([True, False], tmp_path)
        assert new == 0

    def test_update_double_both_miss_resets(self, tmp_path):
        save_streak(5, tmp_path)
        new = update_streak([False, False], tmp_path)
        assert new == 0

    def test_saver_preserves_streak_at_12(self, tmp_path):
        """Miss at streak 12 with saver → streak holds, saver consumed."""
        save_streak(12, tmp_path, saver_available=True)
        new = update_streak([False], tmp_path)
        assert new == 12
        assert load_saver_available(tmp_path) is False

    def test_saver_not_available_after_use(self, tmp_path):
        """After saver is consumed, next miss resets normally."""
        save_streak(12, tmp_path, saver_available=True)
        update_streak([False], tmp_path)  # saver fires
        save_streak(14, tmp_path)  # rebuild streak (saver_available preserved as False)
        new = update_streak([False], tmp_path)
        assert new == 0  # no saver, reset

    def test_saver_does_not_fire_above_15(self, tmp_path):
        save_streak(16, tmp_path, saver_available=True)
        new = update_streak([False], tmp_path)
        assert new == 0
        assert load_saver_available(tmp_path) is True  # unused

    def test_saver_does_not_fire_below_10(self, tmp_path):
        save_streak(8, tmp_path, saver_available=True)
        new = update_streak([False], tmp_path)
        assert new == 0
        assert load_saver_available(tmp_path) is True

    def test_saver_default_available(self, tmp_path):
        assert load_saver_available(tmp_path) is True


def _mock_schedule_response(games):
    """Build a mock MLB schedule API response."""
    return {"dates": [{"games": games}]}


def _mock_feed_response(batter_id, hits, status_code="F"):
    """Build a mock MLB game feed with boxscore stats."""
    return {
        "gameData": {"status": {"abstractGameCode": status_code}},
        "liveData": {
            "boxscore": {
                "teams": {
                    "away": {"players": {}},
                    "home": {
                        "players": {
                            f"ID{batter_id}": {
                                "stats": {"batting": {"hits": hits}},
                            }
                        }
                    },
                }
            }
        },
    }


class TestGameStatuses:
    @patch("bts.picks.retry_urlopen")
    def test_returns_status_map(self, mock_urlopen):
        mock_urlopen.return_value.read.return_value = json.dumps(
            _mock_schedule_response([
                {"gamePk": 100, "status": {"abstractGameCode": "P"}},
                {"gamePk": 200, "status": {"abstractGameCode": "L"}},
                {"gamePk": 300, "status": {"abstractGameCode": "F"}},
            ])
        ).encode()

        result = get_game_statuses("2026-04-01")
        assert result == {100: "P", 200: "L", 300: "F"}

    @patch("bts.picks.retry_urlopen")
    def test_empty_schedule(self, mock_urlopen):
        mock_urlopen.return_value.read.return_value = json.dumps(
            {"dates": []}
        ).encode()
        assert get_game_statuses("2026-04-01") == {}


class TestCheckHit:
    @patch("bts.picks.retry_urlopen")
    def test_batter_got_hit(self, mock_urlopen):
        mock_urlopen.return_value.read.return_value = json.dumps(
            _mock_feed_response(batter_id=700363, hits=2, status_code="F")
        ).encode()
        assert check_hit(778899, 700363) is True

    @patch("bts.picks.retry_urlopen")
    def test_batter_no_hit(self, mock_urlopen):
        mock_urlopen.return_value.read.return_value = json.dumps(
            _mock_feed_response(batter_id=700363, hits=0, status_code="F")
        ).encode()
        assert check_hit(778899, 700363) is False

    @patch("bts.picks.retry_urlopen")
    def test_game_not_final_returns_none(self, mock_urlopen):
        mock_urlopen.return_value.read.return_value = json.dumps(
            _mock_feed_response(batter_id=700363, hits=0, status_code="L")
        ).encode()
        assert check_hit(778899, 700363) is None

    @patch("bts.picks.retry_urlopen")
    def test_batter_not_in_game_returns_none(self, mock_urlopen):
        """Scratched players return None, not False (Issue 5)."""
        mock_urlopen.return_value.read.return_value = json.dumps(
            _mock_feed_response(batter_id=999999, hits=1, status_code="F")
        ).encode()
        # Batter 700363 not in boxscore (only 999999 is)
        assert check_hit(778899, 700363) is None

    @patch("bts.picks.retry_urlopen")
    def test_none_game_pk_falls_back_to_date_search(self, mock_urlopen):
        """Picks with game_pk=None (pre-scheduler) use date-based search."""
        schedule_resp = _mock_schedule_response([
            {"gamePk": 500, "status": {"abstractGameCode": "F"}},
        ])
        feed_resp = _mock_feed_response(batter_id=700363, hits=1, status_code="F")

        def side_effect(url, timeout=15):
            assert "game/None" not in url, "Should not call API with None game_pk"
            data = schedule_resp if "schedule" in url else feed_resp
            m = MagicMock()
            m.read.return_value = json.dumps(data).encode()
            return m

        mock_urlopen.side_effect = side_effect
        assert check_hit(None, 700363, date="2026-03-30") is True

    def test_none_game_pk_no_date_returns_none(self):
        """game_pk=None with no date should return None without API calls."""
        assert check_hit(None, 700363) is None


class TestGetGameStatusesExtended:
    @patch("bts.picks.retry_urlopen")
    def test_returns_detailed_status_map(self, mock_urlopen):
        mock_urlopen.return_value.read.return_value = json.dumps(
            _mock_schedule_response([
                {"gamePk": 100, "status": {"abstractGameCode": "P", "detailedState": "Scheduled"}},
                {"gamePk": 200, "status": {"abstractGameCode": "L", "detailedState": "In Progress"}},
                {"gamePk": 300, "status": {"abstractGameCode": "F", "detailedState": "Final"}},
            ])
        ).encode()

        result = get_game_statuses_detailed("2026-04-01")
        assert result == {
            100: {"abstract": "P", "detailed": "Scheduled"},
            200: {"abstract": "L", "detailed": "In Progress"},
            300: {"abstract": "F", "detailed": "Final"},
        }

    @patch("bts.picks.retry_urlopen")
    def test_suspended_game_detected(self, mock_urlopen):
        mock_urlopen.return_value.read.return_value = json.dumps(
            _mock_schedule_response([
                {"gamePk": 400, "status": {"abstractGameCode": "F", "detailedState": "Suspended"}},
            ])
        ).encode()

        result = get_game_statuses_detailed("2026-04-01")
        assert result[400]["abstract"] == "F"
        assert result[400]["detailed"] == "Suspended"

    @patch("bts.picks.retry_urlopen")
    def test_missing_detailed_state_defaults_to_empty(self, mock_urlopen):
        mock_urlopen.return_value.read.return_value = json.dumps(
            _mock_schedule_response([
                {"gamePk": 500, "status": {"abstractGameCode": "P"}},
            ])
        ).encode()

        result = get_game_statuses_detailed("2026-04-01")
        assert result[500] == {"abstract": "P", "detailed": ""}

    @patch("bts.picks.retry_urlopen")
    def test_empty_schedule(self, mock_urlopen):
        mock_urlopen.return_value.read.return_value = json.dumps(
            {"dates": []}
        ).encode()
        assert get_game_statuses_detailed("2026-04-01") == {}


class TestReconcileResults:
    def test_no_corrections_when_results_match(self, tmp_path):
        from datetime import date as date_cls, timedelta
        from bts.picks import reconcile_results, save_pick, save_streak, DailyPick, Pick

        yesterday = (date_cls.today() - timedelta(days=1)).isoformat()
        pick = Pick(
            batter_name="Test", batter_id=123, team="NYM", lineup_position=1,
            pitcher_name="P", pitcher_id=456, p_game_hit=0.82, flags=[],
            projected_lineup=False, game_pk=100, game_time=f"{yesterday}T23:00:00Z",
        )
        daily = DailyPick(
            date=yesterday, run_time=f"{yesterday}T20:00:00Z",
            pick=pick, double_down=None, runner_up=None,
            result="hit",
        )
        save_pick(daily, tmp_path)
        save_streak(1, tmp_path)

        with patch("bts.picks.check_hit", return_value=True):
            corrections = reconcile_results(tmp_path, lookback_days=8)
        assert corrections == []

    def test_corrects_hit_to_miss(self, tmp_path):
        from datetime import date as date_cls, timedelta
        from bts.picks import reconcile_results, save_pick, save_streak, load_streak, DailyPick, Pick

        # Use yesterday's date so it's always within the 8-day lookback window
        yesterday = (date_cls.today() - timedelta(days=1)).isoformat()
        pick = Pick(
            batter_name="Test", batter_id=123, team="NYM", lineup_position=1,
            pitcher_name="P", pitcher_id=456, p_game_hit=0.82, flags=[],
            projected_lineup=False, game_pk=100, game_time=f"{yesterday}T23:00:00Z",
        )
        daily = DailyPick(
            date=yesterday, run_time=f"{yesterday}T20:00:00Z",
            pick=pick, double_down=None, runner_up=None,
            result="hit",
        )
        save_pick(daily, tmp_path)
        save_streak(1, tmp_path)

        with patch("bts.picks.check_hit", return_value=False):
            corrections = reconcile_results(tmp_path, lookback_days=8)
        assert len(corrections) == 1
        assert corrections[0]["old_result"] == "hit"
        assert corrections[0]["new_result"] == "miss"
        assert load_streak(tmp_path) == 0

    def test_preview_pick_for_tomorrow_does_not_break_streak_walk(self, tmp_path):
        """Regression for 2026-04-15 streak-reset bug.

        When ``bts preview`` pre-generates tomorrow's pick, the file exists
        with ``result=None`` before any games are played. The reconcile
        backward-walk previously hit that file first and broke the loop,
        resetting streak to 0 on every 2am reconcile run — wiping out the
        previous day's win. This test verifies that a today-or-later pick
        file with no result is SKIPPED, not treated as a break condition.
        """
        from datetime import date as date_cls, timedelta
        from bts.picks import reconcile_results, save_pick, save_streak, load_streak, DailyPick, Pick

        today_iso = date_cls.today().isoformat()
        yesterday_iso = (date_cls.today() - timedelta(days=1)).isoformat()

        # Yesterday: a hit + double-down (streak should become +2)
        yest_pick = Pick(
            batter_name="YestPrimary", batter_id=100, team="NYM", lineup_position=1,
            pitcher_name="P1", pitcher_id=200, p_game_hit=0.80, flags=[],
            projected_lineup=False, game_pk=1000, game_time=f"{yesterday_iso}T23:00:00Z",
        )
        yest_dd = Pick(
            batter_name="YestDouble", batter_id=101, team="CLE", lineup_position=2,
            pitcher_name="P2", pitcher_id=201, p_game_hit=0.78, flags=[],
            projected_lineup=False, game_pk=1001, game_time=f"{yesterday_iso}T23:00:00Z",
        )
        save_pick(DailyPick(
            date=yesterday_iso, run_time=f"{yesterday_iso}T20:00:00Z",
            pick=yest_pick, double_down=yest_dd, runner_up=None, result="hit",
        ), tmp_path)

        # Today: pre-generated preview pick, no result yet
        today_pick = Pick(
            batter_name="TodayPrimary", batter_id=200, team="ATL", lineup_position=1,
            pitcher_name="P3", pitcher_id=300, p_game_hit=0.76, flags=[],
            projected_lineup=True, game_pk=2000, game_time=f"{today_iso}T23:00:00Z",
        )
        save_pick(DailyPick(
            date=today_iso, run_time=f"{today_iso}T16:05:00Z",
            pick=today_pick, double_down=None, runner_up=None, result=None,
        ), tmp_path)

        save_streak(2, tmp_path)  # pre-existing state from before reconcile

        with patch("bts.picks.check_hit", return_value=True):
            corrections = reconcile_results(tmp_path, lookback_days=8)

        # The today preview file must not reset the streak; the walk should
        # skip it and pick up yesterday's hit + dd for +2.
        assert load_streak(tmp_path) == 2, (
            f"Today's preview pick with result=None broke the backward walk "
            f"and reset streak. Expected 2, got {load_streak(tmp_path)}."
        )

    def test_shadow_pick_files_are_ignored_in_streak_walk(self, tmp_path):
        """Shadow pick files (stem like '2026-04-14.shadow') should not
        participate in streak counting — they're a separate evaluation track.
        """
        from datetime import date as date_cls, timedelta
        from bts.picks import reconcile_results, save_streak, load_streak

        yesterday_iso = (date_cls.today() - timedelta(days=1)).isoformat()

        # Main pick file — hit
        main = {
            "date": yesterday_iso, "run_time": f"{yesterday_iso}T20:00:00Z",
            "pick": {"batter_name": "M", "batter_id": 1, "team": "T", "lineup_position": 1,
                     "pitcher_name": "P", "pitcher_id": 2, "p_game_hit": 0.8,
                     "flags": [], "projected_lineup": False, "game_pk": 1,
                     "game_time": f"{yesterday_iso}T23:00:00Z"},
            "double_down": None, "runner_up": None, "result": "hit",
        }
        (tmp_path / f"{yesterday_iso}.json").write_text(json.dumps(main))

        # Shadow pick file with result=None (would previously break the walk)
        shadow = {**main, "result": None}
        (tmp_path / f"{yesterday_iso}.shadow.json").write_text(json.dumps(shadow))

        save_streak(0, tmp_path)

        with patch("bts.picks.check_hit", return_value=True):
            reconcile_results(tmp_path, lookback_days=8)

        assert load_streak(tmp_path) == 1, (
            f"Shadow file broke the walk. Expected 1, got {load_streak(tmp_path)}."
        )
