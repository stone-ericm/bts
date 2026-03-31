import json
import pytest
from bts.picks import Pick, DailyPick, save_pick, load_pick, load_streak, save_streak, update_streak
from bts.picks import pick_from_row


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
        streak=3,
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
        assert loaded.streak == 3

    def test_load_nonexistent_returns_none(self, tmp_path):
        assert load_pick("2099-01-01", tmp_path) is None

    def test_save_creates_directory(self, tmp_path):
        subdir = tmp_path / "nested" / "picks"
        daily = _sample_daily()
        save_pick(daily, subdir)
        assert (subdir / "2026-04-01.json").exists()

    def test_save_overwrites_existing(self, tmp_path):
        daily = _sample_daily(streak=3)
        save_pick(daily, tmp_path)

        updated = _sample_daily(streak=4, bluesky_posted=True, bluesky_uri="at://did:plc:xxx/post/yyy")
        save_pick(updated, tmp_path)

        loaded = load_pick("2026-04-01", tmp_path)
        assert loaded.streak == 4
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
        save_streak(10, tmp_path)
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
