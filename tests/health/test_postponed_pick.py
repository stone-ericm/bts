"""Tests for Tier-1 postponed-pick health check."""

from datetime import date

from bts.health.postponed_pick import SOURCE, check
from bts.picks import DailyPick, Pick, save_pick


def _pick(game_pk: int, name: str = "Primary") -> Pick:
    return Pick(
        batter_name=name,
        batter_id=game_pk,
        team="NYY",
        lineup_position=1,
        pitcher_name="Pitcher",
        pitcher_id=1000 + game_pk,
        p_game_hit=0.72,
        flags=[],
        projected_lineup=False,
        game_pk=game_pk,
        game_time="2026-05-05T23:05:00+00:00",
    )


def _write_daily(
    picks_dir,
    *,
    date_iso: str = "2026-05-05",
    primary_game: int = 1,
    dd_game: int | None = None,
    posted: bool = False,
) -> None:
    save_pick(
        DailyPick(
            date=date_iso,
            run_time="2026-05-05T12:00:00+00:00",
            pick=_pick(primary_game),
            double_down=_pick(dd_game, "Double Down") if dd_game is not None else None,
            runner_up=None,
            bluesky_posted=posted,
            bluesky_uri="at://example/post" if posted else None,
        ),
        picks_dir,
    )


class TestPostponedPick:
    def test_no_alert_when_pick_missing(self, tmp_path):
        alerts = check(tmp_path, today=date(2026, 5, 5))
        assert alerts == []

    def test_critical_when_primary_game_postponed(self, tmp_path, monkeypatch):
        _write_daily(tmp_path, primary_game=824362)
        monkeypatch.setattr(
            "bts.picks.get_game_statuses_detailed",
            lambda _date: {824362: {"abstract": "F", "detailed": "Postponed"}},
        )

        alerts = check(tmp_path, today=date(2026, 5, 5))

        assert len(alerts) == 1
        assert alerts[0].level == "CRITICAL"
        assert alerts[0].source == SOURCE
        assert "game_pk=824362" in alerts[0].message
        assert "detailed=Postponed" in alerts[0].message

    def test_critical_when_double_down_game_cancelled(self, tmp_path, monkeypatch):
        _write_daily(tmp_path, primary_game=1, dd_game=2)
        monkeypatch.setattr(
            "bts.picks.get_game_statuses_detailed",
            lambda _date: {
                1: {"abstract": "P", "detailed": "Preview"},
                2: {"abstract": "F", "detailed": "Cancelled"},
            },
        )

        alerts = check(tmp_path, today=date(2026, 5, 5))

        assert len(alerts) == 1
        assert alerts[0].level == "CRITICAL"
        assert "game_pk=2" in alerts[0].message
        assert "detailed=Cancelled" in alerts[0].message

    def test_critical_when_committed_game_missing_from_schedule(self, tmp_path, monkeypatch):
        _write_daily(tmp_path, primary_game=1, dd_game=2)
        monkeypatch.setattr(
            "bts.picks.get_game_statuses_detailed",
            lambda _date: {1: {"abstract": "P", "detailed": "Preview"}},
        )

        alerts = check(tmp_path, today=date(2026, 5, 5))

        assert len(alerts) == 1
        assert alerts[0].level == "CRITICAL"
        assert "reason=missing_from_schedule" in alerts[0].message
        assert "game_pk=2" in alerts[0].message

    def test_no_alert_when_pick_already_posted(self, tmp_path, monkeypatch):
        _write_daily(tmp_path, primary_game=824362, posted=True)

        def _fail(_date):
            raise AssertionError("posted picks should not perform status lookup")

        monkeypatch.setattr("bts.picks.get_game_statuses_detailed", _fail)

        alerts = check(tmp_path, today=date(2026, 5, 5))

        assert alerts == []

    def test_no_alert_when_all_committed_games_preview(self, tmp_path, monkeypatch):
        _write_daily(tmp_path, primary_game=1, dd_game=2)
        monkeypatch.setattr(
            "bts.picks.get_game_statuses_detailed",
            lambda _date: {
                1: {"abstract": "P", "detailed": "Preview"},
                2: {"abstract": "P", "detailed": "Pre-Game"},
            },
        )

        alerts = check(tmp_path, today=date(2026, 5, 5))

        assert alerts == []

    def test_warn_when_status_lookup_fails_closed(self, tmp_path, monkeypatch):
        _write_daily(tmp_path, primary_game=1)

        def _raise(_date):
            raise RuntimeError("mlb unavailable")

        monkeypatch.setattr("bts.picks.get_game_statuses_detailed", _raise)
        monkeypatch.setattr("bts.picks.get_game_statuses", _raise)

        alerts = check(tmp_path, today=date(2026, 5, 5))

        assert len(alerts) == 1
        assert alerts[0].level == "WARN"
        assert alerts[0].source == SOURCE
        assert "failed closed" in alerts[0].message
