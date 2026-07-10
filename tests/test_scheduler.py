"""Tests for the dynamic lineup scheduler."""

import json
import pytest
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


@pytest.fixture(autouse=True)
def _disable_live_detailed_status_lookup():
    with patch("bts.picks.get_game_statuses_detailed", side_effect=RuntimeError("detailed unavailable")):
        yield


def _game(game_pk: int, time_et: str, team_away: str = "NYM", team_home: str = "ATL",
          date: str | None = None):
    """Build a mock MLB schedule game entry."""
    date = date or datetime.now(ET).strftime("%Y-%m-%d")
    et_dt = datetime.strptime(f"{date} {time_et}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
    utc_iso = et_dt.astimezone(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z")
    return {
        "gamePk": game_pk,
        "gameDate": utc_iso,
        "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
        "teams": {
            "away": {"team": {"name": team_away}},
            "home": {"team": {"name": team_home}},
        },
    }


def _daily_pick(
    game_pk: int,
    *,
    batter_name: str = "Díaz",
    double_down_game_pk: int | None = None,
    double_down_projected: bool = False,
    bluesky_posted: bool = False,
):
    from bts.picks import Pick, DailyPick

    pick = Pick(
        batter_name=batter_name, batter_id=1, team="TB",
        lineup_position=1, pitcher_name="Abel", pitcher_id=2,
        p_game_hit=0.72, flags=[], projected_lineup=False,
        game_pk=game_pk, game_time="2026-04-04T23:10:00Z",
    )
    double_down = None
    if double_down_game_pk is not None:
        double_down = Pick(
            batter_name="Double", batter_id=3, team="MIN",
            lineup_position=1, pitcher_name="Gray", pitcher_id=4,
            p_game_hit=0.71, flags=["PROJECTED lineup"] if double_down_projected else [],
            projected_lineup=double_down_projected,
            game_pk=double_down_game_pk, game_time="2026-04-04T23:15:00Z",
        )
    return DailyPick(
        date="2026-04-04",
        run_time="2026-04-04T22:00:00+00:00",
        pick=pick,
        double_down=double_down,
        runner_up=None,
        bluesky_posted=bluesky_posted,
        bluesky_uri="at://did:plc:test/post/123" if bluesky_posted else None,
    )


class TestComputeRunTimes:
    def test_single_game(self):
        from bts.scheduler import compute_run_times
        games = [_game(100, "19:05")]
        runs = compute_run_times(games, offset_min=45, cluster_min=10)
        assert len(runs) == 1
        assert runs[0]["time_et"].hour == 18
        assert runs[0]["time_et"].minute == 20
        assert runs[0]["game_pks"] == [100]

    def test_clusters_nearby_games(self):
        from bts.scheduler import compute_run_times
        games = [
            _game(100, "19:05"),
            _game(200, "19:10"),
            _game(300, "19:15"),
        ]
        runs = compute_run_times(games, offset_min=45, cluster_min=10)
        assert len(runs) == 1
        assert sorted(runs[0]["game_pks"]) == [100, 200, 300]

    def test_separates_distant_games(self):
        from bts.scheduler import compute_run_times
        games = [
            _game(100, "13:10"),
            _game(200, "19:05"),
        ]
        runs = compute_run_times(games, offset_min=45, cluster_min=10)
        assert len(runs) == 2
        assert runs[0]["game_pks"] == [100]
        assert runs[1]["game_pks"] == [200]


class TestOptionalHealthPath:
    def test_disables_when_not_configured(self):
        from bts.scheduler import _optional_health_path

        assert _optional_health_path(None) is None

    def test_can_disable_with_empty_string_or_false(self):
        from bts.scheduler import _optional_health_path

        assert _optional_health_path("") is None
        assert _optional_health_path(False) is None

    def test_uses_configured_path(self, tmp_path):
        from bts.scheduler import _optional_health_path

        configured = tmp_path / "configured"
        assert _optional_health_path(str(configured)) == configured


class TestDetectDoubleheaderGame2:
    def test_finds_doubleheader(self):
        from bts.scheduler import detect_doubleheader_game2s
        games = [
            _game(100, "13:10", "NYM", "ATL"),
            _game(200, "19:05", "NYM", "ATL"),
            _game(300, "19:10", "LAD", "SF"),
        ]
        dh2s = detect_doubleheader_game2s(games)
        assert dh2s == {200}

    def test_no_doubleheader(self):
        from bts.scheduler import detect_doubleheader_game2s
        games = [
            _game(100, "19:05", "NYM", "ATL"),
            _game(200, "19:10", "LAD", "SF"),
        ]
        dh2s = detect_doubleheader_game2s(games)
        assert dh2s == set()


class TestComputeWakeUpTime:
    def test_default_when_no_early_games(self):
        from bts.scheduler import compute_wakeup_time
        games = [_game(100, "19:05")]
        wakeup = compute_wakeup_time(games, default_hour_et=10, early_buffer_min=60)
        assert wakeup.hour == 10
        assert wakeup.minute == 0

    def test_default_uses_game_date_for_tomorrow_lookahead(self):
        from bts.scheduler import compute_wakeup_time
        games = [_game(100, "19:05", date="2026-04-05")]
        wakeup = compute_wakeup_time(games, default_hour_et=10, early_buffer_min=60)
        assert wakeup.date().isoformat() == "2026-04-05"
        assert wakeup.hour == 10
        assert wakeup.minute == 0

    def test_early_wakeup_for_international_game(self):
        from bts.scheduler import compute_wakeup_time
        games = [_game(100, "06:10"), _game(200, "19:05")]
        wakeup = compute_wakeup_time(games, default_hour_et=10, early_buffer_min=60)
        assert wakeup.hour == 5
        assert wakeup.minute == 10


class TestCheckConfirmedLineups:
    @patch("bts.scheduler.retry_urlopen")
    def test_detects_both_sides_confirmed(self, mock_urlopen):
        from bts.scheduler import check_confirmed_lineups

        feed = {
            "liveData": {"boxscore": {"teams": {
                "away": {"players": {
                    "ID123": {"battingOrder": "100", "person": {"fullName": "A"}},
                    "ID456": {"battingOrder": "200", "person": {"fullName": "B"}},
                }},
                "home": {"players": {
                    "ID789": {"battingOrder": "100", "person": {"fullName": "C"}},
                }},
            }}},
        }
        mock_urlopen.return_value.read.return_value = json.dumps(feed).encode()

        result = check_confirmed_lineups([111])
        assert result == {111: {"home", "away"}}

    @patch("bts.scheduler.retry_urlopen")
    def test_detects_only_away_confirmed(self, mock_urlopen):
        from bts.scheduler import check_confirmed_lineups

        feed = {
            "liveData": {"boxscore": {"teams": {
                "away": {"players": {
                    "ID123": {"battingOrder": "100", "person": {"fullName": "A"}},
                }},
                "home": {"players": {}},
            }}},
        }
        mock_urlopen.return_value.read.return_value = json.dumps(feed).encode()

        result = check_confirmed_lineups([111])
        assert result == {111: {"away"}}

    @patch("bts.scheduler.retry_urlopen")
    def test_detects_no_lineup(self, mock_urlopen):
        from bts.scheduler import check_confirmed_lineups

        feed = {
            "liveData": {"boxscore": {"teams": {
                "away": {"players": {}},
                "home": {"players": {}},
            }}},
        }
        mock_urlopen.return_value.read.return_value = json.dumps(feed).encode()

        result = check_confirmed_lineups([111])
        assert result == {111: set()}

    @patch("bts.scheduler.retry_urlopen")
    def test_counts_both_sides_as_two(self, mock_urlopen):
        from bts.scheduler import count_new_confirmations

        feed_confirmed = {
            "liveData": {"boxscore": {"teams": {
                "away": {"players": {"ID1": {"battingOrder": "100", "person": {"fullName": "A"}}}},
                "home": {"players": {"ID2": {"battingOrder": "100", "person": {"fullName": "B"}}}},
            }}},
        }
        mock_urlopen.return_value.read.return_value = json.dumps(feed_confirmed).encode()

        previously_confirmed: set[tuple[int, str]] = set()
        new_count = count_new_confirmations([111], previously_confirmed)
        assert new_count == 2  # both sides just confirmed
        assert (111, "home") in previously_confirmed
        assert (111, "away") in previously_confirmed

    @patch("bts.scheduler.retry_urlopen")
    def test_counts_second_side_as_new_confirmation(self, mock_urlopen):
        """Regression test: game had one side confirmed, then gets the other.
        The old game-level tracking returned 0 new confirmations, hiding that
        the prediction inputs had changed. Now we count the second side as +1.
        """
        from bts.scheduler import count_new_confirmations

        # Initial state: away side already confirmed from a previous check
        previously_confirmed: set[tuple[int, str]] = {(111, "away")}

        # Now the home side has posted its lineup too
        feed = {
            "liveData": {"boxscore": {"teams": {
                "away": {"players": {"ID1": {"battingOrder": "100", "person": {"fullName": "A"}}}},
                "home": {"players": {"ID2": {"battingOrder": "100", "person": {"fullName": "B"}}}},
            }}},
        }
        mock_urlopen.return_value.read.return_value = json.dumps(feed).encode()

        new_count = count_new_confirmations([111], previously_confirmed)
        assert new_count == 1  # only the home side is new
        assert (111, "home") in previously_confirmed
        assert (111, "away") in previously_confirmed  # still there


class TestSchedulerState:
    def test_save_and_load_roundtrip(self, tmp_path):
        from bts.scheduler import SchedulerState, save_state, load_state

        state = SchedulerState(
            date="2026-04-03",
            schedule_fetched_at="2026-04-03T10:00:00-04:00",
            games=[{"game_pk": 100, "game_time_et": "2026-04-03T19:05:00-04:00",
                     "lineup_confirmed": False, "is_doubleheader_game2": False}],
            confirmed_game_pks=[],
            runs_completed=[],
            pick_locked=False,
            pick_locked_at=None,
            result_status=None,
            next_wakeup=None,
        )
        save_state(state, tmp_path)

        loaded = load_state("2026-04-03", tmp_path)
        assert loaded is not None
        assert loaded.date == "2026-04-03"
        assert len(loaded.games) == 1
        assert loaded.pick_locked is False

    def test_save_and_load_roundtrip_preserves_analytics_jobs(self, tmp_path):
        from bts.scheduler import SchedulerState, save_state, load_state

        state = SchedulerState(
            date="2026-04-03",
            schedule_fetched_at="2026-04-03T10:00:00-04:00",
            games=[],
            confirmed_game_pks=[],
            runs_completed=[],
            pick_locked=True,
            pick_locked_at="2026-04-03T12:00:00-04:00",
            result_status=None,
            next_wakeup=None,
            analytics_jobs={
                "shadow": {"status": "dispatched", "updated_at": "now"},
            },
        )
        save_state(state, tmp_path)

        loaded = load_state("2026-04-03", tmp_path)
        assert loaded is not None
        assert loaded.analytics_jobs == state.analytics_jobs

    def test_save_state_merges_existing_analytics_jobs(self, tmp_path):
        from bts.scheduler import SchedulerState, save_state, load_state

        original = SchedulerState(
            date="2026-04-03",
            schedule_fetched_at="2026-04-03T10:00:00-04:00",
            games=[],
            confirmed_game_pks=[],
            runs_completed=[],
            pick_locked=True,
            pick_locked_at="2026-04-03T12:00:00-04:00",
            result_status=None,
            next_wakeup=None,
            analytics_jobs={
                "shadow": {"status": "dispatched", "updated_at": "now"},
            },
        )
        save_state(original, tmp_path)

        stale_in_memory = SchedulerState(
            date="2026-04-03",
            schedule_fetched_at="2026-04-03T10:05:00-04:00",
            games=[],
            confirmed_game_pks=[],
            runs_completed=[],
            pick_locked=True,
            pick_locked_at="2026-04-03T12:00:00-04:00",
            result_status="final",
            next_wakeup=None,
            analytics_jobs=None,
        )
        save_state(stale_in_memory, tmp_path)

        loaded = load_state("2026-04-03", tmp_path)
        assert loaded is not None
        assert loaded.analytics_jobs == original.analytics_jobs
        assert loaded.result_status == "final"

    def test_load_returns_none_when_missing(self, tmp_path):
        from bts.scheduler import load_state

        assert load_state("2026-04-03", tmp_path) is None

    @patch("bts.scheduler.predict_local_shadow")
    def test_shadow_skips_prior_dispatched_attempt(self, mock_predict, tmp_path):
        from bts.scheduler import (
            SchedulerState,
            load_state,
            save_state,
            _run_shadow_prediction,
        )

        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()
        state = SchedulerState(
            date="2026-04-03",
            schedule_fetched_at="2026-04-03T10:00:00-04:00",
            games=[],
            confirmed_game_pks=[],
            runs_completed=[],
            pick_locked=True,
            pick_locked_at="2026-04-03T12:00:00-04:00",
            result_status=None,
            next_wakeup=None,
            analytics_jobs={
                "shadow": {"status": "dispatched", "updated_at": "now"},
            },
        )
        save_state(state, picks_dir)
        config = {
            "orchestrator": {
                "picks_dir": str(picks_dir),
                "data_dir": str(tmp_path / "data"),
                "models_dir": str(tmp_path / "models"),
                "heartbeat_path": str(tmp_path / ".heartbeat"),
            },
        }

        _run_shadow_prediction(config, "2026-04-03", "Prod Pick")

        mock_predict.assert_not_called()
        updated = load_state("2026-04-03", picks_dir)
        assert updated is not None
        assert updated.analytics_jobs is not None
        assert updated.analytics_jobs["shadow"]["status"] == "failed"
        assert updated.analytics_jobs["shadow"]["reason"] == (
            "prior_dispatched_without_artifact"
        )
        assert updated.analytics_jobs["shadow"]["dispatched_at"] == "now"


class TestSchedulerRun:
    @patch("bts.scheduler.check_confirmed_lineups")
    @patch("bts.orchestrator.run_cascade")
    def test_runs_predictions_even_with_no_new_lineups(self, mock_cascade, mock_lineups, tmp_path):
        from bts.scheduler import run_single_check

        mock_lineups.return_value = {100: set()}
        mock_cascade.return_value = (None, None)

        result = run_single_check(
            date="2026-04-03",
            all_game_pks=[100],
            confirmed_sides=set(),
            config={"orchestrator": {"picks_dir": str(tmp_path)}, "tiers": []},
            early_lock_gap=0.03,
        )
        assert result["skipped"] is False
        assert result["new_lineups"] == 0
        mock_cascade.assert_called_once()

    @patch("bts.scheduler.check_confirmed_lineups")
    @patch("bts.orchestrator.run_cascade")
    @patch("bts.strategy.get_game_statuses", return_value={100: "P"})
    @patch("bts.picks.get_game_statuses", return_value={100: "P"})
    @patch("bts.strategy._load_mdp", return_value=None)
    def test_triggers_prediction_on_new_lineup(
        self, _mdp, _sched_statuses, _strat_statuses, mock_cascade, mock_lineups, tmp_path
    ):
        import pandas as pd
        from bts.scheduler import run_single_check

        mock_lineups.return_value = {100: {"home", "away"}}
        mock_cascade.return_value = (
            pd.DataFrame([{
                "batter_name": "Test", "batter_id": 1, "team": "NYM",
                "lineup": 1, "pitcher_name": "P", "pitcher_id": 2,
                "game_pk": 100, "game_time": "2026-04-03T23:05:00Z",
                "p_hit_pa": 0.30, "p_game_hit": 0.82, "flags": "",
            }]),
            "mac",
        )

        result = run_single_check(
            date="2026-04-03",
            all_game_pks=[100],
            confirmed_sides=set(),
            config={
                "orchestrator": {"picks_dir": str(tmp_path)},
                "tiers": [{"name": "mac", "ssh_host": "mac", "bts_dir": "/bts", "timeout_min": 5}],
            },
            early_lock_gap=0.03,
        )
        assert result["skipped"] is False
        # Both sides of game 100 just confirmed → 2 new team-level confirmations.
        # (Previously this was game-level and returned 1; now it reflects the
        # actual granularity the prediction pipeline sees.)
        assert result["new_lineups"] == 2
        # Detailed-status lookup is disabled by the test fixture. The scheduler
        # still caches a best-available pick, but the lock decision fails closed.
        assert result["pick_name"] == "Test"
        assert result["should_post"] is False

    @patch("bts.scheduler.check_confirmed_lineups")
    @patch("bts.orchestrator.run_cascade")
    @patch("bts.picks.get_game_statuses_detailed", return_value={
        100: {"abstract": "P", "detailed": "Pre-Game"},
        200: {"abstract": "P", "detailed": "Postponed"},
    })
    @patch("bts.strategy.get_game_statuses", return_value={100: "P", 200: "P"})
    @patch("bts.picks.get_game_statuses", return_value={100: "P", 200: "P"})
    @patch("bts.strategy._load_mdp", return_value=None)
    def test_should_lock_excludes_postponed_games(
        self, _mdp, _sched_statuses, _strat_statuses, _detailed_statuses,
        mock_cascade, mock_lineups, tmp_path
    ):
        """Projected picks from postponed/finished games shouldn't block locking.

        Reproduces the 2026-04-04 bug shape with the more dangerous MLB shape:
        a postponed game that still appears abstract-preview must not leave its
        projected batters in the should_lock gap check.
        """
        import pandas as pd
        from bts.scheduler import run_single_check

        mock_lineups.return_value = {100: {"home", "away"}, 200: {"home", "away"}}
        mock_cascade.return_value = (
            pd.DataFrame([
                {
                    "batter_name": "Díaz", "batter_id": 1, "team": "TB",
                    "lineup": 1, "pitcher_name": "Abel", "pitcher_id": 2,
                    "game_pk": 100, "game_time": "2026-04-04T23:10:00Z",
                    "p_hit_pa": 0.30, "p_game_hit": 0.82, "flags": "",
                },
                {
                    "batter_name": "Kwan", "batter_id": 3, "team": "CLE",
                    "lineup": 1, "pitcher_name": "Imanaga", "pitcher_id": 4,
                    "game_pk": 200, "game_time": "2026-04-04T23:15:00Z",
                    "p_hit_pa": 0.27, "p_game_hit": 0.80, "flags": "PROJECTED lineup",
                },
            ]),
            "mac",
        )

        result = run_single_check(
            date="2026-04-04",
            all_game_pks=[100, 200],
            confirmed_sides=set(),
            config={
                "orchestrator": {"picks_dir": str(tmp_path)},
                "tiers": [{"name": "mac", "ssh_host": "mac", "bts_dir": "/bts", "timeout_min": 5}],
            },
            early_lock_gap=0.03,
        )

        # Game 200 is Postponed — its projected batter (Kwan, 0.80)
        # should be excluded from the should_lock gap check. Without the fix,
        # the gap (0.82 - 0.80 = 0.02 < 0.03) would block locking.
        # With the fix, only game 100's picks remain — all confirmed → lock.
        assert result["should_post"] is True
        assert result["pick_name"] == "Díaz"
        assert result["pick_p"] is not None

    @patch("bts.scheduler.check_confirmed_lineups")
    @patch("bts.health.alert.dispatch_dm_for_health_alerts")
    @patch("bts.orchestrator.run_and_pick")
    def test_contest_state_error_fails_closed_and_alerts(
        self, mock_run_and_pick, mock_alert, mock_lineups, tmp_path
    ):
        from bts.contest_state import ContestStateError
        from bts.scheduler import run_single_check

        mock_lineups.return_value = {100: {"home", "away"}}
        mock_run_and_pick.side_effect = ContestStateError("contest state bad")

        result = run_single_check(
            date="2026-04-04",
            all_game_pks=[100],
            confirmed_sides=set(),
            config={
                "orchestrator": {"picks_dir": str(tmp_path)},
                "bluesky": {"dm_recipient": "did:plc:test"},
                "health_checks": {"contest_state_expected": True},
            },
            early_lock_gap=0.03,
        )

        assert result["should_post"] is False
        assert result["pick_result"] is None
        assert result["pick_name"] is None
        mock_alert.assert_called_once()

    @patch("bts.picks.get_game_statuses", return_value={100: "P", 200: "P"})
    def test_lock_decision_fails_closed_when_detailed_status_unavailable(
        self, mock_coarse_statuses
    ):
        import pandas as pd
        from bts.scheduler import _lock_decision_from_predictions

        predictions = pd.DataFrame([
            {
                "batter_name": "Top", "batter_id": 1, "team": "NYM",
                "lineup": 1, "pitcher_name": "P", "pitcher_id": 2,
                "game_pk": 100, "game_time": "2026-04-03T23:05:00Z",
                "p_hit_pa": 0.30, "p_game_hit": 0.82, "flags": "",
            },
            {
                "batter_name": "Projected", "batter_id": 3, "team": "ATL",
                "lineup": 1, "pitcher_name": "P", "pitcher_id": 4,
                "game_pk": 200, "game_time": "2026-04-03T23:10:00Z",
                "p_hit_pa": 0.29, "p_game_hit": 0.81, "flags": "PROJECTED lineup",
            },
        ])

        should_post, best_projected, ungated = _lock_decision_from_predictions(
            predictions,
            _daily_pick(100),
            "2026-04-03",
            early_lock_gap=0.03,
        )

        assert should_post is False
        assert best_projected is None
        assert ungated is False  # status failure blocks pre-gate too
        mock_coarse_statuses.assert_not_called()

    # --- F2 (2026-07-09 audit): the scheduler-level case where the projected
    # contender IS the selected double-down. Confirmed .72 primary with a huge
    # gap previously locked and committed the projected DD 51-133min early.

    @patch("bts.picks.get_game_statuses_detailed", return_value={
        100: {"abstract": "P", "detailed": "Scheduled"},
        200: {"abstract": "P", "detailed": "Scheduled"},
    })
    def test_projected_double_down_blocks_lock_decision(self, _detailed):
        import pandas as pd
        from bts.scheduler import _lock_decision_from_predictions

        predictions = pd.DataFrame([
            {
                "batter_name": "Díaz", "batter_id": 1, "team": "TB",
                "lineup": 1, "pitcher_name": "Abel", "pitcher_id": 2,
                "game_pk": 100, "game_time": "2026-04-04T23:10:00Z",
                "p_hit_pa": 0.30, "p_game_hit": 0.72, "flags": "",
            },
            {
                "batter_name": "Double", "batter_id": 3, "team": "MIN",
                "lineup": 1, "pitcher_name": "Gray", "pitcher_id": 4,
                "game_pk": 200, "game_time": "2026-04-04T23:15:00Z",
                "p_hit_pa": 0.28, "p_game_hit": 0.61, "flags": "PROJECTED lineup",
            },
        ])
        daily = _daily_pick(100, double_down_game_pk=200, double_down_projected=True)

        should_post, _, _ = _lock_decision_from_predictions(
            predictions, daily, "2026-04-04", early_lock_gap=0.03,
        )
        assert should_post is False

        # identical slate with the DD confirmed locks as before
        predictions_confirmed = predictions.copy()
        predictions_confirmed.loc[1, "flags"] = ""
        daily_confirmed = _daily_pick(100, double_down_game_pk=200)
        should_post, _, _ = _lock_decision_from_predictions(
            predictions_confirmed, daily_confirmed, "2026-04-04", early_lock_gap=0.03,
        )
        assert should_post is True

    @patch("bts.picks.get_game_statuses_detailed", return_value={
        100: {"abstract": "P", "detailed": "Scheduled"},
        200: {"abstract": "P", "detailed": "Scheduled"},
    })
    def test_lock_decision_reports_ungated_value(self, _detailed):
        # Codex review L1: when ONLY the DD gate blocks, the in-loop fallback
        # must know the pre-gate decision so it delivers instead of archiving
        # the day's only deliverable pair on a stale pending-window flag.
        import pandas as pd
        from bts.scheduler import _lock_decision_from_predictions

        predictions = pd.DataFrame([
            {
                "batter_name": "Díaz", "batter_id": 1, "team": "TB",
                "lineup": 1, "pitcher_name": "Abel", "pitcher_id": 2,
                "game_pk": 100, "game_time": "2026-04-04T23:10:00Z",
                "p_hit_pa": 0.30, "p_game_hit": 0.72, "flags": "",
            },
            {
                "batter_name": "Double", "batter_id": 3, "team": "MIN",
                "lineup": 1, "pitcher_name": "Gray", "pitcher_id": 4,
                "game_pk": 200, "game_time": "2026-04-04T23:15:00Z",
                "p_hit_pa": 0.28, "p_game_hit": 0.61, "flags": "PROJECTED lineup",
            },
        ])
        daily = _daily_pick(100, double_down_game_pk=200, double_down_projected=True)
        should_post, _, ungated = _lock_decision_from_predictions(
            predictions, daily, "2026-04-04", early_lock_gap=0.03,
        )
        assert should_post is False
        assert ungated is True  # gap passes, primary confirmed — gate-only block

    @patch("bts.picks.get_game_statuses_detailed", return_value={
        100: {"abstract": "F", "detailed": "Postponed"},
        200: {"abstract": "P", "detailed": "Scheduled"},
    })
    def test_lock_blocked_when_selected_slot_game_unavailable(self, _detailed):
        # Codex round-2 R4: the pick was selected while its game was Preview;
        # the game went postponed before the lock decision. Its row vanishes
        # from the contender set (no projected contender left), so the stale
        # confirmed-lineup flag would have locked it — fail closed instead.
        import pandas as pd
        from bts.scheduler import _lock_decision_from_predictions

        predictions = pd.DataFrame([
            {
                "batter_name": "Double", "batter_id": 3, "team": "MIN",
                "lineup": 1, "pitcher_name": "Gray", "pitcher_id": 4,
                "game_pk": 200, "game_time": "2026-04-04T23:15:00Z",
                "p_hit_pa": 0.28, "p_game_hit": 0.61, "flags": "",
            },
        ])
        daily = _daily_pick(100)  # primary's game 100 is postponed
        should_post, best_projected, ungated = _lock_decision_from_predictions(
            predictions, daily, "2026-04-04", early_lock_gap=0.03,
        )
        assert should_post is False
        assert ungated is False

    def test_should_defer_at_fallback_gate_only_block_delivers(self):
        from bts.scheduler import _should_defer_at_fallback

        # gate-only block at the deadline → deliver (no defer), even with
        # pending future windows: deferring would archive the only pair
        assert _should_defer_at_fallback(
            should_post=False, should_post_ungated=True,
            has_pending_future_window=True,
        ) is False
        # genuine pre-gate block (primary projected / gap fail) → defer
        assert _should_defer_at_fallback(
            should_post=False, should_post_ungated=False,
            has_pending_future_window=True,
        ) is True
        # unknown ungated (cached/error refresh paths) → preserve old behavior
        assert _should_defer_at_fallback(
            should_post=False, should_post_ungated=None,
            has_pending_future_window=True,
        ) is True
        # no pending windows → never defer
        assert _should_defer_at_fallback(
            should_post=False, should_post_ungated=False,
            has_pending_future_window=False,
        ) is False
        # should_post not False → never defer
        assert _should_defer_at_fallback(
            should_post=None, should_post_ungated=None,
            has_pending_future_window=True,
        ) is False

    @patch("bts.scheduler.check_confirmed_lineups")
    @patch("bts.picks.get_game_statuses_detailed", return_value={
        100: {"abstract": "L", "detailed": "In Progress"},
    })
    @patch("bts.picks.get_game_statuses", return_value={100: "L"})
    def test_short_circuits_when_pick_locked(self, _statuses, _detailed_statuses, mock_lineups, tmp_path):
        """Skip the expensive SSH cascade when pick is already locked."""
        from bts.scheduler import run_single_check
        from bts.picks import save_pick

        mock_lineups.return_value = {100: {"home", "away"}}

        # Pre-save a pick whose game has started (status L)
        save_pick(_daily_pick(100), tmp_path)

        result = run_single_check(
            date="2026-04-04",
            all_game_pks=[100],
            confirmed_sides=set(),
            config={
                "orchestrator": {"picks_dir": str(tmp_path)},
                "tiers": [{"name": "mac", "ssh_host": "mac", "bts_dir": "/bts", "timeout_min": 5}],
            },
            early_lock_gap=0.03,
        )

        assert result["pick_result"].locked is True
        assert result["pick_name"] == "Díaz"
        # No cascade should have been attempted — check_confirmed_lineups
        # was called but run_cascade was NOT (not even imported/mocked)

    @patch("bts.scheduler.check_confirmed_lineups")
    @patch("bts.orchestrator.run_cascade")
    @patch("bts.picks.get_game_statuses_detailed", return_value={
        100: {"abstract": "F", "detailed": "Postponed"},
        200: {"abstract": "P", "detailed": "Pre-Game"},
    })
    @patch("bts.strategy.get_game_statuses", return_value={100: "F", 200: "P"})
    @patch("bts.picks.get_game_statuses", return_value={100: "F", 200: "P"})
    @patch("bts.strategy._load_mdp", return_value=None)
    def test_primary_postponed_regenerates(
        self, _mdp, _pick_statuses, _strategy_statuses, _detailed_statuses,
        mock_cascade, mock_lineups, tmp_path
    ):
        import pandas as pd
        from bts.scheduler import run_single_check
        from bts.picks import save_pick

        save_pick(_daily_pick(100, batter_name="Stale"), tmp_path)
        mock_lineups.return_value = {200: {"home", "away"}}
        mock_cascade.return_value = (
            pd.DataFrame([{
                "batter_name": "Fresh", "batter_id": 10, "team": "NYM",
                "lineup": 1, "pitcher_name": "P", "pitcher_id": 20,
                "game_pk": 200, "game_time": "2026-04-04T23:05:00Z",
                "p_hit_pa": 0.30, "p_game_hit": 0.84, "flags": "",
            }]),
            "mac",
        )

        result = run_single_check(
            date="2026-04-04",
            all_game_pks=[200],
            confirmed_sides=set(),
            config={
                "orchestrator": {"picks_dir": str(tmp_path)},
                "tiers": [{"name": "mac", "ssh_host": "mac", "bts_dir": "/bts", "timeout_min": 5}],
            },
            early_lock_gap=0.03,
        )

        assert result["pick_result"].locked is False
        assert result["pick_name"] == "Fresh"

    @patch("bts.scheduler.check_confirmed_lineups")
    @patch("bts.orchestrator.run_cascade")
    @patch("bts.picks.get_game_statuses_detailed", return_value={
        100: {"abstract": "P", "detailed": "Pre-Game"},
        200: {"abstract": "F", "detailed": "Postponed"},
        300: {"abstract": "P", "detailed": "Pre-Game"},
    })
    @patch("bts.strategy.get_game_statuses", return_value={100: "P", 200: "F", 300: "P"})
    @patch("bts.picks.get_game_statuses", return_value={100: "P", 200: "F", 300: "P"})
    @patch("bts.strategy._load_mdp", return_value=None)
    def test_double_down_postponed_regenerates_whole_pick(
        self, _mdp, _pick_statuses, _strategy_statuses, _detailed_statuses,
        mock_cascade, mock_lineups, tmp_path
    ):
        import pandas as pd
        from bts.scheduler import run_single_check
        from bts.picks import save_pick

        save_pick(_daily_pick(100, double_down_game_pk=200), tmp_path)
        mock_lineups.return_value = {300: {"home", "away"}}
        mock_cascade.return_value = (
            pd.DataFrame([{
                "batter_name": "Replacement", "batter_id": 10, "team": "NYM",
                "lineup": 1, "pitcher_name": "P", "pitcher_id": 20,
                "game_pk": 300, "game_time": "2026-04-04T23:05:00Z",
                "p_hit_pa": 0.30, "p_game_hit": 0.84, "flags": "",
            }]),
            "mac",
        )

        result = run_single_check(
            date="2026-04-04",
            all_game_pks=[300],
            confirmed_sides=set(),
            config={
                "orchestrator": {"picks_dir": str(tmp_path)},
                "tiers": [{"name": "mac", "ssh_host": "mac", "bts_dir": "/bts", "timeout_min": 5}],
            },
            early_lock_gap=0.03,
        )

        assert result["pick_result"].locked is False
        assert result["pick_name"] == "Replacement"

    @patch("bts.scheduler.check_confirmed_lineups")
    @patch("bts.orchestrator.run_cascade")
    @patch("bts.picks.get_game_statuses_detailed", return_value={
        200: {"abstract": "P", "detailed": "Pre-Game"},
    })
    @patch("bts.strategy.get_game_statuses", return_value={200: "P"})
    @patch("bts.picks.get_game_statuses", return_value={200: "P"})
    @patch("bts.strategy._load_mdp", return_value=None)
    def test_missing_primary_game_regenerates(
        self, _mdp, _pick_statuses, _strategy_statuses, _detailed_statuses,
        mock_cascade, mock_lineups, tmp_path
    ):
        import pandas as pd
        from bts.scheduler import run_single_check
        from bts.picks import save_pick

        save_pick(_daily_pick(100, batter_name="Missing"), tmp_path)
        mock_lineups.return_value = {200: {"home", "away"}}
        mock_cascade.return_value = (
            pd.DataFrame([{
                "batter_name": "Fresh Missing Replacement", "batter_id": 10, "team": "NYM",
                "lineup": 1, "pitcher_name": "P", "pitcher_id": 20,
                "game_pk": 200, "game_time": "2026-04-04T23:05:00Z",
                "p_hit_pa": 0.30, "p_game_hit": 0.84, "flags": "",
            }]),
            "mac",
        )

        result = run_single_check(
            date="2026-04-04",
            all_game_pks=[200],
            confirmed_sides=set(),
            config={
                "orchestrator": {"picks_dir": str(tmp_path)},
                "tiers": [{"name": "mac", "ssh_host": "mac", "bts_dir": "/bts", "timeout_min": 5}],
            },
            early_lock_gap=0.03,
        )

        assert result["pick_result"].locked is False
        assert result["pick_name"] == "Fresh Missing Replacement"

    @patch("bts.scheduler.check_confirmed_lineups")
    @patch("bts.orchestrator.run_cascade")
    @patch("bts.picks.get_game_statuses_detailed", return_value={
        100: {"abstract": "F", "detailed": "Postponed"},
    })
    def test_bluesky_posted_locks_even_if_postponed(
        self, _detailed_statuses, mock_cascade, mock_lineups, tmp_path
    ):
        from bts.scheduler import run_single_check
        from bts.picks import save_pick

        save_pick(_daily_pick(100, bluesky_posted=True), tmp_path)
        mock_lineups.return_value = {100: {"home", "away"}}

        result = run_single_check(
            date="2026-04-04",
            all_game_pks=[100],
            confirmed_sides=set(),
            config={
                "orchestrator": {"picks_dir": str(tmp_path)},
                "tiers": [{"name": "mac", "ssh_host": "mac", "bts_dir": "/bts", "timeout_min": 5}],
            },
            early_lock_gap=0.03,
        )

        assert result["pick_result"].locked is True
        assert result["pick_name"] == "Díaz"
        mock_cascade.assert_not_called()


class TestPollResults:
    @patch("bts.scheduler.retry_urlopen")
    def test_returns_final(self, mock_urlopen):
        from bts.scheduler import poll_game_result

        mock_urlopen.return_value.read.return_value = json.dumps({
            "gameData": {"status": {
                "abstractGameCode": "F",
                "detailedState": "Final",
            }},
        }).encode()

        status = poll_game_result(12345)
        assert status == "final"

    @patch("bts.scheduler.retry_urlopen")
    def test_returns_live(self, mock_urlopen):
        from bts.scheduler import poll_game_result

        mock_urlopen.return_value.read.return_value = json.dumps({
            "gameData": {"status": {
                "abstractGameCode": "L",
                "detailedState": "In Progress",
            }},
        }).encode()

        status = poll_game_result(12345)
        assert status == "live"

    @patch("bts.scheduler.retry_urlopen")
    def test_returns_suspended(self, mock_urlopen):
        from bts.scheduler import poll_game_result

        mock_urlopen.return_value.read.return_value = json.dumps({
            "gameData": {"status": {
                "abstractGameCode": "L",
                "detailedState": "Suspended",
            }},
        }).encode()

        status = poll_game_result(12345)
        assert status == "suspended"

    @patch("bts.scheduler.retry_urlopen")
    def test_returns_final_for_postponed_preview(self, mock_urlopen):
        from bts.scheduler import poll_game_result

        mock_urlopen.return_value.read.return_value = json.dumps({
            "gameData": {"status": {
                "abstractGameCode": "P",
                "detailedState": "Postponed",
            }},
        }).encode()

        status = poll_game_result(12345)
        assert status == "final"

    @patch("bts.scheduler.poll_game_result", return_value="final")
    @patch("bts.picks.get_game_statuses_detailed")
    @patch("bts.picks.check_hit")
    def test_run_result_polling_voids_primary_and_scores_double_once(
        self, mock_check, mock_statuses, _mock_poll, tmp_path,
    ):
        from bts.picks import load_pick, load_streak, save_pick, save_streak
        from bts.scheduler import run_result_polling

        save_pick(_daily_pick(100, double_down_game_pk=200), tmp_path)
        save_streak(3, tmp_path)
        mock_statuses.return_value = {
            100: {"abstract": "F", "detailed": "Postponed"},
            200: {"abstract": "F", "detailed": "Final"},
        }
        mock_check.return_value = True

        status = run_result_polling(100, "2026-04-04", tmp_path, cap_hour_et=10)

        assert status == "final"
        daily = load_pick("2026-04-04", tmp_path)
        assert daily.result == "hit"
        assert daily.slot_results == {"pick": "void", "double_down": "hit"}
        assert load_streak(tmp_path) == 4
        mock_check.assert_called_once()

    @patch("bts.scheduler.poll_game_result")
    @patch("bts.picks.get_game_statuses_detailed")
    @patch("bts.picks.check_hit")
    def test_run_result_polling_uses_schedule_void_when_live_feed_preview(
        self, mock_check, mock_statuses, mock_poll, tmp_path,
    ):
        from bts.picks import load_pick, load_streak, save_pick, save_streak
        from bts.scheduler import run_result_polling

        save_pick(_daily_pick(100, double_down_game_pk=200), tmp_path)
        save_streak(3, tmp_path)
        mock_poll.side_effect = lambda game_pk: "preview" if game_pk == 100 else "final"
        mock_statuses.return_value = {
            100: {"abstract": "F", "detailed": "Postponed"},
            200: {"abstract": "F", "detailed": "Final"},
        }
        mock_check.return_value = True

        status = run_result_polling(100, "2026-04-04", tmp_path, cap_hour_et=10)

        assert status == "final"
        daily = load_pick("2026-04-04", tmp_path)
        assert daily.result == "hit"
        assert daily.slot_results == {"pick": "void", "double_down": "hit"}
        assert load_streak(tmp_path) == 4
        mock_check.assert_called_once()

    @patch("bts.scheduler._now_et")
    def test_run_result_polling_cap_does_not_overwrite_resolved_result(
        self, mock_now, tmp_path,
    ):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        from bts.picks import load_pick, save_pick
        from bts.scheduler import run_result_polling

        daily = _daily_pick(100)
        daily.result = "hit"
        daily.slot_results = {"pick": "hit"}
        save_pick(daily, tmp_path)
        mock_now.return_value = datetime(2026, 4, 4, 5, 1, tzinfo=ZoneInfo("America/New_York"))

        status = run_result_polling(100, "2026-04-04", tmp_path)

        assert status == "final"
        assert load_pick("2026-04-04", tmp_path).result == "hit"


class TestRunDay:
    @patch("bts.scheduler.fetch_schedule")
    @patch("bts.scheduler._now_et")
    @patch("bts.scheduler.time.sleep")
    @patch("bts.scheduler.run_single_check")
    @patch("bts.scheduler.run_result_polling")
    def test_dry_run_shows_schedule(
        self, mock_poll, mock_check, mock_sleep, mock_now, mock_schedule,
        tmp_path, capsys
    ):
        from bts.scheduler import run_day

        mock_schedule.return_value = [
            _game(100, "13:10", date="2026-04-03"),
            _game(200, "19:05", date="2026-04-03"),
            _game(300, "19:10", date="2026-04-03"),
        ]
        # Set time past all checks so loop exits immediately
        mock_now.return_value = datetime(2026, 4, 3, 22, 0, tzinfo=ET)

        run_day(
            date="2026-04-03",
            config={"orchestrator": {"picks_dir": str(tmp_path)}, "tiers": [],
                    "scheduler": {"early_lock_gap": 0.03, "lineup_check_offset_min": 45,
                                  "cluster_min": 10, "doubleheader_recheck_min": 15,
                                  "results_poll_interval_min": 15, "results_cap_hour_et": 5}},
            dry_run=True,
        )
        # Should not have called run_single_check in dry_run mode
        mock_check.assert_not_called()

    @patch("bts.health.runner.run_all_checks")
    @patch("bts.scheduler.fetch_schedule")
    @patch("bts.scheduler._now_et")
    @patch("bts.scheduler.time.sleep")
    @patch("bts.scheduler.run_single_check")
    @patch("bts.scheduler.run_result_polling")
    def test_eod_health_uses_run_date_and_pa_attribution(
        self, mock_poll, mock_check, mock_sleep, mock_now, mock_schedule,
        mock_run_all_checks, tmp_path,
    ):
        """End-of-day health must run on the run-day date and supply data_dir so
        realized_calibration uses PA-frame attribution — not date.today() and the
        biased streak-proxy fallback (audit finding H1)."""
        from pathlib import Path
        from datetime import date as _date
        from bts.scheduler import run_day

        mock_run_all_checks.return_value = []
        mock_schedule.side_effect = [
            [_game(100, "13:10", date="2026-04-03")],
            [],  # tomorrow's schedule (lookahead)
        ]
        # Past the only game's check window so the loop exits into the EOD block.
        mock_now.return_value = datetime(2026, 4, 3, 23, 30, tzinfo=ET)
        mock_check.return_value = {
            "skipped": True, "new_lineups": 0, "should_post": False,
            "pick_result": None, "pick_name": None, "pick_p": None,
        }

        run_day(
            date="2026-04-03",
            config={
                "orchestrator": {"picks_dir": str(tmp_path), "data_dir": "data/processed"},
                "tiers": [],
                "scheduler": {"early_lock_gap": 0.03, "lineup_check_offset_min": 45,
                              "cluster_min": 10, "doubleheader_recheck_min": 15,
                              "results_poll_interval_min": 15, "results_cap_hour_et": 5},
            },
        )

        mock_run_all_checks.assert_called_once()
        kwargs = mock_run_all_checks.call_args.kwargs
        assert kwargs.get("today") == _date(2026, 4, 3), (
            "EOD health ran on the wrong date; checks would no-op or evaluate the "
            "wrong day on a post-midnight run"
        )
        assert kwargs.get("data_dir") == Path("data/processed"), (
            "data_dir not passed; realized_calibration falls back to the biased "
            "streak-proxy attribution path"
        )

    @patch("bts.scheduler.fetch_schedule")
    @patch("bts.scheduler._now_et")
    @patch("bts.scheduler.time.sleep")
    @patch("bts.scheduler.run_single_check")
    @patch("bts.scheduler.run_result_polling")
    @patch("bts.posting.post_to_bluesky")
    @patch("bts.scheduler._trigger_live_forward_capture_on_lock")
    def test_fallback_fires_when_pick_game_before_next_check(
        self, mock_capture, mock_post, mock_poll, mock_check, mock_sleep, mock_now, mock_schedule,
        tmp_path, capsys
    ):
        """When no refreshed lock decision is available, fallback remains
        fail-closed: wake at game_time - 15min and post rather than drifting
        past first pitch.

        Reproduces the 2026-04-06 bug: Hoerner (CHC, 4:10 PM game) was the top
        pick at 3:25 PM with 0% gap, but the scheduler slept until 5:25 PM.
        """
        from bts.scheduler import FallbackRefreshResult, run_day
        from bts.picks import Pick, DailyPick, save_pick

        # Two game clusters: early (16:10 ET) and late (19:05 ET)
        mock_schedule.side_effect = [
            [_game(100, "16:10", date="2026-04-06"),
             _game(200, "19:05", date="2026-04-06")],
            [],  # tomorrow's schedule
        ]

        # Pre-save the candidate pick (game 100, 16:10 ET)
        daily = DailyPick(
            date="2026-04-06",
            run_time="2026-04-06T19:29:00+00:00",
            pick=Pick(
                batter_name="Hoerner", batter_id=1, team="CHC",
                lineup_position=1, pitcher_name="Baz", pitcher_id=2,
                p_game_hit=0.73, flags=[], projected_lineup=False,
                game_pk=100, game_time="2026-04-06T20:10:00Z",
            ),
            double_down=None, runner_up=None,
        )
        save_pick(daily, tmp_path)

        # Mock check returns should_post=False (gap=0%)
        from bts.strategy import PickResult
        mock_check.return_value = {
            "skipped": False, "new_lineups": 7, "should_post": False,
            "pick_result": PickResult(daily=daily, locked=False),
            "pick_name": "Hoerner", "pick_p": 0.73,
        }

        # Time is fixed at 15:29 — past the first check (15:25)
        # but well before the second check (18:20)
        mock_now.return_value = datetime(2026, 4, 6, 15, 29, tzinfo=ET)
        mock_post.return_value = "at://did:example/post/1"
        mock_poll.return_value = "final"

        with patch("bts.scheduler._refresh_pick_at_fallback_decision") as mock_refresh:
            mock_refresh.return_value = FallbackRefreshResult(
                daily=daily,
                should_post=None,
            )
            run_day(
                date="2026-04-06",
                config={
                    "orchestrator": {"picks_dir": str(tmp_path)},
                    "tiers": [],
                    "scheduler": {
                        "early_lock_gap": 0.03,
                        "lineup_check_offset_min": 45,
                        "cluster_min": 10,
                        "doubleheader_recheck_min": 15,
                        "fallback_deadline_min": 15,
                        "fallback_deadline_min_morning": 15,
                        "results_poll_interval_min": 15,
                        "results_cap_hour_et": 5,
                    },
                },
            )

        # Verify: posted to Bluesky via fallback
        mock_post.assert_called_once()
        mock_capture.assert_called_once()

        # Verify: only one prediction check ran (15:25, not 18:20)
        assert mock_check.call_count == 1

        # Verify: slept for the fallback window (~26 min = 1560 sec)
        # First sleep is to reach 15:25 (but now=15:29 > target, so no sleep there).
        # The fallback sleep should be game_time - 15min - now = 15:55 - 15:29 = 26 min.
        sleep_args = [call.args[0] for call in mock_sleep.call_args_list]
        fallback_sleep = [s for s in sleep_args if 1500 < s < 1700]
        assert len(fallback_sleep) == 1, f"Expected one ~26min sleep, got: {sleep_args}"

        captured = capsys.readouterr()
        assert "FALLBACK" in captured.err
        assert "LOCKED" in captured.err

    @patch("bts.scheduler.fetch_schedule")
    @patch("bts.scheduler._now_et")
    @patch("bts.scheduler.time.sleep")
    @patch("bts.scheduler.run_single_check")
    @patch("bts.scheduler.run_result_polling")
    @patch("bts.posting.post_to_bluesky")
    @patch("bts.scheduler._trigger_live_forward_capture_on_lock")
    @patch("bts.scheduler._refresh_pick_at_fallback_decision")
    def test_fallback_defers_when_should_lock_false_and_future_checks_remain(
        self, mock_refresh, mock_capture, mock_post, mock_poll, mock_check,
        mock_sleep, mock_now, mock_schedule, tmp_path, capsys,
    ):
        from bts.scheduler import FallbackRefreshResult, run_day
        from bts.picks import Pick, DailyPick, save_pick
        from bts.strategy import PickResult

        mock_schedule.side_effect = [
            [_game(100, "16:10", date="2026-04-06"),
             _game(200, "19:05", date="2026-04-06")],
            [],
        ]
        daily = DailyPick(
            date="2026-04-06",
            run_time="2026-04-06T19:29:00+00:00",
            pick=Pick(
                batter_name="Hoerner", batter_id=1, team="CHC",
                lineup_position=1, pitcher_name="Baz", pitcher_id=2,
                p_game_hit=0.73, flags=[], projected_lineup=False,
                game_pk=100, game_time="2026-04-06T20:10:00Z",
            ),
            double_down=None, runner_up=None,
        )
        save_pick(daily, tmp_path)
        mock_check.return_value = {
            "skipped": False, "new_lineups": 7, "should_post": False,
            "pick_result": PickResult(daily=daily, locked=False),
            "pick_name": "Hoerner", "pick_p": 0.73,
        }
        mock_refresh.return_value = FallbackRefreshResult(daily=daily, should_post=False)
        mock_now.return_value = datetime(2026, 4, 6, 15, 29, tzinfo=ET)

        run_day(
            date="2026-04-06",
            config={
                "orchestrator": {"picks_dir": str(tmp_path)},
                "tiers": [],
                "scheduler": {
                    "early_lock_gap": 0.03,
                    "lineup_check_offset_min": 45,
                    "cluster_min": 10,
                    "doubleheader_recheck_min": 15,
                    "fallback_deadline_min": 15,
                    "fallback_deadline_min_morning": 15,
                    "results_poll_interval_min": 15,
                    "results_cap_hour_et": 5,
                },
            },
        )

        mock_post.assert_not_called()
        mock_capture.assert_not_called()
        mock_poll.assert_not_called()
        assert not (tmp_path / "2026-04-06.json").exists()
        archives = list((tmp_path / "2026-04-06").glob("deferred_fallback_*.json"))
        assert len(archives) == 1
        archived = json.loads(archives[0].read_text())
        assert archived["pick"]["batter_name"] == "Hoerner"
        assert archived["deferred_fallback"]["reason"] == (
            "should_lock_false_future_checks_remain"
        )
        captured = capsys.readouterr()
        assert "FALLBACK DEFERRED" in captured.err

    @patch("bts.scheduler.fetch_schedule")
    @patch("bts.scheduler._now_et")
    @patch("bts.scheduler.time.sleep")
    @patch("bts.scheduler.run_single_check")
    @patch("bts.scheduler.run_result_polling")
    @patch("bts.posting.post_to_bluesky")
    @patch("bts.scheduler._trigger_live_forward_capture_on_lock")
    @patch("bts.scheduler._refresh_pick_at_fallback_decision")
    def test_fallback_defers_when_double_down_game_creates_early_deadline(
        self, mock_refresh, mock_capture, mock_post, mock_poll, mock_check,
        mock_sleep, mock_now, mock_schedule, tmp_path, capsys,
    ):
        from bts.scheduler import FallbackRefreshResult, run_day
        from bts.picks import Pick, DailyPick, save_pick
        from bts.strategy import PickResult

        mock_schedule.side_effect = [
            [_game(100, "14:20", date="2026-05-22"),
             _game(200, "19:05", date="2026-05-22"),
             _game(300, "19:15", date="2026-05-22")],
            [],
        ]
        daily = DailyPick(
            date="2026-05-22",
            run_time="2026-05-22T17:40:00+00:00",
            pick=Pick(
                batter_name="Later Primary", batter_id=1, team="ATL",
                lineup_position=1, pitcher_name="Pitcher", pitcher_id=2,
                p_game_hit=0.73, flags=[], projected_lineup=False,
                game_pk=200, game_time="2026-05-22T23:05:00Z",
            ),
            double_down=Pick(
                batter_name="Early Double", batter_id=3, team="CHC",
                lineup_position=1, pitcher_name="Starter", pitcher_id=4,
                p_game_hit=0.70, flags=[], projected_lineup=False,
                game_pk=100, game_time="2026-05-22T18:20:00Z",
            ),
            runner_up=None,
        )
        save_pick(daily, tmp_path)
        mock_check.return_value = {
            "skipped": False, "new_lineups": 2, "should_post": False,
            "pick_result": PickResult(daily=daily, locked=False),
            "pick_name": "Later Primary", "pick_p": 0.73,
        }
        mock_refresh.return_value = FallbackRefreshResult(daily=daily, should_post=False)
        mock_now.return_value = datetime(2026, 5, 22, 13, 40, tzinfo=ET)

        run_day(
            date="2026-05-22",
            config={
                "orchestrator": {"picks_dir": str(tmp_path)},
                "tiers": [],
                "scheduler": {
                    "early_lock_gap": 0.03,
                    "lineup_check_offset_min": 45,
                    "cluster_min": 10,
                    "doubleheader_recheck_min": 15,
                    "fallback_deadline_min": 15,
                    "fallback_deadline_min_morning": 15,
                    "results_poll_interval_min": 15,
                    "results_cap_hour_et": 5,
                },
            },
        )

        mock_post.assert_not_called()
        mock_capture.assert_not_called()
        mock_poll.assert_not_called()
        assert not (tmp_path / "2026-05-22.json").exists()
        archives = list((tmp_path / "2026-05-22").glob("deferred_fallback_*.json"))
        assert len(archives) == 1
        archived = json.loads(archives[0].read_text())
        assert archived["pick"]["batter_name"] == "Later Primary"
        assert archived["double_down"]["batter_name"] == "Early Double"
        assert archived["deferred_fallback"]["reason"] == (
            "should_lock_false_future_checks_remain"
        )
        captured = capsys.readouterr()
        assert "Earliest pick game at 14:20 ET" in captured.err
        assert "FALLBACK DEFERRED" in captured.err

    @patch("bts.scheduler.fetch_schedule")
    @patch("bts.scheduler._now_et")
    @patch("bts.scheduler.time.sleep")
    @patch("bts.scheduler.run_single_check")
    @patch("bts.scheduler.run_result_polling")
    @patch("bts.posting.post_to_bluesky")
    @patch("bts.scheduler._trigger_live_forward_capture_on_lock")
    @patch("bts.scheduler._refresh_pick_at_fallback_decision")
    def test_fallback_delivers_when_future_checks_have_no_pending_lineups(
        self, mock_refresh, mock_capture, mock_post, mock_poll, mock_check,
        mock_sleep, mock_now, mock_schedule, tmp_path,
    ):
        from bts.scheduler import FallbackRefreshResult, run_day
        from bts.picks import Pick, DailyPick, save_pick
        from bts.strategy import PickResult

        mock_schedule.side_effect = [
            [_game(100, "16:10", date="2026-04-06"),
             _game(200, "19:05", date="2026-04-06")],
            [],
        ]
        daily = DailyPick(
            date="2026-04-06",
            run_time="2026-04-06T19:29:00+00:00",
            pick=Pick(
                batter_name="Hoerner", batter_id=1, team="CHC",
                lineup_position=1, pitcher_name="Baz", pitcher_id=2,
                p_game_hit=0.73, flags=[], projected_lineup=False,
                game_pk=100, game_time="2026-04-06T20:10:00Z",
            ),
            double_down=None, runner_up=None,
        )
        save_pick(daily, tmp_path)
        mock_check.return_value = {
            "skipped": False, "new_lineups": 7, "should_post": False,
            "pick_result": PickResult(daily=daily, locked=False),
            "pick_name": "Hoerner", "pick_p": 0.73,
        }
        mock_refresh.return_value = FallbackRefreshResult(daily=daily, should_post=False)
        mock_now.return_value = datetime(2026, 4, 6, 15, 29, tzinfo=ET)
        mock_post.return_value = "at://did:example/post/1"
        mock_poll.return_value = "final"

        with patch(
            "bts.scheduler._has_pending_future_confirmation_window",
            return_value=False,
        ):
            run_day(
                date="2026-04-06",
                config={
                    "orchestrator": {"picks_dir": str(tmp_path)},
                    "tiers": [],
                    "scheduler": {
                        "early_lock_gap": 0.03,
                        "lineup_check_offset_min": 45,
                        "cluster_min": 10,
                        "doubleheader_recheck_min": 15,
                        "fallback_deadline_min": 15,
                        "fallback_deadline_min_morning": 15,
                        "results_poll_interval_min": 15,
                        "results_cap_hour_et": 5,
                    },
                },
            )

        mock_post.assert_called_once()
        mock_capture.assert_called_once()

    @patch("bts.scheduler.fetch_schedule")
    @patch("bts.scheduler._now_et")
    @patch("bts.scheduler.time.sleep")
    @patch("bts.scheduler.run_single_check")
    @patch("bts.scheduler.run_result_polling")
    @patch("bts.posting.post_to_bluesky")
    @patch("bts.scheduler._trigger_live_forward_capture_on_lock")
    @patch("bts.scheduler._refresh_pick_at_fallback_decision")
    def test_fallback_delivers_when_no_future_checks_remain(
        self, mock_refresh, mock_capture, mock_post, mock_poll, mock_check,
        mock_sleep, mock_now, mock_schedule, tmp_path,
    ):
        from bts.scheduler import FallbackRefreshResult, run_day
        from bts.picks import Pick, DailyPick, save_pick
        from bts.strategy import PickResult

        mock_schedule.side_effect = [
            [_game(100, "16:10", date="2026-04-06")],
            [],
        ]
        daily = DailyPick(
            date="2026-04-06",
            run_time="2026-04-06T19:29:00+00:00",
            pick=Pick(
                batter_name="Hoerner", batter_id=1, team="CHC",
                lineup_position=1, pitcher_name="Baz", pitcher_id=2,
                p_game_hit=0.73, flags=[], projected_lineup=False,
                game_pk=100, game_time="2026-04-06T20:10:00Z",
            ),
            double_down=None, runner_up=None,
        )
        save_pick(daily, tmp_path)
        mock_check.return_value = {
            "skipped": False, "new_lineups": 7, "should_post": False,
            "pick_result": PickResult(daily=daily, locked=False),
            "pick_name": "Hoerner", "pick_p": 0.73,
        }
        mock_refresh.return_value = FallbackRefreshResult(daily=daily, should_post=False)
        mock_now.return_value = datetime(2026, 4, 6, 15, 29, tzinfo=ET)
        mock_post.return_value = "at://did:example/post/1"
        mock_poll.return_value = "final"

        run_day(
            date="2026-04-06",
            config={
                "orchestrator": {"picks_dir": str(tmp_path)},
                "tiers": [],
                "scheduler": {
                    "early_lock_gap": 0.03,
                    "lineup_check_offset_min": 45,
                    "cluster_min": 10,
                    "doubleheader_recheck_min": 15,
                    "fallback_deadline_min": 15,
                    "fallback_deadline_min_morning": 15,
                    "results_poll_interval_min": 15,
                    "results_cap_hour_et": 5,
                },
            },
        )

        mock_post.assert_called_once()
        mock_capture.assert_called_once()

    @patch("bts.scheduler.fetch_schedule")
    @patch("bts.scheduler._now_et")
    @patch("bts.scheduler.time.sleep")
    @patch("bts.scheduler.run_single_check")
    @patch("bts.scheduler.run_result_polling")
    @patch("bts.posting.post_to_bluesky")
    @patch("bts.scheduler._trigger_live_forward_capture_on_lock")
    @patch("bts.dm.send_dm")
    def test_dm_delivery_locks_without_public_post(
        self, mock_dm, mock_capture, mock_post, mock_poll, mock_check, mock_sleep,
        mock_now, mock_schedule, tmp_path, capsys,
    ):
        from bts.scheduler import run_day
        from bts.picks import Pick, DailyPick, save_pick

        mock_schedule.side_effect = [
            [_game(100, "16:10", date="2026-04-06"),
             _game(200, "19:05", date="2026-04-06")],
            [],
        ]
        daily = DailyPick(
            date="2026-04-06",
            run_time="2026-04-06T19:29:00+00:00",
            pick=Pick(
                batter_name="Hoerner", batter_id=1, team="CHC",
                lineup_position=1, pitcher_name="Baz", pitcher_id=2,
                p_game_hit=0.73, flags=[], projected_lineup=False,
                game_pk=100, game_time="2026-04-06T20:10:00Z",
            ),
            double_down=None, runner_up=None,
        )
        save_pick(daily, tmp_path)
        (tmp_path / "streak.json").write_text(json.dumps({
            "streak": 4,
            "saver_available": True,
        }))
        state_dir = tmp_path / "account_state"
        state_dir.mkdir()
        (state_dir / "contest_streak.manual.json").write_text(json.dumps({
            "active_streak": 7,
            "source": "manual_screenshot",
            "source_date": "2026-04-06",
        }))

        from bts.strategy import PickResult
        mock_check.return_value = {
            "skipped": False, "new_lineups": 7, "should_post": False,
            "pick_result": PickResult(daily=daily, locked=False),
            "pick_name": "Hoerner", "pick_p": 0.73,
        }
        mock_now.return_value = datetime(2026, 4, 6, 15, 29, tzinfo=ET)
        mock_dm.return_value = "msg-456"
        mock_poll.return_value = "final"

        run_day(
            date="2026-04-06",
            config={
                "orchestrator": {"picks_dir": str(tmp_path)},
                "bluesky": {"dm_recipient": "stonehengee.bsky.social"},
                "tiers": [],
                "health_checks": {"enabled": False},
                "scheduler": {
                    "pick_delivery": "dm",
                    "early_lock_gap": 0.03,
                    "lineup_check_offset_min": 45,
                    "cluster_min": 10,
                    "doubleheader_recheck_min": 15,
                    "fallback_deadline_min": 15,
                    "fallback_deadline_min_morning": 15,
                    "results_poll_interval_min": 15,
                    "results_cap_hour_et": 5,
                },
            },
        )

        mock_dm.assert_called_once()
        assert "Streak: 7" in mock_dm.call_args.args[1]
        mock_post.assert_not_called()
        mock_capture.assert_called_once()
        data = json.loads((tmp_path / "2026-04-06.json").read_text())
        assert data["bluesky_posted"] is False
        assert data["bluesky_uri"] is None
        assert data["notification_sent"] is True
        assert data["notification_channel"] == "bluesky_dm"
        assert data["notification_id"] == "msg-456"
        captured = capsys.readouterr()
        assert "Public Bluesky posting disabled" in captured.err
        assert "Pick DM sent" in captured.err


class TestPendingFutureConfirmationWindow:
    def test_detects_unconfirmed_future_side(self):
        from bts.scheduler import _has_pending_future_confirmation_window

        assert _has_pending_future_confirmation_window(
            [{"game_pks": [200]}],
            {(200, "away")},
        )

    def test_false_when_future_games_fully_confirmed(self):
        from bts.scheduler import _has_pending_future_confirmation_window

        assert not _has_pending_future_confirmation_window(
            [{"game_pks": [200, 201]}],
            {
                (200, "away"), (200, "home"),
                (201, "away"), (201, "home"),
            },
        )


class TestEarliestPickGameEt:
    """The fallback deadline must use the earlier of primary + double-down
    game times, since the BTS app rejects submissions once the FIRST game
    has started — not the primary's game.
    """

    def _daily(self, primary_game_time: str, double_game_time: str | None = None):
        from bts.picks import Pick, DailyPick
        primary = Pick(
            batter_name="A", batter_id=1, team="BOS", lineup_position=1,
            pitcher_name="P1", pitcher_id=10, p_game_hit=0.7, flags=[],
            projected_lineup=False, game_pk=100, game_time=primary_game_time,
        )
        double = None
        if double_game_time:
            double = Pick(
                batter_name="B", batter_id=2, team="MIN", lineup_position=2,
                pitcher_name="P2", pitcher_id=20, p_game_hit=0.7, flags=[],
                projected_lineup=False, game_pk=200, game_time=double_game_time,
            )
        return DailyPick(
            date="2026-04-12", run_time="2026-04-12T15:00:00+00:00",
            pick=primary, double_down=double, runner_up=None,
        )

    def test_returns_primary_when_no_double_down(self):
        from bts.scheduler import _earliest_pick_game_et
        daily = self._daily(primary_game_time="2026-04-12T18:15:00Z")
        result = _earliest_pick_game_et(daily)
        assert result.hour == 14 and result.minute == 15  # 18:15 UTC = 14:15 ET

    def test_returns_primary_when_primary_is_earlier(self):
        from bts.scheduler import _earliest_pick_game_et
        daily = self._daily(
            primary_game_time="2026-04-12T17:37:00Z",  # 13:37 ET
            double_game_time="2026-04-12T18:15:00Z",   # 14:15 ET
        )
        result = _earliest_pick_game_et(daily)
        assert result.hour == 13 and result.minute == 37

    def test_returns_double_down_when_double_is_earlier(self):
        """Bug repro: the 2026-04-12 morning had Roman Anthony (14:15 game) as
        primary and Luke Keaschall (13:37 game) as double-down. The scheduler
        was using the primary's game time, putting fallback at 13:40 ET — three
        minutes after Luke's 13:37 game started, missing the BTS deadline.
        """
        from bts.scheduler import _earliest_pick_game_et
        daily = self._daily(
            primary_game_time="2026-04-12T18:15:00Z",  # 14:15 ET (later)
            double_game_time="2026-04-12T17:37:00Z",   # 13:37 ET (earlier)
        )
        result = _earliest_pick_game_et(daily)
        assert result.hour == 13 and result.minute == 37


class TestPollIntervalSleep:
    """run_result_polling's inter-iteration sleep must keep the heartbeat fresh.

    Regression: 2026-04-22 ~19:05 ET. After ship of b681f8a enabled result-polling
    for today's Laureano+Henderson day, the scheduler entered a 15-min poll loop
    with no heartbeat refresh between iterations. Result: every 5-min cron fire
    of check_heartbeat.py found stale, pinged HC /fail. 12 false-alarm emails
    over 2 hours before discovery.
    """

    def test_no_heartbeat_path_still_sleeps(self, tmp_path):
        from bts.scheduler import _poll_interval_sleep
        import time as _time
        t0 = _time.monotonic()
        _poll_interval_sleep(None, seconds=0.05)
        assert _time.monotonic() - t0 >= 0.04  # sleep actually ran

    def test_heartbeat_path_refreshes_during_sleep(self, tmp_path):
        """Heartbeat file timestamp advances during the sleep when a path is given."""
        from pathlib import Path
        from bts.heartbeat import HeartbeatState, read_heartbeat, write_heartbeat
        from bts.scheduler import _poll_interval_sleep
        import time as _time

        hb = tmp_path / ".heartbeat"
        write_heartbeat(hb, state=HeartbeatState.RUNNING)
        initial_ts = read_heartbeat(hb)["timestamp"]
        _time.sleep(0.01)

        _poll_interval_sleep(hb, seconds=0.25, watchdog_interval_sec=0.05)

        final_ts = read_heartbeat(hb)["timestamp"]
        assert final_ts > initial_ts


class TestIdleUntilNextWakeup:
    """After writing IDLE_END_OF_DAY heartbeat at end of run_day, the scheduler
    must sleep until tomorrow's wakeup instead of returning. Without it, the
    process exits, systemd Restart=always re-launches, and run_day cycles
    through its short post-lock logic every ~3 min (observed 2026-04-23 evening
    post-games: NRestarts grew from 0 → 7 in 25 min).
    """

    def test_no_sleep_if_next_wakeup_is_none(self, tmp_path):
        from bts.scheduler import _idle_until_next_wakeup
        import time as _time
        t0 = _time.monotonic()
        _idle_until_next_wakeup(None, tmp_path / ".heartbeat")
        assert _time.monotonic() - t0 < 0.1

    def test_no_sleep_if_next_wakeup_is_past(self, tmp_path):
        from bts.scheduler import _idle_until_next_wakeup
        from datetime import datetime, timedelta, timezone
        import time as _time
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        t0 = _time.monotonic()
        _idle_until_next_wakeup(past, tmp_path / ".heartbeat")
        assert _time.monotonic() - t0 < 0.1

    def test_no_sleep_if_malformed_iso(self, tmp_path):
        from bts.scheduler import _idle_until_next_wakeup
        import time as _time
        t0 = _time.monotonic()
        _idle_until_next_wakeup("not-an-iso-string", tmp_path / ".heartbeat")
        assert _time.monotonic() - t0 < 0.1


class TestWatchdogPingSleep:
    """During SLEEPING-state waits (main-loop between checks, fallback deadline,
    pre-polling wait), the scheduler needs to emit notify_watchdog() pings so
    systemd's WatchdogSec=1800 doesn't SIGABRT-kill it. But it must NOT
    overwrite the heartbeat file — the pre-sleep heartbeat encodes
    sleeping_until, which check_heartbeat.py needs for fresh-sleeping logic.

    Regression: 2026-04-23 overnight. The scheduler was watchdog-killed every
    30min during the idle_end_of_day → 10:00 ET sleep, NRestarts=21.
    """

    def test_watchdog_ping_sleep_does_not_touch_heartbeat_file(self, tmp_path):
        from bts.scheduler import _watchdog_ping_sleep
        from bts.heartbeat import HeartbeatState, read_heartbeat, write_heartbeat

        hb = tmp_path / ".heartbeat"
        from datetime import datetime, timedelta, timezone
        wake = datetime.now(timezone.utc) + timedelta(hours=2)
        write_heartbeat(hb, state=HeartbeatState.SLEEPING, sleeping_until=wake)
        pre = read_heartbeat(hb)
        assert pre["state"] == HeartbeatState.SLEEPING

        _watchdog_ping_sleep(seconds=0.15, interval_sec=0.05)

        post = read_heartbeat(hb)
        assert post == pre  # unchanged — helper did not write

    def test_watchdog_ping_sleep_actually_sleeps(self):
        from bts.scheduler import _watchdog_ping_sleep
        import time as _time
        t0 = _time.monotonic()
        _watchdog_ping_sleep(seconds=0.1, interval_sec=0.02)
        assert _time.monotonic() - t0 >= 0.09


class TestComputeResultPollStart:
    """Result-polling start must use the EARLIEST of primary or double-down
    game start + 10 minutes. Primary-only calc makes the scheduler sleep
    through the earlier game — missing mid-game result persistence.

    Regression: 2026-04-22 had Laureano (SD, 20:40 ET) primary + Henderson
    (BAL, 14:10 ET) double-down. Inline primary-only computation put the
    scheduler asleep until 20:50 ET, through Henderson's full live-game
    window. Dashboard display was unaffected (fetch_live_scorecard runs
    independently), but daily.result for Henderson was never persisted
    mid-game.
    """

    def _daily(self, primary_game_time: str, double_game_time: str | None = None):
        from bts.picks import Pick, DailyPick
        primary = Pick(
            batter_name="A", batter_id=1, team="BOS", lineup_position=1,
            pitcher_name="P1", pitcher_id=10, p_game_hit=0.7, flags=[],
            projected_lineup=False, game_pk=100, game_time=primary_game_time,
        )
        double = None
        if double_game_time:
            double = Pick(
                batter_name="B", batter_id=2, team="MIN", lineup_position=2,
                pitcher_name="P2", pitcher_id=20, p_game_hit=0.7, flags=[],
                projected_lineup=False, game_pk=200, game_time=double_game_time,
            )
        return DailyPick(
            date="2026-04-22", run_time="2026-04-22T17:00:00+00:00",
            pick=primary, double_down=double, runner_up=None,
        )

    def test_no_double_down_uses_primary_plus_10(self):
        from bts.scheduler import _compute_result_poll_start
        daily = self._daily(primary_game_time="2026-04-23T00:40:00Z")  # 20:40 ET
        result = _compute_result_poll_start(daily)
        assert result.hour == 20 and result.minute == 50

    def test_primary_earlier_uses_primary_plus_10(self):
        from bts.scheduler import _compute_result_poll_start
        daily = self._daily(
            primary_game_time="2026-04-22T17:37:00Z",  # 13:37 ET (primary, earlier)
            double_game_time="2026-04-22T18:15:00Z",   # 14:15 ET (double, later)
        )
        result = _compute_result_poll_start(daily)
        assert result.hour == 13 and result.minute == 47

    def test_double_down_earlier_uses_double_plus_10(self):
        """The actual 2026-04-22 prod scenario: Laureano 20:40 + Henderson 14:10."""
        from bts.scheduler import _compute_result_poll_start
        daily = self._daily(
            primary_game_time="2026-04-23T00:40:00Z",  # 20:40 ET (primary, later)
            double_game_time="2026-04-22T18:10:00Z",   # 14:10 ET (double, earlier)
        )
        result = _compute_result_poll_start(daily)
        assert result.hour == 14 and result.minute == 20


class TestRefreshPickAtFallback:
    """_refresh_pick_at_fallback re-runs predictions right before the fallback
    posts, so late-arriving lineups (e.g., PHI lineup posted 10 min before its
    13:35 first pitch) can swap in a better pick than the one cached from the
    last scheduled check.

    Regression: 2026-04-12 Sunday slate locked Donovan (SEA) from the 13:19 ET
    cached prediction, but by the 13:40 ET fallback fire, PHI's lineup had
    confirmed and Trea Turner (0.7426) would have been a better primary. The
    fallback just reposted the cached 13:19 pick without refreshing, leaving
    Turner on the table.
    """

    def _daily(self, batter_name="Old Batter", batter_id=1001, p=0.70,
               game_pk=778899):
        from bts.picks import DailyPick, Pick
        return DailyPick(
            date="2026-04-12",
            run_time="2026-04-12T17:19:00+00:00",
            pick=Pick(
                batter_name=batter_name, batter_id=batter_id, team="SEA",
                lineup_position=1, pitcher_name="Opener", pitcher_id=9999,
                p_game_hit=p, flags=[], projected_lineup=False,
                game_pk=game_pk, game_time="2026-04-12T20:10:00Z",
                pitcher_team="HOU",
            ),
            double_down=None, runner_up=None,
        )

    def test_swaps_to_fresh_pick_when_refresh_returns_different_batter(self, tmp_path):
        from bts.scheduler import _refresh_pick_at_fallback
        from bts.strategy import PickResult, SelectionResult

        cached = self._daily(batter_name="Old Batter", batter_id=1001, p=0.70)
        fresh_daily = self._daily(batter_name="Trea Turner", batter_id=2002, p=0.74)
        fresh_result = PickResult(daily=fresh_daily, locked=False)
        fresh_sel = SelectionResult(fresh_result, None, None, None, None, None)

        config = {"orchestrator": {"picks_dir": str(tmp_path)}}

        with patch("bts.scheduler.run_and_pick",
                   return_value=(None, fresh_sel, "local")):
            result = _refresh_pick_at_fallback(config, "2026-04-12", cached)

        assert result.pick.batter_name == "Trea Turner"
        assert result.pick.batter_id == 2002
        assert result.pick.p_game_hit == 0.74

    def test_logs_when_pick_changes(self, tmp_path, capsys):
        from bts.scheduler import _refresh_pick_at_fallback
        from bts.strategy import PickResult, SelectionResult

        cached = self._daily(batter_name="Brendan Donovan", batter_id=680977, p=0.7169)
        fresh_daily = self._daily(batter_name="Trea Turner", batter_id=607208, p=0.7426)
        fresh_result = PickResult(daily=fresh_daily, locked=False)
        fresh_sel = SelectionResult(fresh_result, None, None, None, None, None)

        with patch("bts.scheduler.run_and_pick",
                   return_value=(None, fresh_sel, "local")):
            _refresh_pick_at_fallback(
                {"orchestrator": {"picks_dir": str(tmp_path)}},
                "2026-04-12",
                cached,
            )
        err = capsys.readouterr().err
        assert "CHANGED" in err
        assert "Brendan Donovan" in err
        assert "Trea Turner" in err

    def test_keeps_cached_when_fresh_pick_matches(self, tmp_path, capsys):
        from bts.scheduler import _refresh_pick_at_fallback
        from bts.strategy import PickResult, SelectionResult

        cached = self._daily(batter_name="Same Batter", batter_id=5, p=0.70)
        fresh_daily = self._daily(batter_name="Same Batter", batter_id=5, p=0.71)
        fresh_result = PickResult(daily=fresh_daily, locked=False)
        fresh_sel = SelectionResult(fresh_result, None, None, None, None, None)

        with patch("bts.scheduler.run_and_pick",
                   return_value=(None, fresh_sel, "local")):
            result = _refresh_pick_at_fallback(
                {"orchestrator": {"picks_dir": str(tmp_path)}},
                "2026-04-12",
                cached,
            )
        err = capsys.readouterr().err
        assert "unchanged" in err.lower()
        assert result.pick.batter_id == 5

    def test_falls_back_to_cached_on_exception(self, tmp_path):
        from bts.scheduler import _refresh_pick_at_fallback

        cached = self._daily(batter_name="Cached", batter_id=1, p=0.70)

        with patch("bts.scheduler.run_and_pick",
                   side_effect=RuntimeError("cascade failed")):
            result = _refresh_pick_at_fallback(
                {"orchestrator": {"picks_dir": str(tmp_path)}},
                "2026-04-12",
                cached,
            )
        assert result is cached

    def test_falls_back_to_cached_when_pick_result_is_none(self, tmp_path):
        from bts.scheduler import _refresh_pick_at_fallback

        cached = self._daily()
        with patch("bts.scheduler.run_and_pick",
                   return_value=(None, None, "local")):
            result = _refresh_pick_at_fallback(
                {"orchestrator": {"picks_dir": str(tmp_path)}},
                "2026-04-12",
                cached,
            )
        assert result is cached


class TestLiveForwardCaptureTrigger:
    def test_queues_default_systemd_capture_nonblocking(self, capsys):
        from bts.scheduler import _trigger_live_forward_capture_on_lock

        with patch("bts.scheduler.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = ""
            mock_run.return_value.stderr = ""

            _trigger_live_forward_capture_on_lock({"scheduler": {}}, "2026-05-16")

        mock_run.assert_called_once()
        assert mock_run.call_args.args[0] == [
            "systemctl",
            "--user",
            "start",
            "--no-block",
            "bts-live-forward-capture.service",
        ]
        assert mock_run.call_args.kwargs["timeout"] == 10
        assert "queued" in capsys.readouterr().err

    def test_disabled_capture_trigger_is_noop(self):
        from bts.scheduler import _trigger_live_forward_capture_on_lock

        with patch("bts.scheduler.subprocess.run") as mock_run:
            _trigger_live_forward_capture_on_lock(
                {"scheduler": {"live_forward_capture_on_lock": False}},
                "2026-05-16",
            )

        mock_run.assert_not_called()


class TestShadowPredictionTrigger:
    def _state(self, tmp_path, *, analytics_jobs=None):
        from bts.scheduler import SchedulerState, save_state

        picks_dir = tmp_path / "picks"
        state = SchedulerState(
            date="2026-05-16",
            schedule_fetched_at="2026-05-16T10:00:00-04:00",
            games=[],
            confirmed_game_pks=[],
            runs_completed=[],
            pick_locked=True,
            pick_locked_at="2026-05-16T12:00:00-04:00",
            result_status=None,
            next_wakeup=None,
            analytics_jobs=analytics_jobs or {},
        )
        save_state(state, picks_dir)
        return picks_dir

    def test_default_shadow_trigger_preserves_inline_behavior(self):
        from bts.scheduler import _trigger_shadow_prediction_on_lock

        config = {"scheduler": {}}
        with patch("bts.scheduler._run_shadow_prediction") as mock_shadow:
            _trigger_shadow_prediction_on_lock(config, "2026-05-16", "Prod Pick")

        mock_shadow.assert_called_once_with(config, "2026-05-16", "Prod Pick")

    def test_queues_configured_systemd_shadow_nonblocking(self, tmp_path, capsys):
        from bts.scheduler import _trigger_shadow_prediction_on_lock, load_state

        picks_dir = self._state(tmp_path)
        config = {
            "orchestrator": {"picks_dir": str(picks_dir)},
            "scheduler": {"shadow_model_unit": "bts-shadow-prediction.service"},
        }
        with patch("bts.scheduler.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = ""
            mock_run.return_value.stderr = ""

            _trigger_shadow_prediction_on_lock(config, "2026-05-16", "Prod Pick")

        mock_run.assert_called_once()
        assert mock_run.call_args.args[0] == [
            "systemctl",
            "--user",
            "start",
            "--no-block",
            "bts-shadow-prediction.service",
        ]
        assert mock_run.call_args.kwargs["timeout"] == 10
        loaded = load_state("2026-05-16", picks_dir)
        assert loaded.analytics_jobs["shadow"]["status"] == "dispatched"
        assert loaded.analytics_jobs["shadow"]["reason"] == "trigger_queued"
        assert loaded.analytics_jobs["shadow"]["unit"] == "bts-shadow-prediction.service"
        assert "Trigger queued" in capsys.readouterr().err

    def test_shadow_trigger_skips_prior_attempt(self, tmp_path):
        from bts.scheduler import _trigger_shadow_prediction_on_lock

        picks_dir = self._state(
            tmp_path,
            analytics_jobs={"shadow": {"status": "dispatched", "updated_at": "now"}},
        )
        config = {
            "orchestrator": {"picks_dir": str(picks_dir)},
            "scheduler": {"shadow_model_unit": "bts-shadow-prediction.service"},
        }
        with patch("bts.scheduler.subprocess.run") as mock_run:
            _trigger_shadow_prediction_on_lock(config, "2026-05-16", "Prod Pick")

        mock_run.assert_not_called()
