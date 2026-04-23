"""Tests for the dynamic lineup scheduler."""

import json
import pytest
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


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

    def test_load_returns_none_when_missing(self, tmp_path):
        from bts.scheduler import load_state

        assert load_state("2026-04-03", tmp_path) is None


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

    @patch("bts.scheduler.check_confirmed_lineups")
    @patch("bts.orchestrator.run_cascade")
    @patch("bts.strategy.get_game_statuses", return_value={100: "P", 200: "F"})
    @patch("bts.picks.get_game_statuses", return_value={100: "P", 200: "F"})
    @patch("bts.strategy._load_mdp", return_value=None)
    def test_should_lock_excludes_postponed_games(
        self, _mdp, _sched_statuses, _strat_statuses, mock_cascade, mock_lineups, tmp_path
    ):
        """Projected picks from postponed/finished games shouldn't block locking.

        Reproduces the 2026-04-04 bug: CHC@CLE was postponed (status=F) but its
        projected batters prevented should_lock from returning True because
        the gap was under early_lock_gap.
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

        # Game 200 is Final (postponed) — its projected batter (Kwan, 0.80)
        # should be excluded from the should_lock gap check. Without the fix,
        # the gap (0.82 - 0.80 = 0.02 < 0.03) would block locking.
        # With the fix, only game 100's picks remain — all confirmed → lock.
        assert result["should_post"] is True
        assert result["pick_name"] == "Díaz"
        assert result["pick_p"] is not None

    @patch("bts.scheduler.check_confirmed_lineups")
    @patch("bts.picks.get_game_statuses", return_value={100: "L"})
    def test_short_circuits_when_pick_locked(self, _statuses, mock_lineups, tmp_path):
        """Skip the expensive SSH cascade when pick is already locked."""
        from bts.scheduler import run_single_check
        from bts.picks import Pick, DailyPick, save_pick

        mock_lineups.return_value = {100: {"home", "away"}}

        # Pre-save a pick whose game has started (status L)
        existing = DailyPick(
            date="2026-04-04",
            run_time="2026-04-04T22:00:00+00:00",
            pick=Pick(
                batter_name="Díaz", batter_id=1, team="TB",
                lineup_position=1, pitcher_name="Abel", pitcher_id=2,
                p_game_hit=0.72, flags=[], projected_lineup=False,
                game_pk=100, game_time="2026-04-04T23:10:00Z",
            ),
            double_down=None, runner_up=None,
        )
        save_pick(existing, tmp_path)

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

    @patch("bts.scheduler.fetch_schedule")
    @patch("bts.scheduler._now_et")
    @patch("bts.scheduler.time.sleep")
    @patch("bts.scheduler.run_single_check")
    @patch("bts.scheduler.run_result_polling")
    @patch("bts.posting.post_to_bluesky")
    def test_fallback_fires_when_pick_game_before_next_check(
        self, mock_post, mock_poll, mock_check, mock_sleep, mock_now, mock_schedule,
        tmp_path, capsys
    ):
        """When the top pick plays in the earliest game and should_lock=False,
        the scheduler should wake up at game_time - 15min to force-post,
        rather than sleeping until the next cluster check.

        Reproduces the 2026-04-06 bug: Hoerner (CHC, 4:10 PM game) was the top
        pick at 3:25 PM with 0% gap, but the scheduler slept until 5:25 PM.
        """
        from bts.scheduler import run_day
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
        from bts.strategy import PickResult

        cached = self._daily(batter_name="Old Batter", batter_id=1001, p=0.70)
        fresh_daily = self._daily(batter_name="Trea Turner", batter_id=2002, p=0.74)
        fresh_result = PickResult(daily=fresh_daily, locked=False)

        config = {"orchestrator": {"picks_dir": str(tmp_path)}}

        with patch("bts.scheduler.run_and_pick",
                   return_value=(None, fresh_result, "local")):
            result = _refresh_pick_at_fallback(config, "2026-04-12", cached)

        assert result.pick.batter_name == "Trea Turner"
        assert result.pick.batter_id == 2002
        assert result.pick.p_game_hit == 0.74

    def test_logs_when_pick_changes(self, tmp_path, capsys):
        from bts.scheduler import _refresh_pick_at_fallback
        from bts.strategy import PickResult

        cached = self._daily(batter_name="Brendan Donovan", batter_id=680977, p=0.7169)
        fresh_daily = self._daily(batter_name="Trea Turner", batter_id=607208, p=0.7426)
        fresh_result = PickResult(daily=fresh_daily, locked=False)

        with patch("bts.scheduler.run_and_pick",
                   return_value=(None, fresh_result, "local")):
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
        from bts.strategy import PickResult

        cached = self._daily(batter_name="Same Batter", batter_id=5, p=0.70)
        fresh_daily = self._daily(batter_name="Same Batter", batter_id=5, p=0.71)
        fresh_result = PickResult(daily=fresh_daily, locked=False)

        with patch("bts.scheduler.run_and_pick",
                   return_value=(None, fresh_result, "local")):
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
