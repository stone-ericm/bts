"""Thin slates get early lineup checks (roadmap thread B, 2026-11-13).

2026-07-16: a one-game slate planned a single check at start minus 60; MLB then moved the game up an hour, the lone
check fired at first pitch and the day locked undelivered. Opening Night 2027-03-24 is a one-game slate.
A check that runs before the cutoff is enough: from then on the fallback deadline is computed from the pick's
CURRENT game time, which the prediction pipeline re-fetches.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from bts.scheduler import compute_run_times, plan_run_times
from tests.test_scheduler import _game

ET = ZoneInfo("America/New_York")
D = "2026-07-16"
LEAD_NEEDED = timedelta(minutes=5 + 12 + 10)      # cutoff + cascade budget + operator reserve (the fallback floor)


def _at(hhmm: str) -> datetime:
    return datetime.strptime(f"{D} {hhmm}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)


def _times(runs):
    return [r["time_et"].strftime("%H:%M") for r in runs]


def test_a_one_game_slate_gets_three_checks():
    runs = plan_run_times([_game(100, "19:10", date=D)], offset_min=60, cluster_min=10, now=_at("10:00"))
    assert _times(runs) == ["16:10", "17:10", "18:10"]
    assert all(r["game_pks"] == [100] for r in runs)


def test_the_july_16_replay_has_a_check_in_time_after_the_game_moves_up_an_hour():
    """Planned at 10:00 from the original 19:10 start; the game is then played at 18:10."""
    planned = plan_run_times([_game(100, "19:10", date=D)], offset_min=60, cluster_min=10, now=_at("10:00"))
    actual_first_pitch = _at("18:10")
    assert [r for r in planned if r["time_et"] <= actual_first_pitch - LEAD_NEEDED]
    old = compute_run_times([_game(100, "19:10", date=D)], offset_min=60, cluster_min=10)
    assert not [r for r in old if r["time_et"] <= actual_first_pitch - LEAD_NEEDED]     # what happened on 7/16


def test_a_two_hour_move_still_leaves_one_check_in_time():
    planned = plan_run_times([_game(100, "19:10", date=D)], offset_min=60, cluster_min=10, now=_at("10:00"))
    assert [r for r in planned if r["time_et"] <= _at("17:10") - LEAD_NEEDED]


def test_a_two_game_slate_gets_early_checks_for_both_games():
    runs = plan_run_times([_game(100, "13:10", date=D), _game(200, "19:10", date=D)],
                          offset_min=60, cluster_min=10, now=_at("09:00"))
    assert _times(runs) == ["10:10", "11:10", "12:10", "16:10", "17:10", "18:10"]
    assert [r["game_pks"] for r in runs] == [[100], [100], [100], [200], [200], [200]]


def test_a_normal_slate_is_planned_exactly_as_before():
    games = [_game(100, "13:10", date=D), _game(200, "19:05", date=D), _game(300, "19:10", date=D)]
    assert plan_run_times(games, offset_min=60, cluster_min=10, now=_at("10:00")) == \
        compute_run_times(games, offset_min=60, cluster_min=10)


def test_early_checks_already_in_the_past_are_dropped_but_the_regular_check_is_kept():
    """A late start or a restart must not fire three back-to-back cascades."""
    runs = plan_run_times([_game(100, "19:10", date=D)], offset_min=60, cluster_min=10, now=_at("17:00"))
    assert _times(runs) == ["17:10", "18:10"]
    runs = plan_run_times([_game(100, "19:10", date=D)], offset_min=60, cluster_min=10, now=_at("18:30"))
    assert _times(runs) == ["18:10"]                       # the regular check always stays, as before


def test_early_checks_of_two_close_games_cluster_like_regular_checks():
    runs = plan_run_times([_game(100, "19:05", date=D), _game(200, "19:10", date=D)],
                          offset_min=60, cluster_min=10, now=_at("10:00"))
    assert _times(runs) == ["16:05", "17:05", "18:05"]
    assert all(sorted(r["game_pks"]) == [100, 200] for r in runs)


def test_an_offset_no_earlier_than_the_regular_check_adds_nothing():
    runs = plan_run_times([_game(100, "19:10", date=D)], offset_min=180, cluster_min=10, now=_at("10:00"))
    assert _times(runs) == ["16:10"]


def test_no_games_plans_nothing():
    assert plan_run_times([], now=_at("10:00")) == []


def test_run_day_plans_a_one_game_slate_with_the_early_checks(monkeypatch, tmp_path, capsys):
    import bts.scheduler as SCH
    monkeypatch.setattr(SCH, "fetch_schedule", lambda date: [_game(100, "19:10", date=D)])
    monkeypatch.setattr(SCH, "_now_et", lambda: _at("10:00"))
    config = {"orchestrator": {"picks_dir": str(tmp_path / "picks"), "heartbeat_path": str(tmp_path / ".hb")},
              "scheduler": {"lineup_check_offset_min": 60, "pick_delivery": "private"}}
    SCH.run_day(D, config, dry_run=True)
    err = capsys.readouterr().err
    assert "1 games, 3 scheduled checks" in err
    assert "16:10 ET" in err and "17:10 ET" in err and "18:10 ET" in err


def test_the_day_loop_runs_the_early_checks_at_their_times(monkeypatch, tmp_path):
    """The real run_day on a moving clock: a one-game 19:10 slate planned at 10:00 runs its checks at 16:10, 17:10
    and 18:10. On 7/16 the first (and only) check ran at 18:10, which was first pitch after the game moved up."""
    import bts.scheduler as sch
    from tests import test_scheduler_decision_record_integration as H
    clock = {"now": _at("10:00")}
    calls = []

    def sleep(seconds, *a, **k):
        clock["now"] += timedelta(seconds=seconds)

    def check(**k):
        calls.append(clock["now"].strftime("%H:%M"))
        return H._result()

    monkeypatch.setattr(sch, "fetch_schedule", H._seq([[_game(100, "19:10", date=D)], []]))
    monkeypatch.setattr(sch, "_now_et", lambda: clock["now"])
    monkeypatch.setattr(sch, "_watchdog_ping_sleep", sleep)
    monkeypatch.setattr(sch.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(sch, "run_result_polling", lambda *a, **k: "final")
    monkeypatch.setattr(sch, "_trigger_live_forward_capture_on_lock", lambda *a, **k: None)
    monkeypatch.setattr(sch, "_idle_until_next_wakeup", lambda *a, **k: None)
    monkeypatch.setattr(sch, "run_single_check", check)
    config = {"orchestrator": {"picks_dir": str(tmp_path), "heartbeat_path": str(tmp_path / ".hb")}, "tiers": [],
              "scheduler": {"pick_delivery": "private", "early_lock_gap": 0.03, "lineup_check_offset_min": 60,
                            "cluster_min": 10, "doubleheader_recheck_min": 15, "missed_pick_alert_min": 10,
                            "results_poll_interval_min": 15, "results_cap_hour_et": 5},
              "health_checks": {"enabled": False}}
    sch.run_day(date=D, config=config, dry_run=False)
    assert calls[:3] == ["16:10", "17:10", "18:10"]
