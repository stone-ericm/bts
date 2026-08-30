"""Replay of 2026-08-30 with an ADVANCING clock.

Kwan (CLE, first pitch 13:40 ET, cutoff 13:35). The 12:35 check found
should_lock=False (gap 1.5% vs a projected contender in the 16:07 PHI game); the
next scheduled check (13:10) was after the T−35 fallback deadline (13:05), so the
loop slept to 13:05 and ran the fallback refresh, which took 15.5 minutes.

Production deferred at 13:20 on a stale boolean, then ran the overdue 13:10 check
(another 15.5 min) and DM'd Kwan at 13:36 — after the cutoff. The planner must
DELIVER Kwan at 13:20: the contender's window (15:05 run) cannot finish before
Kwan's deliver-by time, so waiting cannot change the decision in time.
"""
import json
from datetime import datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

from tests.test_scheduler import _game

ET = ZoneInfo("America/New_York")
DATE = "2026-08-30"
CASCADE = timedelta(minutes=15, seconds=30)


class Clock:
    def __init__(self, start):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, delta):
        self.now = self.now + delta


def _config(tmp_path):
    return {
        "orchestrator": {"picks_dir": str(tmp_path), "heartbeat_path": str(tmp_path / ".hb")},
        "tiers": [],
        "bluesky": {"dm_recipient": "eric"},
        "scheduler": {"pick_delivery": "dm", "early_lock_gap": 0.03,
                      "lineup_check_offset_min": 60, "cluster_min": 10,
                      "doubleheader_recheck_min": 15, "fallback_deadline_min": 35,
                      "fallback_deadline_min_morning": 25, "results_poll_interval_min": 15,
                      "results_cap_hour_et": 5, "cascade_budget_min": 12,
                      "operator_reserve_min": 10},
        "health_checks": {"enabled": False},
    }


def _kwan_daily():
    from bts.picks import DailyPick, Pick
    kwan = Pick(batter_name="Steven Kwan", batter_id=680757, team="CLE", lineup_position=1,
                pitcher_name="Lugo", pitcher_id=607625, p_game_hit=0.7566, flags=[],
                projected_lineup=False, game_pk=824393, game_time="2026-08-30T17:40:00Z")
    return DailyPick(date=DATE, run_time="2026-08-30T16:50:55+00:00", pick=kwan,
                     double_down=None, runner_up=None)


def _schedule():
    return [_game(824393, "13:40", date=DATE), _game(823662, "14:10", date=DATE),
            _game(823987, "16:07", date=DATE), _game(824636, "19:20", date=DATE)]


def _run(tmp_path, clock, refresh_result_factory):
    from bts.picks import save_pick
    from bts.scheduler import run_day
    from bts.strategy import PickResult

    daily = _kwan_daily()
    save_pick(daily, tmp_path)

    def fake_check(**kw):            # the 12:35 lineup check: 15.5-min cascade, should_lock False
        clock.advance(CASCADE)
        return {"skipped": False, "new_lineups": 4, "should_post": False,
                "pick_result": PickResult(daily=daily, locked=False),
                "pick_name": "Steven Kwan", "pick_p": 0.7566}

    def fake_refresh(config, date, cached, gap, **kw):   # the 13:05 fallback refresh
        clock.advance(CASCADE)
        return refresh_result_factory(daily)

    def fake_sleep(secs):
        clock.advance(timedelta(seconds=secs))

    with patch("bts.scheduler.fetch_schedule", side_effect=[_schedule(), []]), \
         patch("bts.scheduler._now_et", side_effect=clock), \
         patch("bts.scheduler.time.sleep", side_effect=fake_sleep), \
         patch("bts.scheduler.run_single_check", side_effect=fake_check) as mock_check, \
         patch("bts.scheduler._refresh_pick_at_fallback_decision", side_effect=fake_refresh), \
         patch("bts.scheduler.count_new_confirmations", return_value=0), \
         patch("bts.scheduler.run_result_polling", return_value="final"), \
         patch("bts.scheduler._trigger_live_forward_capture_on_lock"), \
         patch("bts.dm.send_dm", return_value="dm-1") as mock_dm, \
         patch("bts.contest_state.load_decision_streak_state") as dss:
        dss.return_value.streak = 0
        run_day(date=DATE, config=_config(tmp_path))
    return mock_check, mock_dm


def test_kwan_delivered_at_13_20(tmp_path, capsys):
    from bts.scheduler import FallbackRefreshResult

    clock = Clock(datetime(2026, 8, 30, 12, 35, tzinfo=ET))
    mock_check, mock_dm = _run(
        tmp_path, clock,
        lambda daily: FallbackRefreshResult(daily=daily, should_post=False,
                                            should_post_ungated=False, block_reason="gap",
                                            contender_game_pk=823987))
    mock_dm.assert_called_once()
    assert mock_check.call_count == 1                       # the overdue 13:10 check never ran
    delivered = json.loads((tmp_path / f"{DATE}.json").read_text())
    assert delivered["delivered_at"].startswith("2026-08-30T13:20")
    state = json.loads((tmp_path / DATE / "scheduler_state.json").read_text())
    assert state["fallback_refreshes"][0]["action"] == "deliver"
    assert state["fallback_refreshes"][0]["reason"] == "gap_no_feasible_window"
    assert state["fallback_refreshes"][0]["duration_sec"] is not None
    assert state["runs_completed"][0]["duration_sec"] is not None
    err = capsys.readouterr().err
    assert "gap_no_feasible_window" in err and "LOCKED (fallback)" in err


def test_check_overrunning_deadline_reuses_its_cascade(tmp_path, capsys):
    """Codex r2 F3: a scheduled check that finishes at/after the fallback deadline
    must not launch a SECOND full refresh — its own fresh LockDecision feeds the
    planner directly (the floor only budgets ONE cascade after the deadline)."""
    from bts.picks import save_pick
    from bts.scheduler import LockDecision, run_day
    from bts.strategy import PickResult

    clock = Clock(datetime(2026, 8, 30, 12, 40, tzinfo=ET))
    daily = _kwan_daily()
    save_pick(daily, tmp_path)

    def fake_check(**kw):                     # starts 12:40, overruns to 13:10 (> 13:05 deadline)
        clock.advance(timedelta(minutes=30))
        return {"skipped": False, "new_lineups": 4, "should_post": False,
                "pick_result": PickResult(daily=daily, locked=False),
                "pick_name": "Steven Kwan", "pick_p": 0.7566, "selection": None,
                "lock_decision": LockDecision(False, 0.741, False, "gap", 823987, (823987,))}

    def fake_sleep(secs):
        clock.advance(timedelta(seconds=secs))

    with patch("bts.scheduler.fetch_schedule", side_effect=[_schedule(), []]), \
         patch("bts.scheduler._now_et", side_effect=clock), \
         patch("bts.scheduler.time.sleep", side_effect=fake_sleep), \
         patch("bts.scheduler.run_single_check", side_effect=fake_check), \
         patch("bts.scheduler._refresh_pick_at_fallback_decision") as mock_refresh, \
         patch("bts.scheduler.count_new_confirmations", return_value=0), \
         patch("bts.scheduler.run_result_polling", return_value="final"), \
         patch("bts.scheduler._trigger_live_forward_capture_on_lock"), \
         patch("bts.dm.send_dm", return_value="dm-1") as mock_dm, \
         patch("bts.contest_state.load_decision_streak_state") as dss:
        dss.return_value.streak = 0
        run_day(date=DATE, config=_config(tmp_path))

    mock_refresh.assert_not_called()
    mock_dm.assert_called_once()
    delivered = json.loads((tmp_path / f"{DATE}.json").read_text())
    assert delivered["delivered_at"].startswith("2026-08-30T13:10")
    state = json.loads((tmp_path / DATE / "scheduler_state.json").read_text())
    assert state["fallback_refreshes"][0]["reused_check_cascade"] is True


def test_confirmation_sync_is_bounded_to_planner_relevant_games(tmp_path):
    """Codex r2 F4: the fallback syncs only the games in remaining scheduled runs
    (the only games the planner consults), not the whole slate."""
    from bts.scheduler import FallbackRefreshResult

    clock = Clock(datetime(2026, 8, 30, 12, 35, tzinfo=ET))
    calls = []

    def capture_cnc(pks, confirmed):
        calls.append(sorted(pks))
        return 0

    from bts.picks import save_pick
    from bts.scheduler import run_day
    from bts.strategy import PickResult

    daily = _kwan_daily()
    save_pick(daily, tmp_path)

    def fake_check(**kw):
        clock.advance(CASCADE)
        return {"skipped": False, "new_lineups": 4, "should_post": False,
                "pick_result": PickResult(daily=daily, locked=False),
                "pick_name": "Steven Kwan", "pick_p": 0.7566}

    def fake_refresh(config, date, cached, gap, **kw):
        clock.advance(CASCADE)
        return FallbackRefreshResult(daily=daily, should_post=False, should_post_ungated=False,
                                     block_reason="gap", contender_game_pk=823987)

    def fake_sleep(secs):
        clock.advance(timedelta(seconds=secs))

    with patch("bts.scheduler.fetch_schedule", side_effect=[_schedule(), []]), \
         patch("bts.scheduler._now_et", side_effect=clock), \
         patch("bts.scheduler.time.sleep", side_effect=fake_sleep), \
         patch("bts.scheduler.run_single_check", side_effect=fake_check), \
         patch("bts.scheduler._refresh_pick_at_fallback_decision", side_effect=fake_refresh), \
         patch("bts.scheduler.count_new_confirmations", side_effect=capture_cnc), \
         patch("bts.scheduler.run_result_polling", return_value="final"), \
         patch("bts.scheduler._trigger_live_forward_capture_on_lock"), \
         patch("bts.dm.send_dm", return_value="dm-1"), \
         patch("bts.contest_state.load_decision_streak_state") as dss:
        dss.return_value.streak = 0
        run_day(date=DATE, config=_config(tmp_path))

    assert calls, "fallback confirmation sync never ran"
    allowed = {823662, 823987, 824636}          # games of the remaining scheduled runs
    for pks in calls:
        assert set(pks) <= allowed and 824393 not in pks


def test_stale_refresh_result_never_sends_after_cutoff(tmp_path, capsys):
    """Even if the planner were to deliver late (here: a cascade that overran the
    cutoff), the chokepoint guard refuses — nothing is sent after 13:35."""
    from bts.scheduler import FallbackRefreshResult

    clock = Clock(datetime(2026, 8, 30, 12, 35, tzinfo=ET))

    def slow_refresh(daily):
        clock.advance(timedelta(minutes=20))               # on top of the 15.5-min cascade → 13:40:30
        return FallbackRefreshResult(daily=daily, should_post=True)

    with patch("bts.health.alert.dispatch_dm_for_health_alerts", return_value=True) as mock_alert:
        _, mock_dm = _run(tmp_path, clock, slow_refresh)
    mock_dm.assert_not_called()                              # no pick DM
    alerts = mock_alert.call_args.args[0]                    # but a CRITICAL refusal notice
    assert alerts[0].level == "CRITICAL" and alerts[0].source == "late_delivery"
    assert not (tmp_path / f"{DATE}.json").exists()
    assert list((tmp_path / DATE).glob("refused_delivery_*.json"))
    state = json.loads((tmp_path / DATE / "scheduler_state.json").read_text())
    assert state["pick_locked"] is False
    assert state["delivery_refusals"][0]["batter"] == "Steven Kwan"
    assert "DELIVERY REFUSED" in capsys.readouterr().err
