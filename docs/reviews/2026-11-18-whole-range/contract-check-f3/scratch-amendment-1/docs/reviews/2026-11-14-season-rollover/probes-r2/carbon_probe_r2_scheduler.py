"""Round-2 adversarial probe for S6 scheduler corrections."""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from bts.picks import DailyPick, Pick, save_pick
from bts.scheduler import FallbackRefreshResult, LockDecision, run_day
from bts.strategy import PickResult, SelectionResult

ET = ZoneInfo("America/New_York")


class Clock:
    def __init__(self, start: datetime):
        self.now = start
        self.mono = 0.0

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now = self.now + delta
        self.mono += delta.total_seconds()

    def sleep(self, secs: float) -> None:
        self.advance(timedelta(seconds=secs))


def _game(pk: int, hhmm: str, date: str = "2027-04-10", away: str = "BOS", home: str = "NYY") -> dict:
    h, m = map(int, hhmm.split(":"))
    dt_et = datetime.fromisoformat(f"{date}T00:00:00").replace(tzinfo=ET) + timedelta(hours=h, minutes=m)
    dt_utc = dt_et.astimezone(ZoneInfo("UTC"))
    return {
        "gamePk": pk,
        "gameType": "R",
        "officialDate": date,
        "gameDate": dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
        "teams": {
            "away": {"team": {"id": 111 + pk % 10, "name": away}},
            "home": {"team": {"id": 147 + pk % 10, "name": home}},
        },
    }


def _cfg(picks_dir: Path, health_enabled: bool = False) -> dict:
    return {
        "orchestrator": {"picks_dir": str(picks_dir), "heartbeat_path": str(picks_dir / ".hb")},
        "tiers": [],
        "bluesky": {"dm_recipient": "eric"},
        "scheduler": {
            "pick_delivery": "dm",
            "early_lock_gap": 0.03,
            "lineup_check_offset_min": 60,
            "cluster_min": 10,
            "doubleheader_recheck_min": 15,
            "fallback_deadline_min": 35,
            "fallback_deadline_min_morning": 25,
            "results_poll_interval_min": 15,
            "results_cap_hour_et": 5,
            "cascade_budget_min": 12,
            "operator_reserve_min": 10,
            "live_forward_capture_on_lock": False,
        },
        "health_checks": {"enabled": health_enabled},
    }


def _daily(date: str, pk: int, hhmm: str, name: str = "Batter A") -> DailyPick:
    g = _game(pk, hhmm, date=date)
    p = Pick(
        batter_name=name, batter_id=1000 + pk, team="BOS", lineup_position=1,
        pitcher_name="Pitcher", pitcher_id=2000 + pk, p_game_hit=0.76,
        flags=[], projected_lineup=False, game_pk=pk, game_time=g["gameDate"],
    )
    return DailyPick(date=date, run_time=f"{date}T18:00:00+00:00", pick=p, double_down=None, runner_up=None)


def test_r1_scheduler_closures() -> None:
    print("=== 1. Verify Round-1 Scheduler Findings (B2, B3, B4, B5, B8, e-j) ===")
    # B2/(e): Extra attempt with should_post=False reuses its own lock_decision (no 2nd cascade)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        date = "2027-04-10"
        clock = Clock(datetime(2027, 4, 10, 10, 0, tzinfo=ET))
        games = [_game(9001, "19:10", date=date, away="BOS", home="NYY"),
                 _game(9002, "19:10", date=date, away="LAD", home="SFG")]

        def fake_check(**kw):
            if kw.get("include_lock_decision"):
                clock.advance(timedelta(minutes=20))  # 18:35 -> 18:55 (< 19:05 cutoff!)
                d = _daily(date, 9001, "19:10")
                save_pick(d, p)
                return {
                    "skipped": False, "new_lineups": 2, "should_post": False,
                    "pick_result": PickResult(daily=d, locked=False),
                    "pick_name": d.pick.batter_name, "pick_p": 0.76, "selection": None,
                    "lock_decision": LockDecision(False, 0.75, False, "gap", 9002, (9002,)),
                }
            return {"skipped": False, "new_lineups": 0, "should_post": False,
                    "pick_result": None, "pick_name": None, "pick_p": None, "selection": None}

        with patch("bts.scheduler.fetch_schedule", side_effect=[games, []]), \
             patch("bts.scheduler._now_et", side_effect=clock), \
             patch("bts.scheduler._watchdog_ping_sleep", side_effect=clock.sleep), \
             patch("bts.scheduler.time.monotonic", side_effect=lambda: clock.mono), \
             patch("bts.scheduler.run_single_check", side_effect=fake_check), \
             patch("bts.scheduler._refresh_pick_at_fallback_decision") as mock_refresh, \
             patch("bts.scheduler._delivery_game_statuses", return_value={9001: {"detailed": "Scheduled"}, 9002: {"detailed": "Scheduled"}}), \
             patch("bts.scheduler.count_new_confirmations", return_value=0), \
             patch("bts.scheduler.run_result_polling", return_value="final"), \
             patch("bts.dm.send_dm", return_value="dm-1") as mock_dm, \
             patch("bts.contest_state.load_decision_streak_state") as dss:
            dss.return_value.streak = 0
            run_day(date=date, config=_cfg(p))
        assert mock_refresh.call_count == 0
        assert mock_dm.call_count == 1
        print("  B2/(e) extra attempt reuses lock_decision and delivers before cutoff: CLOSED")

    # B4/(g): All-postponed slate runs zero extra cascades and sends zero alerts
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        date = "2027-04-10"
        clock = Clock(datetime(2027, 4, 10, 10, 0, tzinfo=ET))
        games = [_game(9001, "19:10", date=date)]
        checks_at = []

        def fake_check(**kw):
            checks_at.append(clock.now.strftime("%H:%M"))
            return {"skipped": False, "new_lineups": 0, "should_post": False,
                    "pick_result": None, "pick_name": None, "pick_p": None, "selection": None}

        with patch("bts.scheduler.fetch_schedule", side_effect=[games, []]), \
             patch("bts.scheduler._now_et", side_effect=clock), \
             patch("bts.scheduler._watchdog_ping_sleep", side_effect=clock.sleep), \
             patch("bts.scheduler.run_single_check", side_effect=fake_check), \
             patch("bts.scheduler._delivery_game_statuses", return_value={9001: {"detailed": "Postponed"}}), \
             patch("bts.dm.send_dm", return_value="dm-1") as mock_dm:
            run_day(date=date, config=_cfg(p))
        assert "18:35" not in checks_at and mock_dm.call_count == 0
        print(f"  B4/(g) all-postponed slate: checks={checks_at}, DMs={mock_dm.call_count}: CLOSED")


def test_extra_attempt_void_race_aborts_later_playable_games() -> None:
    print("\n=== 2. Probe S6 Multi-Start Void Race on Extra Attempt ===")
    # Slate: 9001 at 19:10, 9002 at 19:20, 9003 at 20:10? Wait: with cluster_min=10,
    # 9001 (19:10) and 9002 (19:20) have ONE planned cluster check at 18:10 (plus thin-slate early checks).
    # Planned checks end by 18:10! Extra attempts are at 18:35 (for 9001) and 18:45 (for 9002)!
    # Suppose at 18:35, 9001 is Scheduled when _playable_game_pks & run_single_check run;
    # run_single_check selects Batter A from 9001 (should_post=True or False);
    # right before _deliver_and_lock_pick checks _void_committed_games, 9001 flips to Postponed!
    # _deliver_and_lock_pick refuses 9001 (game_not_playable) and deletes 2027-04-10.json!
    # Meanwhile 9002 (19:20) IS PLAYABLE and its lineup posts at 18:40 (before its 18:45 extra attempt)!
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        date = "2027-04-10"
        clock = Clock(datetime(2027, 4, 10, 10, 0, tzinfo=ET))
        games = [
            _game(9001, "19:10", date=date, away="BOS", home="NYY"),
            _game(9002, "19:20", date=date, away="LAD", home="SFG"),
        ]
        checks_at = []
        alerts_sent = []
        status_state = {9001: "Scheduled", 9002: "Scheduled"}

        def fake_statuses(_date: str):
            return {pk: {"detailed": st, "abstract": "F" if st == "Postponed" else "P"}
                    for pk, st in status_state.items()}

        def fake_check(**kw):
            t_str = clock.now.strftime("%H:%M")
            checks_at.append(t_str)
            if t_str == "18:35":
                d = _daily(date, 9001, "19:10", name="Postponed Game Batter")
                save_pick(d, p)
                # Game 9001 postpones right as the cascade finishes (before delivery guard)!
                status_state[9001] = "Postponed"
                return {
                    "skipped": False, "new_lineups": 1, "should_post": True,
                    "pick_result": PickResult(daily=d, locked=False),
                    "pick_name": d.pick.batter_name, "pick_p": 0.76, "selection": None,
                    "lock_decision": LockDecision(True, None, True, None, None, ()),
                }
            if t_str == "18:45":
                d = _daily(date, 9002, "19:20", name="Playable Game Batter")
                save_pick(d, p)
                return {
                    "skipped": False, "new_lineups": 1, "should_post": True,
                    "pick_result": PickResult(daily=d, locked=False),
                    "pick_name": d.pick.batter_name, "pick_p": 0.76, "selection": None,
                    "lock_decision": LockDecision(True, None, True, None, None, ()),
                }
            return {"skipped": False, "new_lineups": 0, "should_post": False,
                    "pick_result": None, "pick_name": None, "pick_p": None, "selection": None}

        with patch("bts.scheduler.fetch_schedule", side_effect=[games, []]), \
             patch("bts.scheduler._now_et", side_effect=clock), \
             patch("bts.scheduler._watchdog_ping_sleep", side_effect=clock.sleep), \
             patch("bts.scheduler.run_single_check", side_effect=fake_check), \
             patch("bts.scheduler._delivery_game_statuses", side_effect=fake_statuses), \
             patch("bts.scheduler.count_new_confirmations", return_value=0), \
             patch("bts.scheduler.run_result_polling", return_value="final"), \
             patch("bts.dm.send_dm", side_effect=lambda rec, txt: alerts_sent.append((clock.now.strftime("%H:%M"), txt)) or "dm-1"), \
             patch("bts.contest_state.load_decision_streak_state") as dss:
            dss.return_value.streak = 0
            run_day(date=date, config=_cfg(p))

        print(f"  Checks executed: {checks_at} (18:45 for playable 9002 NEVER RAN!)")
        print(f"  Pick on disk: {(p / f'{date}.json').exists()}")
        print(f"  DMs sent: {alerts_sent}")


def test_doubleheader_final_game2_in_playable_filter() -> None:
    print("\n=== 3. Probe Doubleheader Placeholder 23:33 ET Game 2 Already Final ===")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        date = "2027-04-10"
        clock = Clock(datetime(2027, 4, 10, 10, 0, tzinfo=ET))
        # Traditional DH: Game 1 at 13:10, Game 2 placeholder 23:33 ET (actually played at 16:40 and is Final by 19:30!)
        games = [
            _game(9001, "13:10", date=date, away="BOS", home="NYY"),
            _game(9002, "23:33", date=date, away="BOS", home="NYY"),
        ]
        checks_at = []
        alerts_sent = []

        def fake_statuses(_date: str):
            if clock.now >= datetime(2027, 4, 10, 19, 30, tzinfo=ET):
                return {
                    9001: {"detailed": "Final", "abstract": "F"},
                    9002: {"detailed": "Final", "abstract": "F"},
                }
            return {
                9001: {"detailed": "Scheduled", "abstract": "P"},
                9002: {"detailed": "Scheduled", "abstract": "P"},
            }

        def fake_check(**kw):
            checks_at.append((clock.now.strftime("%H:%M"), kw.get("include_lock_decision", False)))
            return {"skipped": False, "new_lineups": 0, "should_post": False,
                    "pick_result": None, "pick_name": None, "pick_p": None, "selection": None}

        with patch("bts.scheduler.fetch_schedule", side_effect=[games, []]), \
             patch("bts.scheduler._now_et", side_effect=clock), \
             patch("bts.scheduler._watchdog_ping_sleep", side_effect=clock.sleep), \
             patch("bts.scheduler.run_single_check", side_effect=fake_check), \
             patch("bts.scheduler._delivery_game_statuses", side_effect=fake_statuses), \
             patch("bts.scheduler.count_new_confirmations", return_value=0), \
             patch("bts.dm.send_dm", side_effect=lambda rec, txt: alerts_sent.append((clock.now.strftime("%H:%M"), txt)) or "dm-1"):
            run_day(date=date, config=_cfg(p))

        print(f"  DH checks (time, is_extra): {checks_at}")
        print(f"  DMs sent: {alerts_sent}")


if __name__ == "__main__":
    test_r1_scheduler_closures()
    test_extra_attempt_void_race_aborts_later_playable_games()
    test_doubleheader_final_game2_in_playable_filter()
