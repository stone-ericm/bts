"""Adversarial probes for 9998bbf (thin-slate plan_run_times) & 6467cfc (run_day fallback/EOD)."""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from bts.daily_decision import load_decision
from bts.day_outcome import gather
from bts.picks import DailyPick, Pick, load_pick, save_pick
from bts.scheduler import plan_run_times, run_day
from bts.strategy import PickResult, SelectionResult

ET = ZoneInfo("America/New_York")


def _game(pk: int, time_et_iso: str, dh: str = "N", gnum: int = 1, away_name: str = "NYY", home_name: str = "BOS") -> dict:
    dt = datetime.fromisoformat(time_et_iso)
    utc = dt.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "gamePk": pk,
        "gameType": "R",
        "doubleHeader": dh,
        "gameNumber": gnum,
        "gameDate": utc,
        "officialDate": time_et_iso[:10],
        "teams": {"away": {"team": {"id": pk, "name": away_name}}, "home": {"team": {"id": pk + 1000, "name": home_name}}},
        "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
    }


def _daily(date: str, pk: int, time_et_iso: str, name: str = "Judge", delivered: bool = False) -> DailyPick:
    dt = datetime.fromisoformat(time_et_iso)
    utc = dt.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
    return DailyPick(
        date=date, run_time="11:10 ET",
        pick=Pick(
            batter_name=name, batter_id=592450, team="NYY", lineup_position=2,
            pitcher_name="Pitcher", pitcher_id=999, p_game_hit=0.79,
            flags=[], projected_lineup=True, game_pk=pk, game_time=utc,
        ),
        double_down=None, runner_up=None,
        delivered_at=f"{date}T12:00:00-04:00" if delivered else None,
        notification_sent=delivered,
        notification_channel="bluesky_dm" if delivered else None,
    )


def probe_plan_run_times_unmerged_collision():
    """Check whether plan_run_times leaves regular checks and early checks unmerged within cluster_min."""
    d = "2027-05-01"
    games = [
        _game(101, f"{d}T13:10:00-04:00"),
        _game(202, f"{d}T14:25:00-04:00"),
    ]
    runs = plan_run_times(games, offset_min=45, cluster_min=10, now=datetime(2027, 5, 1, 9, 0, tzinfo=ET))
    print("[PROBE 4A plan_run_times collision (13:10 & 14:25 games)]")
    for r in runs:
        print(f"  {r['time_et'].strftime('%H:%M ET')} -> {r['game_pks']}")
    gaps = [((runs[i + 1]["time_et"] - runs[i]["time_et"]).total_seconds() / 60.0)
            for i in range(len(runs) - 1)]
    print(f"  adjacent gaps (min): {gaps} (cluster_min=10!)")


def probe_early_game_check_returns_none_skips_inloop_fallback():
    """On a 2-game slate (13:10 & 19:10), early check selects Game 1 (13:10, fallback 12:45),
    then 12:25 regular check returns pick_result=None (e.g. ContestStateError or no eligible pick)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        picks_dir = root / "picks"
        picks_dir.mkdir()
        d = "2027-05-01"
        games = [
            _game(101, f"{d}T13:10:00-04:00", away_name="NYY", home_name="BOS"),
            _game(202, f"{d}T19:10:00-04:00", away_name="LAD", home_name="SFG"),
        ]
        clock = [datetime(2027, 5, 1, 10, 0, tzinfo=ET)]
        deliveries: list[tuple[datetime, int]] = []
        check_calls: list[datetime] = []

        def fake_now():
            return clock[0]

        def fake_sleep(secs):
            if secs > 0:
                clock[0] = clock[0] + timedelta(seconds=secs)

        def fake_run_single_check(*, date, all_game_pks, confirmed_sides, config, early_lock_gap, unavailable_game_pks=None):
            t = clock[0]
            check_calls.append(t)
            if t.hour == 10:  # 10:10 early check selects projected Game 1 pick
                dp = _daily(date, 101, f"{d}T13:10:00-04:00", name="EarlyGame1Batter")
                save_pick(dp, picks_dir)
                return {
                    "skipped": False, "new_lineups": 1, "should_post": False,
                    "pick_result": PickResult(daily=dp, locked=False),
                    "pick_name": dp.pick.batter_name, "pick_p": 0.79,
                    "selection": SelectionResult(PickResult(daily=dp), "single", "mdp", {"batter_id": 592450}, None, None),
                }
            if t.hour == 12:  # 12:25 regular check fails/returns None!
                return {
                    "skipped": False, "new_lineups": 0, "should_post": False,
                    "pick_result": None, "pick_name": None, "pick_p": None,
                    "selection": None,
                }
            # Afternoon checks run at 16:10+ (long after Game 1 started at 13:10!)
            dp2 = _daily(date, 202, f"{d}T19:10:00-04:00", name="LateGame2Batter")
            save_pick(dp2, picks_dir)
            return {
                "skipped": False, "new_lineups": 1, "should_post": False,
                "pick_result": PickResult(daily=dp2, locked=False),
                "pick_name": dp2.pick.batter_name, "pick_p": 0.78,
                "selection": SelectionResult(PickResult(daily=dp2), "single", "mdp", {"batter_id": 592450}, None, None),
            }

        def fake_deliver(daily, config, pdir, state, date, label, *, selection=None):
            deliveries.append((clock[0], daily.pick.game_pk))
            daily.delivered_at = clock[0].isoformat()
            daily.notification_sent = True
            save_pick(daily, pdir)
            state.pick_locked = True
            state.committed_pick_written = True
            return True

        config = {
            "orchestrator": {"picks_dir": str(picks_dir), "heartbeat_path": str(root / ".hb")},
            "scheduler": {"pick_delivery": "dm", "shadow_model": False, "entry_intent": "enter"},
            "health_checks": {"enabled": False},
        }
        with patch("bts.scheduler.fetch_schedule", return_value=games), \
             patch("bts.scheduler._now_et", side_effect=fake_now), \
             patch("bts.scheduler._watchdog_ping_sleep", side_effect=fake_sleep), \
             patch("bts.scheduler.write_heartbeat"), \
             patch("bts.scheduler.notify_watchdog"), \
             patch("bts.scheduler.run_single_check", side_effect=fake_run_single_check), \
             patch("bts.scheduler._deliver_and_lock_pick", side_effect=fake_deliver), \
             patch("bts.scheduler.run_result_polling", return_value="final"), \
             patch("bts.scheduler._next_day_wakeup", return_value=datetime(2027, 5, 2, 10, 0, tzinfo=ET)), \
             patch("bts.scheduler._idle_until_next_wakeup"):
            run_day(d, config)

        print("[PROBE 4B 10:10 early check selects Game 1 (13:10), 12:25 check returns None]")
        print(f"  checks ran at: {[t.strftime('%H:%M') for t in check_calls]}")
        print(f"  deliveries: {[(t.strftime('%H:%M'), pk) for t, pk in deliveries]}")


def probe_dh_skip_day_blocks_write_endofday_skip_past_2350():
    """On a DH Game-2 skip day with evening check ending ~21:30 ET, step 6 sleeps 150 min before _write_endofday_skip."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        picks_dir = root / "picks"
        picks_dir.mkdir()
        d = "2027-05-01"
        games = [
            _game(101, f"{d}T16:10:00-04:00", dh="S", gnum=1),
            _game(102, f"{d}T22:10:00-04:00", dh="S", gnum=2),
        ]
        clock = [datetime(2027, 5, 1, 13, 0, tzinfo=ET)]
        decision_written_at: list[datetime] = []

        def fake_now():
            return clock[0]

        def fake_sleep(secs):
            if secs > 0:
                clock[0] = clock[0] + timedelta(seconds=secs)

        def fake_run_single_check(*, date, all_game_pks, confirmed_sides, config, early_lock_gap, unavailable_game_pks=None):
            confirmed_sides.add((101, "away"))
            confirmed_sides.add((101, "home"))
            sel = SelectionResult(
                pick_result=None, action="skip", source="mdp",
                primary_candidate={"batter_id": 1, "batter_name": "X", "p_game_hit": 0.72},
                double_candidate=None, no_pick_reason="below_bar",
                streak=0, saver_available=True, state_source="contest",
                state_status="trusted", allow_double=True, contest_source_date=date,
            )
            return {
                "skipped": False, "new_lineups": 0, "should_post": False,
                "pick_result": None, "pick_name": None, "pick_p": None,
                "skip_summary": {"best_batter": "X", "best_team": "NYY", "best_p": 0.72, "streak": 0, "bar": 0.75},
                "selection": sel,
            }

        orig_write_skip = __import__("bts.scheduler", fromlist=["_write_endofday_skip"])._write_endofday_skip

        def spy_write_skip(pdir, date, state):
            decision_written_at.append(clock[0])
            return orig_write_skip(pdir, date, state)

        config = {
            "orchestrator": {"picks_dir": str(picks_dir), "heartbeat_path": str(root / ".hb")},
            "scheduler": {"pick_delivery": "dm", "shadow_model": False, "entry_intent": "enter", "dh_recheck_min": 15},
            "bluesky": {},
            "health_checks": {"enabled": False},
        }
        with patch("bts.scheduler.fetch_schedule", return_value=games), \
             patch("bts.scheduler._now_et", side_effect=fake_now), \
             patch("bts.scheduler._watchdog_ping_sleep", side_effect=fake_sleep), \
             patch("bts.scheduler.write_heartbeat"), \
             patch("bts.scheduler.notify_watchdog"), \
             patch("bts.scheduler.run_single_check", side_effect=fake_run_single_check), \
             patch("bts.scheduler.count_new_confirmations", return_value=0), \
             patch("bts.scheduler._write_endofday_skip", side_effect=spy_write_skip), \
             patch("bts.scheduler._next_day_wakeup", return_value=datetime(2027, 5, 2, 10, 0, tzinfo=ET)), \
             patch("bts.scheduler._idle_until_next_wakeup"):
            run_day(d, config)

        print("[PROBE 4C DH Game-2 recheck delays _write_endofday_skip past 23:50 day-outcome]")
        print(f"  _write_endofday_skip called at: {decision_written_at[0].isoformat()}")
        # At 23:50 before _write_endofday_skip finished:
        # Remove decision briefly to evaluate exact 23:50 state before 23:55 write:
        dec = load_decision(d, picks_dir)
        print(f"  final decision on disk: {dec and dec.get('action')} / {dec and dec.get('source')}")
        if decision_written_at[0].hour == 23 and decision_written_at[0].minute >= 50:
            (picks_dir / d / "decision.json").unlink()
            with patch("bts.scheduler.fetch_schedule", return_value=games):
                outcome_at_2350 = gather(d, picks_dir, config, expected_username="stonehengee")
            print(f"  day_outcome at 23:50 ET: state={outcome_at_2350.state}, ok={outcome_at_2350.ok}")


if __name__ == "__main__":
    probe_plan_run_times_unmerged_collision()
    probe_early_game_check_returns_none_skips_inloop_fallback()
    probe_dh_skip_day_blocks_write_endofday_skip_past_2350()
