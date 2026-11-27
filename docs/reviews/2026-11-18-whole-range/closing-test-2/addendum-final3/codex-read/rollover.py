import inspect
import json
from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from pytest import MonkeyPatch

import bts
from bts import contest_state, dm, orchestrator, picks, scheduler, strategy
from bts.health import alert, runner
from scripts.audit.season_walk import walk


ET = scheduler.ET


def clock(day, hour, minute):
    return datetime.fromisoformat(day).replace(tzinfo=ET, hour=hour, minute=minute)


class Day:
    def __init__(self, parent, start, game_times, mp):
        self.day = start.date().isoformat()
        self.picks = parent / "relative config" / " live picks \\name "
        self.picks.mkdir(parents=True)
        self.config = walk._config(self.picks)
        self.config["bluesky"] = {"dm_recipient": "capture.invalid"}
        self.config["orchestrator"]["heartbeat_path"] = str(parent / "heartbeat")
        self.config["scheduler"]["live_forward_capture_on_lock"] = False
        self.config["health_checks"]["enabled"] = False
        self.receipt = self.picks.parent / "health_state" / "health_dm_delivery_status.json"
        self.dispatches = []
        self.health_dms = []
        self.pick_dms = []
        self.predictions = walk._predictions(start.date(), .94).iloc[:len(game_times)].copy()
        self.predictions["game_time"] = [game.isoformat() for game in game_times]
        self.statuses = {pk: {"abstract": "P", "detailed": "Scheduled", "code": "S"}
                         for pk in self.predictions.game_pk}
        self.games = [{"gamePk": pk, "gameDate": game.isoformat(), "gameType": "R",
                       "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
                       "teams": {"away": {"team": {"name": str(pk)}}, "home": {"team": {"name": "Home"}}}}
                      for pk, game in zip(self.predictions.game_pk, game_times)]
        self.cascade = Mock(return_value=(self.predictions, "synthetic"))
        self.on_sleep = lambda old, new: None
        self.on_idle = lambda: None
        self.delivery_status = Mock(side_effect=lambda _: self.statuses)
        mp.setattr(scheduler, "fetch_schedule", lambda day: self.games if self.day == day else [])
        mp.setattr(scheduler, "check_confirmed_lineups", lambda *a: {})
        mp.setattr(scheduler, "heartbeat_watchdog", lambda *a, **kw: nullcontext())
        mp.setattr(scheduler, "_watchdog_ping_sleep", self.sleep)
        mp.setattr(scheduler, "_idle_until_next_wakeup", lambda *a: self.on_idle())
        mp.setattr(scheduler, "notify_watchdog", Mock())
        mp.setattr(scheduler, "notify_ready", Mock())
        mp.setattr(scheduler, "run_result_polling", Mock(return_value="final"))
        mp.setattr(orchestrator, "run_cascade", self.cascade)
        mp.setattr(picks, "get_game_statuses_detailed", lambda *_: self.statuses)
        mp.setattr(strategy, "get_game_statuses_detailed", lambda *_: self.statuses)
        mp.setattr(scheduler, "_delivery_game_statuses", self.delivery_status)
        mp.setattr(dm, "send_dm", self.pick_dm)
        mp.setattr(alert, "send_dm", self.health_dm)
        original = alert.dispatch_dm_for_health_alerts

        def record(alerts, recipient, *args, **kwargs):
            if kwargs.get("status_path") != self.receipt:
                return original(alerts, recipient, *args, **kwargs)
            frame = inspect.currentframe().f_back
            caller = f"{frame.f_code.co_name}<-{frame.f_back.f_code.co_name}:{frame.f_back.f_lineno}"
            before = len(self.health_dms)
            attempted = original(alerts, recipient, *args, **kwargs)
            receipt = self.read_receipt()
            self.dispatches.append({"sources": [a.source for a in alerts], "caller": caller,
                                    "wall": scheduler._now_et().isoformat(), "arg_day": str(kwargs.get("now_et_date")),
                                    "sent_day": receipt.get("sent_day"), "attempt": attempted,
                                    "wire": len(self.health_dms) > before and receipt.get("status") == "sent"})
            return attempted

        mp.setattr(alert, "dispatch_dm_for_health_alerts", record)
        mp.setattr(runner, "dispatch_dm_for_health_alerts", record)

    def read_receipt(self):
        return json.loads(self.receipt.read_text()) if self.receipt.exists() else {}

    def sleep(self, seconds):
        old = scheduler._now_et()
        new = old + timedelta(seconds=seconds)
        walk._set_clock(new)
        self.on_sleep(old, new)

    def health_dm(self, recipient, body):
        self.health_dms.append(body)
        if len(body) > 1000:
            raise ValueError("DM exceeded chat limit")
        return "health-capture"

    def pick_dm(self, recipient, body):
        self.pick_dms.append(body)
        return "pick-capture"

    def write_bad_observation(self, filename="contest_streak.json"):
        path = self.picks / "account_state" / filename
        path.parent.mkdir(exist_ok=True)
        path.write_text("[]")

    def file_today(self, streak=57):
        rc, msg = walk._invoke("set-contest-streak", "--date", scheduler._now_et().date(), "--streak", streak,
                               "--best-streak", streak, "--saver", "used", "--picks-dir", self.picks)
        assert rc == 0, msg

    def seed(self):
        sel = strategy.select_pick(self.predictions, self.day, self.picks, 57, False,
                                   game_statuses_detailed=self.statuses,
                                   best_streak=57, best_status="trusted")
        assert sel.pick_result is not None
        daily = sel.pick_result.daily
        picks.save_pick(daily, self.picks)
        return daily

    def run(self):
        scheduler.run_day(self.day, self.config)


@contextmanager
def scenario(parent, start, game_times):
    with walk.offline(start), MonkeyPatch.context() as mp:
        # Keep the synthetic December games out of the independent K2 change.
        if start.month == 12:
            mp.setitem(strategy.SEASON_END_DATES, start.year, f"{start.year + 1}-01-02")
        yield Day(parent, start, game_times, mp), mp


def run_case(day, name, parent):
    start = clock(day, 23, 58)
    midnight = start + timedelta(minutes=2)
    late_games = [midnight + timedelta(minutes=42), midnight + timedelta(minutes=69)]
    games = late_games if name in {"check", "in_loop"} else late_games[:1]
    with scenario(parent, start, games) as (sim, mp):
        if name == "check":
            sim.write_bad_observation()
        elif name == "in_loop":
            sim.file_today()
            mp.setattr(scheduler, "_lock_decision_from_predictions", lambda *a, **kw: scheduler.LockDecision(False, None, False))
            sim.on_sleep = lambda old, new: sim.write_bad_observation("contest_streak.manual.json") if old.date() < new.date() else None
        else:
            sim.seed()
            sim.cascade.return_value = (None, None)
            if name in {"final", "loader"}:
                sim.write_bad_observation()
            if name == "final":
                sim.cascade.side_effect = [(None, None), (sim.predictions, "final-fallback")]
            if name == "void":
                sim.delivery_status.return_value = None
                sim.delivery_status.side_effect = lambda _: {pk: {"detailed": "Postponed"} for pk in sim.statuses}
            if name == "late":
                def status_after_cutoff(_):
                    walk._set_clock(games[0] - timedelta(minutes=4))
                    return sim.statuses
                sim.delivery_status.side_effect = status_after_cutoff
            if name == "saved":
                sim.on_sleep = lambda old, new: sim.file_today() if old.date() < new.date() else None
        sim.run()
        print(f"TRACE {day} {name} {json.dumps(sim.dispatches)}", flush=True)
        by_source = {source for row in sim.dispatches for source in row["sources"]}
        expected = {"late": "late_delivery", "void": "postponed_pick"}.get(name, "contest_state")
        assert expected in by_source
        assert not sim.pick_dms
        if name in {"in_loop", "final", "loader", "saved"}:
            assert "missed_pick" in by_source
            assert max(row["wall"] for row in sim.dispatches if "missed_pick" in row["sources"]) > midnight.isoformat()
        if name == "check":
            # The following date's own lineup check should have its own receipt.
            sim.day = midnight.date().isoformat()
            following = midnight + timedelta(hours=18)
            walk._set_clock(following)
            sim.games = [sim.games[0] | {"gameDate": (following + timedelta(hours=1)).isoformat()}]
            sim.run()
            print(f"NEXT {day} {json.dumps(sim.dispatches[-1])}", flush=True)
        if name in {"late", "void"}:
            # The actual EOD digest runs the two delivery checks, using the archived refusal.
            original = runner._safe_run
            mp.setattr(runner, "_safe_run", lambda key, fn: original(key, fn) if key in {"late_delivery", "postponed_pick"} else [])
            runner.run_all_checks(sim.picks, parent / "models", "capture.invalid", today=start.date())
            print(f"EOD {day} {name} {json.dumps(sim.dispatches[-1])}", flush=True)


def distinct_refusals(day, parent):
    start = clock(day, 23, 58)
    later = start + timedelta(minutes=26)
    with scenario(parent, start, [start + timedelta(minutes=44), later + timedelta(minutes=45)]) as (sim, _):
        state = walk.State("automatic", 57, 57, "used", .94)
        assert not walk.install_state(sim.picks, start.date(), state)
        walk._set_clock(start)
        sim.cascade.side_effect = [(sim.predictions.iloc[[i]], "synthetic") for i in (0, 1)]

        def last_status(_):
            picked = picks.load_pick(day, sim.picks).pick.game_pk
            detail = "Postponed" if picked == sim.games[0]["gamePk"] else "Cancelled"
            return {picked: {"detailed": detail}}

        sim.delivery_status.side_effect = last_status
        sim.run()
        refused = scheduler.load_state(day, sim.picks).delivery_refusals
        assert [r["void_games"] for r in refused] == [
            {str(sim.games[0]["gamePk"]): "Postponed"}, {str(sim.games[1]["gamePk"]): "Cancelled"},
        ]
        assert len(sim.dispatches) == 2 and not sim.pick_dms
        assert all(r["sources"] == ["postponed_pick"] for r in sim.dispatches)
        assert [r["wall"] for r in sim.dispatches] == [start.isoformat(), later.isoformat()]
        print(f"DISTINCT {day} {json.dumps(sim.dispatches)}", flush=True)
        print(f"DISTINCT WIRE {day} {json.dumps(sim.health_dms)}", flush=True)


def digest_then_first_refusal(day, parent):
    start = clock(day, 23, 46)
    with scenario(parent, start, [start + timedelta(minutes=44)]) as (sim, mp):
        original = runner._safe_run
        mp.setattr(runner, "_safe_run", lambda key, fn: original(key, fn) if key == "contest_state" else [])
        mp.setattr(runner, "read_systemd_nrestarts", lambda: None)
        sim.config["health_checks"]["enabled"] = True
        sim.write_bad_observation()
        sim.cascade.return_value = (None, None)
        sim.run()
        assert len(sim.dispatches) == 1 and sim.dispatches[0]["caller"].startswith("run_all_checks<-")
        assert sim.dispatches[0]["sources"] == ["contest_state"] and sim.dispatches[0]["wire"]

        walk._set_clock(clock(day, 23, 59))

        def cascade(*_):
            walk._set_clock(start + timedelta(minutes=15))
            (sim.picks / "account_state" / "contest_streak.json").write_text('{"active_streak": -1}')
            return sim.predictions, "synthetic"

        sim.cascade.side_effect = cascade
        sim.run()
        assert sim.dispatches[1]["caller"].startswith("_alert_contest_state_failure<-run_single_check:")
        assert sim.dispatches[1]["wall"] == (start + timedelta(minutes=15)).isoformat()
        assert sim.dispatches[1]["sources"] == ["contest_state"] and not sim.pick_dms
        print(f"DIGEST FIRST {day} {json.dumps(sim.dispatches)}", flush=True)
        print(f"DIGEST FIRST WIRE {day} {json.dumps(sim.health_dms)}", flush=True)


if __name__ == "__main__":
    print(f"bts.__file__={bts.__file__}", flush=True)
    for date in ("2027-06-14", "2027-12-31"):
        for name in ("check", "in_loop", "final", "loader", "saved", "late", "void"):
            with TemporaryDirectory() as temp:
                run_case(date, name, Path(temp))
    with TemporaryDirectory() as temp:
        distinct_refusals("2027-06-14", Path(temp))
    with TemporaryDirectory() as temp:
        digest_then_first_refusal("2027-06-14", Path(temp))
