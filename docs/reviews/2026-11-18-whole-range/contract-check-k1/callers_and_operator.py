"""Exercise all refusal callers, status output and a real cross-midnight fallback."""

import contextlib
import inspect
import io
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from pytest import MonkeyPatch
from unittest.mock import Mock

from bts import contest_state, dm, orchestrator, picks, scheduler, strategy
from bts.daily_decision import load_decision
from bts.health import alert
from scripts.audit.season_walk.walk import ET, _invoke, _set_clock, offline
from tests.test_pick_chosen_on_cs5a import DAY, daily_pick, set_dated
from tests.test_saved_pick_paths_cs5a import prepare_run_day, scheduler_state, world


SCRATCH = sys.argv[1] == "scratch"


def command(cmd, directory, *args):
    rc, text = _invoke("season-contract", cmd, "--picks-dir", directory, *args)
    assert rc == 0, text
    return text


for case in ("check", "revoked_at_check", "delivery_loader", "saved_changed", "in_loop", "final_midnight"):
    with TemporaryDirectory() as temp, MonkeyPatch.context() as mp, contextlib.redirect_stderr(io.StringIO()) as err:
        directory0 = Path(temp)
        cfg, directory, now, statuses = world.__wrapped__(directory0, mp)
        mp.setattr(strategy, "_mdp_cache", None)
        mp.setattr(scheduler, "notify_ready", lambda: None)
        mp.setattr(scheduler, "notify_watchdog", lambda: None)
        prepare_run_day(mp)
        day = DAY
        cached = daily_pick()
        sent = []
        routes = []

        def journal_cause():
            rows = err.getvalue().splitlines()
            return [line for line in rows if "CONTEST STATE ERROR" in line or "FALLBACK BLOCKED" in line]

        def send(_, text):
            assert journal_cause()
            sent.append(text)
            if len(text) > 1000:
                raise ValueError("length rejected")
            return "wire"

        dispatch = alert.dispatch_dm_for_health_alerts

        def observe(*args, **kwargs):
            caller = inspect.currentframe().f_back
            if caller.f_code.co_name == "_alert_contest_state_failure":
                parent = caller.f_back
                routes.append((parent.f_code.co_name, parent.f_lineno, kwargs.get("now_et_date")))
            return dispatch(*args, **kwargs)

        mp.setattr(alert, "send_dm", send)
        mp.setattr(alert, "dispatch_dm_for_health_alerts", observe)
        if case == "check":
            scheduler.run_single_check(day, [101, 102], set(), cfg, 0.03)
        elif case == "revoked_at_check":
            now[0] -= timedelta(hours=8)
            command("enable", directory, "--season", 2027)
            now[0] += timedelta(hours=8)
            command("revoke", directory, "--season", 2027, "--reason", "operator rehearsal")
            scheduler.run_single_check(day, [101, 102], set(), cfg, 0.03)
        elif case in ("delivery_loader", "saved_changed"):
            if case == "saved_changed":
                set_dated(directory, streak=12, best=16)
                old = contest_state.load_decision_streak_state(directory, require_contest_state=True)
                cached.chosen_on = picks.chosen_on_record(old, required=True)
                set_dated(directory, streak=13, best=16)
            path = picks.save_pick(cached, directory)
            original = path.read_bytes()
            state = scheduler_state()
            assert not scheduler._deliver_and_lock_pick(cached, cfg, directory, state, day, "lineup")
            assert original == path.read_bytes() and not state.pick_locked and load_decision(day, directory) is None
            dm.send_dm.assert_not_called()
        else:
            cfg["scheduler"]["entry_intent"] = "enter"
            if case == "in_loop":
                set_dated(directory)
                mp.setattr(scheduler, "_lock_decision_from_predictions",
                           lambda *a, **kw: scheduler.LockDecision(False, None, False))
                now[0] -= timedelta(minutes=10)
                frame, tier = orchestrator.run_cascade.return_value

                def predict(*_):
                    if orchestrator.run_cascade.call_count > 1:
                        (directory / "account_state/contest_streak.manual.json").unlink(missing_ok=True)
                    return frame, tier

                orchestrator.run_cascade.side_effect = predict
            else:
                games = scheduler.fetch_schedule(day)
                midnight = datetime.fromisoformat(day).replace(tzinfo=ET) + timedelta(days=1)
                for i, game in enumerate(games):
                    game["gameDate"] = (midnight + timedelta(minutes=40 + 30 * i)).isoformat()
                cached.pick.game_time = games[0]["gameDate"]
                now[0] = midnight - timedelta(minutes=5)
                scheduler.fetch_schedule.side_effect = lambda date: games if date == day else []
                frame, tier = orchestrator.run_cascade.return_value
                orchestrator.run_cascade.side_effect = [(None, None), (None, None), (frame, tier)]

                def advance(seconds):
                    now[0] += timedelta(seconds=seconds)

                mp.setattr(scheduler, "_watchdog_ping_sleep", advance)
            picks.save_pick(cached, directory)
            scheduler.run_day(day, cfg)

        before = {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()} if directory.exists() else {}
        status = command("status", directory)
        after = {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()} if directory.exists() else {}
        assert after == before
        live, = [line for line in status.splitlines() if line.startswith("live decision now:")]
        assert routes and sent
        if case == "saved_changed":
            assert "chosen on dated" in journal_cause()[0] and "accepted" in live and "streak=13" in live
        else:
            assert "refused" in live
            assert "Alternative dated path" in status and "Alternative automatic path" in status
        if case == "final_midnight":
            assert all(name == "run_day" for name, _, _ in routes)
            assert now[0].date().isoformat() > day
        if SCRATCH:
            assert all(d.isoformat() == day for _, _, d in routes)
        print("JOURNAL", case, json.dumps(journal_cause()))
        print("ROUTE", case, routes, now[0].isoformat(), len(sent[0]), live)
        if case == "final_midnight":
            print("FINAL", [line for line in err.getvalue().splitlines()
                            if "Morning state" in line or "sleeping until the fallback" in line])


# E4 through CLI enable + real loader, with the ordinary pre-fetch refusal that fits the base.
d = "2027-06-14"
tomorrow = "2027-06-15"
with TemporaryDirectory() as temp, offline(datetime(2027, 6, 14, 23, 58, tzinfo=ET)):
    directory = Path(temp) / "data/picks"
    command("enable", directory, "--season", 2027)
    cfg = {"orchestrator": {"picks_dir": str(directory)}, "bluesky": {"dm_recipient": "op.invalid"}}
    sent = []

    def wire(_, text):
        assert len(text) < 1000
        sent.append(text)
        return "wire"

    with MonkeyPatch.context() as mp:
        mp.setattr(alert, "send_dm", wire)
        for day, hour, minute in ((d, 0, 2), (tomorrow, 8, 0)):
            _set_clock(datetime(2027, 6, 15, hour, minute, tzinfo=ET))
            try:
                contest_state.load_decision_streak_state(directory, require_contest_state=True)
            except contest_state.ContestStateError as error:
                assert "no automatic observation" in str(error)
                scheduler._alert_contest_state_failure(cfg, error, **({"slate_date": day} if SCRATCH else {}))
            record = directory.parent / "health_state/health_dm_delivery_status.json"
            print("E4 REAL", day, len(sent), json.loads(record.read_text())["sent_day"])
    assert len(sent) == (2 if SCRATCH else 1)
