"""Compare real scheduler entry points and trace their ET dates across rollover."""

import inspect
import json
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, time as day_time, timedelta
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from pytest import MonkeyPatch

import bts
from bts import contest_state, dm, orchestrator, picks, posting, scheduler, strategy
from bts.health import alert
from scripts.audit.season_walk.walk import ET, _set_clock, offline
from tests.test_pick_chosen_on_cs5a import DAY, NOW, account_file, daily_pick
from tests.test_saved_pick_paths_cs5a import prepare_run_day, scheduler_state, world


def set_dated(directory, now, streak=7, best=9):
    day = now.date()
    account_file(directory, "contest_streak.manual.json", {
        "schema_version": contest_state.DATED_MANUAL_SCHEMA,
        "effective_date": day.isoformat(),
        "expires_at": datetime.combine(day, day_time.max, ET).isoformat(),
        "season": day.year, "active_streak": streak, "best_streak": best,
        "saver_state": "used", "recorded_at": now.isoformat(), "source": "MLB",
    })


def replay(case, requested_day, midnight, overnight):
    with TemporaryDirectory(prefix="wrk1-caller-") as temp, offline(NOW), MonkeyPatch.context() as mp:
        tmp = Path(temp)
        config, directory, now, statuses = world.__wrapped__(tmp, mp)
        prepare_run_day(mp)
        mp.setattr(strategy, "_mdp_cache", None)
        mp.setattr(scheduler, "notify_ready", lambda: None)
        mp.setattr(scheduler, "notify_watchdog", lambda: None)
        mp.setattr(scheduler, "time", SimpleNamespace(monotonic=lambda: 0, sleep=time.sleep, time=time.time))
        mp.chdir(tmp)
        config["scheduler"]["entry_intent"] = "enter"
        daily = daily_pick()
        daily.date = requested_day
        state = scheduler_state()
        state.date = requested_day
        first_pitch = midnight + timedelta(minutes=55)
        daily.pick.game_time = first_pitch.isoformat()
        games = scheduler.fetch_schedule(DAY)
        for index, game in enumerate(games):
            game["gameDate"] = (first_pitch + timedelta(minutes=30 * index)).isoformat()
        scheduler.fetch_schedule.side_effect = lambda slate: games if slate == requested_day else []
        frame, tier = orchestrator.run_cascade.return_value
        frame = frame.assign(game_time=[g["gameDate"] for g in games])
        orchestrator.run_cascade.return_value = (frame, tier)
        journal, stdout = StringIO(), StringIO()
        sent, trace = [], []
        dispatch = alert.dispatch_dm_for_health_alerts
        run_lines, run_start = inspect.getsourcelines(scheduler.run_day)
        fallback_sites = [i for i, line in enumerate(run_lines, run_start)
                          if "_alert_contest_state_failure(config, e" in line]
        status_path = directory.parent / "health_state/health_dm_delivery_status.json"
        first = midnight - timedelta(minutes=6) if case in {"final", "in_loop"} else midnight + timedelta(minutes=5)

        def tick(at):
            now[0] = at
            _set_clock(at)

        def observe(alerts, *args, **kwargs):
            parent = inspect.currentframe().f_back
            req = kwargs.get("now_et_date")
            if kwargs.get("status_path") == status_path:
                upstream = parent.f_back
                site = (("in_loop", "final")[fallback_sites.index(upstream.f_lineno)]
                        if upstream.f_lineno in fallback_sites and upstream.f_code.co_name == "run_day" else
                        inspect.getframeinfo(upstream).code_context[0].strip())
                trace.append([alerts[0].source, upstream.f_code.co_name,
                              site,
                              now[0].isoformat(), req.isoformat() if req else None])
            return dispatch(alerts, *args, **kwargs)

        def send(_, body):
            sent.append(body)
            assert journal.getvalue(), case
            return "health-captured"

        mp.setattr(alert, "dispatch_dm_for_health_alerts", observe)
        mp.setattr(alert, "send_dm", send)
        mp.setattr(scheduler, "_watchdog_ping_sleep", lambda seconds: tick(now[0] + timedelta(seconds=seconds)))
        tick(first)
        with redirect_stderr(journal), redirect_stdout(stdout):
            if case == "check":
                value = scheduler.run_single_check(requested_day, [101, 102], set(), config, 0.03)
            elif case in {"in_loop", "final"}:
                picks.save_pick(daily, directory)
                if case == "final":
                    orchestrator.run_cascade.side_effect = [(None, None), (None, None), (frame, tier)]
                else:
                    # The first check happens after midnight on the requested stale slate.
                    # Give that check a current observation, then remove it during refresh.
                    set_dated(directory, now=midnight + timedelta(minutes=10))
                    config["scheduler"]["lineup_check_offset_min"] = 60
                    if overnight:
                        tick(midnight + timedelta(minutes=1))
                    mp.setattr(scheduler, "_lock_decision_from_predictions",
                               lambda *a, **kw: scheduler.LockDecision(False, None, False))

                    def predict(*_):
                        if orchestrator.run_cascade.call_count > 1:
                            (directory / "account_state/contest_streak.manual.json").unlink(missing_ok=True)
                        return frame, tier

                    orchestrator.run_cascade.side_effect = predict
                value = scheduler.run_day(requested_day, config)
            elif case.startswith("missed"):
                config["scheduler"]["contest_state_required"] = case == "missed_live"
                picks.save_pick(daily, directory)
                tick(midnight - timedelta(minutes=8))
                value = scheduler._maybe_alert_missed_pick(config, requested_day, directory, 10, None, state)
            else:
                if case == "saved":
                    set_dated(directory, now=now[0], streak=12, best=16)
                    original = contest_state.load_decision_streak_state(directory, require_contest_state=True)
                    daily.chosen_on = picks.chosen_on_record(original, required=True)
                    set_dated(directory, now=now[0], streak=13, best=16)
                elif case == "void":
                    statuses[101] = {"abstract": "P", "detailed": "Cancelled", "code": "C"}
                elif case.startswith("late"):
                    cutoff = picks.submission_cutoff_et(daily)
                    if case == "late_initial":
                        tick(cutoff)
                    else:
                        config["scheduler"]["pick_delivery"] = case.removeprefix("late_")
                        set_dated(directory, now=now[0], streak=12, best=16)
                        original = contest_state.load_decision_streak_state(directory, require_contest_state=True)
                        daily.chosen_on = picks.chosen_on_record(original, required=True)
                        real_format = scheduler._format_pick_delivery_text

                        def format_with_delay(*args):
                            tick(cutoff)
                            return real_format(*args)

                        mp.setattr(scheduler, "_format_pick_delivery_text", format_with_delay)
                elif case == "delivered":
                    set_dated(directory, now=now[0], streak=12, best=16)
                    original = contest_state.load_decision_streak_state(directory, require_contest_state=True)
                    daily.chosen_on = picks.chosen_on_record(original, required=True)
                picks.save_pick(daily, directory)
                value = scheduler._deliver_and_lock_pick(daily, config, directory, state, requested_day, case)

        normalize = lambda s: s.replace(str(tmp), "<TMP>")
        files = {str(p.relative_to(tmp)): normalize(p.read_text()) for p in tmp.rglob("*")
                 if p.is_file() and p != status_path}
        receipt = json.loads(status_path.read_text()) if status_path.exists() else None
        current = "slate_date" in inspect.signature(scheduler._alert_contest_state_failure).parameters
        for source, caller, line, wall_time, chosen in trace:
            assert chosen == (requested_day if current else None)
            if overnight and (case != "in_loop" or source == "contest_state"):
                assert wall_time[:10] != requested_day, (case, trace)
        if case != "delivered":
            assert trace, (case, journal.getvalue())
            assert not dm.send_dm.called and not posting.post_to_bluesky.called
        else:
            assert value is True and dm.send_dm.call_count == 1 and state.pick_locked
        if receipt:
            sent_day = requested_day if current else trace[-1][3][:10]
            assert receipt["sent_day"] == sent_day, (case, receipt, trace)
        if case in {"in_loop", "final"}:
            assert any(source == "contest_state" and site == case for source, _, site, _, _ in trace), (
                case, trace, journal.getvalue())
        return {"result": value, "journal": normalize(journal.getvalue()), "stdout": normalize(stdout.getvalue()),
                "files": files, "receipt": receipt, "wire": [normalize(body) for body in sent],
                "picks": [call.args[1] for call in dm.send_dm.call_args_list], "trace": trace}


CASES = ("check", "loader", "saved", "in_loop", "final", "missed_live", "missed_preview",
         "void", "late_initial", "late_dm", "late_public", "delivered")


def compare(base, current):
    for key in base:
        old, new = base[key], current[key]
        for field in ("result", "journal", "stdout", "files", "picks"):
            assert old[field] == new[field], (key, field, old[field], new[field])
        assert len(old["wire"]) == len(new["wire"]), key
        for before, after in zip(old["wire"], new["wire"]):
            if "- [contest_state]" not in before:
                assert before == after, key
        if old["receipt"]:
            assert old["receipt"].keys() == new["receipt"].keys(), key
            difference = {field for field, val in old["receipt"].items() if new["receipt"][field] != val}
            assert difference <= {"sent_day", "body_sha256"}, (key, difference)
        print(key, "unchanged journal, return, pick/decision/archive/state files and other wire bodies")


def main():
    if sys.argv[1] == "compare":
        compare(*(json.loads(Path(p).read_text()) for p in sys.argv[2:]))
        return
    print("bts:", bts.__file__)
    results = {}
    for day, midnight, overnight in (
        (DAY, datetime.fromisoformat(DAY).replace(tzinfo=ET, hour=18), False),
        ("2027-06-14", datetime(2027, 6, 15, tzinfo=ET), True),
        ("2027-12-31", datetime(2028, 1, 1, tzinfo=ET), True),
    ):
        for case in CASES:
            key = f"{day}/{case}"
            results[key] = replay(case, day, midnight, overnight)
            print(key, results[key]["trace"])
    Path(sys.argv[1]).write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
