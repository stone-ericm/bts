import io
import json
import socket
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pytest import MonkeyPatch
from bts import dm, orchestrator, picks, posting, scheduler, strategy
from bts.daily_decision import decision_path, load_decision
from bts.health import alert
from tests.test_pick_chosen_on_cs5a import DAY, NOW, daily_pick, set_automatic, set_dated
from tests.test_saved_pick_paths_cs5a import invoke, world


class Interrupted(BaseException):
    pass


def forbidden(*args, **kwargs):
    raise AssertionError(f"Unexpected network call: {args}")


def pin(mp):
    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr",
                 "getnameinfo", "create_connection"):
        mp.setattr(socket, name, forbidden)
    for name in ("connect", "connect_ex", "send", "sendall", "sendto", "sendmsg"):
        if hasattr(socket.socket, name):
            mp.setattr(socket.socket, name, forbidden)
    for name in ("notify_ready", "notify_watchdog"):
        mp.setattr(scheduler, name, Mock())


def et(time):
    return datetime.fromisoformat(f"{DAY}T{time}:00-04:00")


def setup(root, mp, *, times=("19:10", "19:40"), doubleheader=False):
    pin(mp)
    config, directory, clock, statuses = world.__wrapped__(root, mp)
    config["scheduler"]["entry_intent"] = "enter"
    games = []
    for pk, when in enumerate(times, 101):
        games.append({
            "gamePk": pk, "gameDate": et(when).isoformat(),
            "teams": {side: {"team": {"name": f"{side} {101 if doubleheader else pk}"}}
                      for side in ("away", "home")},
        })
    scheduler.fetch_schedule.side_effect = lambda date: games if date == DAY else []
    mp.setattr(scheduler, "check_confirmed_lineups", Mock(return_value={}))
    mp.setattr(scheduler, "_idle_until_next_wakeup", Mock())
    mp.setattr(scheduler, "run_result_polling", Mock(return_value="final"))
    mp.setattr(strategy, "get_game_statuses", lambda *_: {pk: "P" for pk in statuses})
    frame, tier = orchestrator.run_cascade.return_value
    frame = frame.iloc[:len(games)].copy()
    frame["game_time"] = [g["gameDate"] for g in games]
    mp.setattr(orchestrator, "run_cascade", Mock(return_value=(frame, tier)))
    mp.setattr(posting, "reply_to_bluesky", Mock(return_value="offline-reply"))
    return config, directory, clock, statuses, frame, tier


def run(config, directory, clock, *, stop_on=None):
    sleeps = []
    selections = []
    stdout, stderr = io.StringIO(), io.StringIO()
    lock = []
    outcome = "returned"
    original = strategy.select_pick
    original_delivery = scheduler._deliver_and_lock_pick
    for capture in (dm.send_dm, posting.post_to_bluesky, alert.send_dm, orchestrator.run_cascade,
                    scheduler.run_result_polling, scheduler.fetch_schedule,
                    scheduler.check_confirmed_lineups, scheduler._idle_until_next_wakeup):
        capture.reset_mock()

    def sleep(seconds):
        sleeps.append(round(seconds / 60, 1))
        if stop_on == len(sleeps):
            raise Interrupted
        clock[0] += timedelta(seconds=seconds)

    def select(predictions, *args, **kwargs):
        result = original(predictions, *args, **kwargs)
        if result.decision is not None:
            kind = f"fresh:{result.action}"
        elif result.pick_result is not None:
            kind = f"reuse:{'locked' if result.pick_result.locked else 'unlocked'}"
        else:
            kind = result.no_pick_reason
        selections.append(kind)
        return result

    def deliver(daily, config, directory, state, date, label, *, selection=None):
        lock.append(label)
        return original_delivery(daily, config, directory, state, date, label, selection=selection)

    start = clock[0]
    with MonkeyPatch.context() as mp, redirect_stdout(stdout), redirect_stderr(stderr):
        mp.setattr(scheduler, "_watchdog_ping_sleep", sleep)
        mp.setattr(strategy, "select_pick", select)
        mp.setattr(scheduler, "_deliver_and_lock_pick", deliver)
        try:
            scheduler.run_day(DAY, config)
        except Interrupted:
            outcome = "interrupted"
    state = scheduler.load_state(DAY, directory)
    record = load_decision(DAY, directory)
    saved = picks.load_pick(DAY, directory)
    messages = [arg.args[1] for arg in alert.send_dm.call_args_list]
    out = dict(
        start=start.isoformat(), end=clock[0].isoformat(), outcome=outcome,
        checks=[x["time"][11:16] for x in state.runs_completed],
        cascades=orchestrator.run_cascade.call_count, selects=selections,
        locks=lock, notes=[line.strip() for line in stderr.getvalue().splitlines()
                           if any(x in line for x in ("REFUSED", "ERROR", "UNKNOWN",
                                                       "committed", "standing MDP",
                                                       "archiving", "confirmed.", "locked",
                                                       "no fresh pick"))],
        stamp=saved.chosen_on if saved else None, slot=saved.pick.game_pk if saved else None,
        pick_dms=[x.args[1].splitlines()[0] for x in dm.send_dm.call_args_list],
        posts=posting.post_to_bluesky.call_count, locked=state.pick_locked,
        locked_at=state.pick_locked_at, alerts=messages,
        decision=(record["action"], record["delivery_status"]) if record else None,
        skip=bool(state.final_skip_candidate), summary=bool(state.skip_summary),
        committed=state.committed_pick_written, sleeps=sleeps,
        poll=[(x.args[0], x.args[1]) for x in scheduler.run_result_polling.call_args_list],
        statuses=[p.name for p in directory.glob("*pick_*.json")],
        fetches=[x.args[0] for x in scheduler.fetch_schedule.call_args_list],
    )
    return out


def scenarios(name, root, mp):
    options = {
        "thin-future": dict(times=("19:10",)),
        "past-fallback": dict(times=("19:10",)),
        "warmup": dict(times=("19:10",)),
        "slow-fetch": dict(times=("19:10",)),
        "dh": dict(times=("19:10", "20:00"), doubleheader=True),
        "second-cascade": dict(times=("19:10",)),
    }
    config, directory, clock, statuses, frame, tier = setup(root, mp, **options.get(name, {}))
    good = frame.assign(p_game_hit=[0.99, 0.81][:len(frame)])
    close = frame.copy()
    if len(close) == 2:
        close.loc[1, ["flags", "projected_lineup"]] = ("PROJECTED", True)
    before = None
    if name.startswith("c-"):
        set_dated(directory, streak=7, best=9)
        clock[0] = et("18:31")
        orchestrator.run_cascade.return_value = close, tier
        before = run(config, directory, clock, stop_on=1)
        assert before["slot"] == 101 and not before["locked"] and before["stamp"]["streak"] == 7, before
        clock[0] = et("18:56")
        set_dated(directory, now=clock[0], streak=31, best=31)
        orchestrator.run_cascade.return_value = (
            (None, None) if name == "c-fail" else (good, tier)
        )
    elif name in {"d", "d-no-record", "attempted"}:
        set_dated(directory, streak=30, best=30)
        clock[0] = et("18:31")
        orchestrator.run_cascade.return_value = good, tier
        if name == "attempted":
            dm.send_dm.side_effect = Interrupted
            before = run(config, directory, clock)
            assert picks.load_pick(DAY, directory).delivery_attempted and not before["locked"]
            dm.send_dm.side_effect = None
        else:
            before = run(config, directory, clock)
            assert before["locked"]
            if name == "d-no-record":
                decision_path(DAY, directory).unlink()
        record_path = decision_path(DAY, directory)
        pick_path = directory / f"{DAY}.json"
        raw_decision = record_path.read_bytes() if record_path.exists() else None
        raw_pick = pick_path.read_bytes()
        clock[0] = et("20:33")
        set_dated(directory, now=clock[0], streak=31, best=31)
        out = run(config, directory, clock)
        out.update(pick_same=pick_path.read_bytes() == raw_pick,
                   decision_same=record_path.read_bytes() == raw_decision)
        if name == "d":
            clock[0] += timedelta(seconds=10)
            return [before, out, run(config, directory, clock)]
        return [before, out]
    elif name == "e":
        set_dated(directory, streak=7, best=9)
        clock[0] = et("18:31")
        orchestrator.run_cascade.return_value = close, tier
        before = run(config, directory, clock, stop_on=1)
        clock[0] = et("19:41")
        orchestrator.run_cascade.return_value = good, tier
    else:
        if name in {"a", "warmup"}:
            set_automatic(directory, streak=0, covered=False)
            result = invoke("preview", config, root)
            assert result.exit_code == 0, result.output
            assert picks.load_pick(DAY, directory).double_down is not None if name == "a" else True
        if name in {"a", "b", "past-fallback", "warmup", "dh", "thin-future"}:
            clock[0] = et("18:56") if len(frame) == 2 else et("18:36")
            if name == "thin-future":
                # The in-flight first invocation has its 16:10/17:10 early checks.
                clock[0] = et("16:09")
                before = run(config, directory, clock, stop_on=3)
                clock[0] = et("17:11")
            elif name == "warmup":
                clock[0] = et("18:57")
                before = run(config, directory, clock)
                clock[0] = et("18:58")
                statuses[101] = {"abstract": "L", "detailed": "Warmup", "code": "PW"}
            elif name == "dh":
                before = run(config, directory, clock, stop_on=1)
                clock[0] = et("19:16")
                # Game 1 is unenterable. Game 2's predictions only become available
                # when the scheduler's real DH confirmation loop wakes at 19:31.
                orchestrator.run_cascade.side_effect = (
                    lambda *_: (good.iloc[[1]], tier) if clock[0] >= et("19:31") else (None, None)
                )
                scheduler.check_confirmed_lineups.side_effect = (
                    lambda pks: {102: {"away"}} if 102 in pks and clock[0] >= et("19:31") else {}
                )
            else:
                before = run(config, directory, clock)
                clock[0] += timedelta(minutes=1)
            set_dated(directory, now=clock[0], streak=30, best=30)
        elif name == "second-cascade":
            clock[0] = et("18:45")
            set_dated(directory, now=clock[0], streak=30, best=30)
            projected = good.assign(flags="PROJECTED", projected_lineup=True)

            def cascade(*_):
                clock[0] += timedelta(minutes=11)
                return projected, tier

            orchestrator.run_cascade.side_effect = cascade
        elif name in {"past-saved", "live-old-time", "lookup-lock", "slow-fetch"}:
            set_dated(directory, streak=30, best=30)
            picks.save_pick(daily_pick(), directory)
            clock[0] = et("19:06") if name == "past-saved" else et("18:56")
            if name == "live-old-time":
                statuses[101] = {"abstract": "L", "detailed": "In Progress", "code": "I"}
            if name == "lookup-lock":
                details = Mock()

                def status_read(*_):
                    if details.call_count == 1:
                        raise OSError("transient")
                    return statuses

                details.side_effect = status_read
                mp.setattr(picks, "get_game_statuses_detailed", details)
                mp.setattr(picks, "get_game_statuses", Mock(side_effect=OSError("transient")))
            if name == "slow-fetch":
                clock[0] = et("19:04")
                original_fetch = scheduler.fetch_schedule.side_effect

                def slow_fetch(date):
                    if date == DAY:
                        clock[0] += timedelta(minutes=2)
                    return original_fetch(date)

                scheduler.fetch_schedule.side_effect = slow_fetch
        elif name.startswith("skip-"):
            clock[0] = et("18:56")
            set_dated(directory, streak=56, best=56)
            picks.save_pick(daily_pick(), directory)
            orchestrator.run_cascade.return_value = frame.assign(p_game_hit=0.10), tier
            before = run(config, directory, clock)
            state = scheduler.load_state(DAY, directory)
            assert state.final_skip_candidate is not None and state.skip_summary is not None
            if name == "skip-summary":
                state.final_skip_candidate = None
            elif name == "skip-committed":
                state.committed_pick_written = True
            scheduler.save_state(state, directory)
            clock[0] = et("18:57")
            set_dated(directory, now=clock[0], streak=30, best=30)
            if name != "skip-fresh":
                orchestrator.run_cascade.side_effect = lambda *_: (None, None)
        else:
            raise AssertionError(name)
        orchestrator.run_cascade.return_value = good, tier

    if name == "thin-future":
        waiting = run(config, directory, clock, stop_on=1)
        return [before, waiting, run(config, directory, clock)]
    after = run(config, directory, clock)
    if name == "slow-fetch":
        clock[0] = et("19:04")
        scheduler.fetch_schedule.side_effect = original_fetch
        return [after, run(config, directory, clock)]
    if name == "dh":
        assert after["cascades"] == 2 and not after["locked"] and after["end"][11:16] == "19:31"
        return [before, after, run(config, directory, clock)]
    if name in {"live-old-time", "lookup-lock"}:
        # Same input frame and time. No stale candidate means no pre-cascade
        # classification lock; the fresh-pick path can select game 102.
        (directory / f"{DAY}.json").unlink()
        return [after, run(config, directory, clock)]
    return [before, after] if before else [after]


if __name__ == "__main__":
    cases = sys.argv[1:] or [
        "a", "b", "c-ok", "c-fail", "d", "d-no-record", "e", "attempted",
        "thin-future", "past-fallback", "past-saved", "warmup", "slow-fetch", "second-cascade",
        "dh", "live-old-time", "lookup-lock",
        "skip-fresh", "skip-fail", "skip-summary", "skip-committed",
    ]
    for case in cases:
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp, MonkeyPatch.context() as mp:
            print(case, flush=True)
            try:
                print(json.dumps(scenarios(case, Path(tmp), mp)), flush=True)
            except Exception:
                import traceback
                traceback.print_exc()
