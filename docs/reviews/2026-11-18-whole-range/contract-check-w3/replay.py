"""Run targeted, offline W3 contract examples on main."""

import argparse
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import sys
from unittest.mock import patch

from bts import contest_state, daily_decision, dm, picks, scheduler


SCRATCH = Path(__file__).resolve().parent
ROOT = SCRATCH.parents[1]
REVIEWS = ROOT / "docs/reviews/2026-11-18-whole-range"
sys.path[:0] = [str(REVIEWS / "slice-3-codex"), str(REVIEWS / "contract-check-a")]
import adversarial
import interactions
import offline


offline.HERE = SCRATCH
offline.ROOT = ROOT
DAY = "2027-03-24"


@contextmanager
def bounded_read():
    def expired(*_):
        raise TimeoutError("pick reload blocked for one second")

    old = signal.signal(signal.SIGALRM, expired)
    signal.alarm(1)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def original():
    interactions.overrun_double_cascade()
    adversarial.overrun_replacement()


def run(case, reuse):
    with offline.offline(offline.instant(DAY, 18, 25)) as (w, mp):
        two_games = {"confirmations", "reload_cutoff", "attempted"}
        if case not in two_games:
            w.schedule[DAY] = w.schedule[DAY][:1]
        w.dated(0)
        path = w.picks / f"{DAY}.json"
        starts = []
        calls = []
        written = []
        check = scheduler.run_single_check
        lock = scheduler._lock_decision_from_predictions
        atomic_write = picks.atomic_write_text
        final_reader = contest_state.load_decision_streak_state
        read_pick = picks.load_pick
        returned = False

        def forecast():
            starts.append(w.now.strftime("%H:%M"))
            if case == "warmup" and starts == ["18:25"]:
                return w.frame((0.99,), (True,))
            if starts == ["18:25"]:
                w.advance((40 if case == "attempted" else 35) * 60)
                return w.frame((0.99, 0.98), (True, True))
            if case == "lineup":
                frame = w.frame((0.99,), (False,))
                frame.loc[0, ["batter_name", "batter_id"]] = ["Confirmed Batter", 999]
                return frame
            if case in {"confirmations", "attempted"} and len(starts) == 2:
                return w.frame((0.99, 0.98), (True, False))
            return None

        def capture_write(pick_path, text):
            atomic_write(pick_path, text)
            if pick_path == path:
                written.append(text.encode("utf-8"))

        def capture_lock(*args, **kwargs):
            result = lock(*args, **kwargs)
            calls.append(result)
            return result

        def wrapped_check(*args, **kwargs):
            nonlocal returned
            calls.clear()
            with patch.object(scheduler, "_lock_decision_from_predictions", capture_lock):
                result = check(*args, **kwargs)
            if not calls or returned:
                return result
            returned = True
            assert not result["should_post"]
            saved = written[-1]
            if case == "warmup":
                assert result["selection"].action is None
                assert result["selection"].primary_candidate is None
            if case == "replacement":
                daily = deepcopy(result["pick_result"].daily)
                daily.pick = replace(daily.pick, batter_name="Replacement Batter", batter_id=999)
                picks.save_pick(daily, w.picks)
            elif case == "attempted":
                daily = deepcopy(result["pick_result"].daily)
                daily.delivery_attempted = True
                picks.save_pick(daily, w.picks)
            elif case == "committed":
                record = daily_decision.write_decision(
                    DAY, w.picks, action="single", source="mdp", scoreable=True,
                    delivery_status="private_locked",
                    primary={"batter_name": "Previously Committed", "batter_id": 999},
                )
                assert record is not None
            elif case == "dated_correction":
                w.dated(0, best=10)
            elif case == "void":
                w.schedule[DAY][0]["status"] = {
                    "abstractGameCode": "P", "detailedState": "Postponed", "statusCode": "PO",
                }
            elif case == "absent":
                path.unlink()
            elif case == "malformed":
                path.write_bytes(b"{")
            elif case == "shape":
                path.write_bytes(b"[]")
            elif case == "directory":
                path.unlink()
                path.mkdir()
            elif case == "fifo":
                path.unlink()
                os.mkfifo(path)
            if reuse and case not in {"absent", "malformed", "shape", "directory", "fifo", "unreadable"}:
                # Simulate the proposed producer and comparison around main's
                # existing consumer. The injected key is its only reuse switch.
                if path.read_bytes() == saved:
                    result["lock_decision"], = calls
                    result["saved_pick_bytes"] = saved
            return result

        def read(date, directory):
            if returned and case == "reload_cutoff" and w.now.hour == 19 and w.now.minute == 0:
                w.advance(5 * 60)
            if returned and case == "unreadable":
                raise PermissionError("simulated EACCES")
            return read_pick(date, directory)

        def contest_read(*args, **kwargs):
            result = final_reader(*args, **kwargs)
            if returned and case == "final_cutoff" and w.now.minute == 0:
                w.advance(5 * 60)
            return result

        def add_attempt_write_time(pick_path, text):
            capture_write(pick_path, text)
            data = json.loads(text)
            if returned and data.get("delivery_attempted") and not data.get("notification_sent"):
                w.advance(5 * 60)

        def confirmations(game_pks):
            return {pk: {"home", "away"} for pk in game_pks
                    if pk == 101 or case == "confirmations" and returned}

        if case == "warmup":
            assert check(DAY, [101], set(), w.config, 0.03)["pick_result"] is not None
            starts.clear()
            w.schedule[DAY][0]["status"] = {
                "abstractGameCode": "L", "detailedState": "Warmup", "statusCode": "PW",
            }

            def warmup():
                starts.append(w.now.strftime("%H:%M"))
                if len(starts) == 1:
                    w.advance(35 * 60)
                    return w.frame((0.99,), (True,))
                return None

            w.predictions = warmup
        else:
            w.predictions = forecast
        if case in two_games:
            mp.setattr(scheduler, "check_confirmed_lineups", confirmations)
        if case == "failed":
            mp.setattr(dm, "send_dm", lambda *_: (_ for _ in ()).throw(RuntimeError("fixture failure")))
        if case == "interrupted":
            mp.setattr(dm, "send_dm", lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        mp.setattr(picks, "atomic_write_text", add_attempt_write_time if case == "marker_cutoff" else capture_write)
        mp.setattr(picks, "load_pick", read)
        mp.setattr(contest_state, "load_decision_streak_state", contest_read)
        mp.setattr(scheduler, "run_single_check", wrapped_check)
        error = None
        try:
            with bounded_read() if case == "fifo" else nullcontext():
                scheduler.run_day(DAY, w.config)
        except (KeyboardInterrupt, OSError, TimeoutError, ValueError, TypeError) as exc:
            error = type(exc).__name__
            if case not in {"interrupted", "unreadable", "malformed", "shape", "directory", "fifo"}:
                raise

        state = scheduler.load_state(DAY, w.picks)
        commit = daily_decision.load_decision(DAY, w.picks) or {}
        readable = case not in {"unreadable", "malformed", "shape", "directory", "fifo"}
        current = read_pick(DAY, w.picks) if readable else None
        facts = {
            "case": case, "injected_reuse": reuse, "starts": starts, "error": error,
            "refreshes": [(row["reused_check_cascade"], row["reason"]) for row in state.fallback_refreshes or []],
            "dm": [(when.strftime("%H:%M"), body.splitlines()[0]) for kind, when, body in w.events if kind == "dm"],
            "alert": [body.splitlines()[-1].split("]")[0].split("[")[-1] for body in w.sends("health-dm")],
            "committed": (commit.get("primary") or {}).get("batter_name"),
            "selection": commit.get("delivery_status"),
            "attempt": current.delivery_attempted if current else None,
            "file_exists": path.exists(),
        }
        print("RESULT", json.dumps(facts), flush=True)
        return facts


def demonstrate(args):
    if args.case == "original":
        return original()
    if args.case == "files":
        for case, error in {
            "absent": None, "unreadable": "PermissionError", "malformed": "JSONDecodeError",
            "shape": "TypeError", "directory": "IsADirectoryError", "fifo": "TimeoutError",
        }.items():
            facts = run(case, False)
            assert facts["error"] == error and not facts["dm"]
            assert facts["file_exists"] == (case != "absent")
        return
    run(args.case, args.reuse)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("--reuse", action="store_true")
    demonstrate(parser.parse_args())
