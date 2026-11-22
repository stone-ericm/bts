"""Exercise pick readers at the initial reload and after the added polls."""

import json
import os
from pathlib import Path
import signal
import sys

from bts import daily_decision, picks, scheduler

from review_world import DAY, at, health, offline, pending_window


IS_BASE = "base/src" in scheduler.__file__


class OpenDeadline(BaseException):
    pass


def run(stage, case):
    with offline(at(18, 25)) as (world, mp):
        world.dated(0)
        pending_window(world, mp, scheduler)
        path = world.picks / f"{DAY}.json"
        target = path.with_name("regular-target.json")
        calls = []
        real_check = scheduler.run_single_check
        real_open = os.open
        real_path_open = Path.open
        damaged = False
        restored = False
        evidence = None

        def describe():
            if not os.path.lexists(path):
                return "absent"
            if path.is_symlink():
                return f"link:{path.readlink().name}"
            if path.is_fifo():
                return "fifo"
            if path.is_dir():
                return "directory"
            return os.stat(path).st_size

        def damage():
            nonlocal damaged, evidence
            if case in {"absence", "dangling", "regular-link", "directory", "fifo"}:
                if case == "regular-link":
                    path.rename(target)
                else:
                    path.unlink()
                if case == "dangling":
                    path.symlink_to(path.with_name("not-there"))
                elif case == "regular-link":
                    path.symlink_to(target)
                elif case == "directory":
                    path.mkdir()
                elif case == "fifo":
                    os.mkfifo(path)
            elif case in {"bad-json", "bad-shape"}:
                path.write_bytes(b"{" if case == "bad-json" else b"[]")
            elif case == "bad-time":
                payload = json.loads(path.read_text())
                payload["pick"]["game_time"] = "bad time"
                path.write_text(json.dumps(payload))
            evidence = describe()
            damaged = True

        def restore():
            nonlocal restored
            assert describe() == evidence
            if case != "unreadable" and os.path.lexists(path):
                if path.is_dir():
                    path.rmdir()
                else:
                    path.unlink()
            restored = True

        def forecast():
            calls.append(world.now)
            if len(calls) == 1:
                world.advance(22 * 60)
                return world.frame((0.99, 0.60), (True, True))
            if world.now < at(18, 50):
                return None
            return world.frame((0.60, 0.99), (True, False)).iloc[[1, 0]]

        def check(*args, **kwargs):
            if damaged and not restored:
                restore()
            result = real_check(*args, **kwargs)
            if stage == "initial" and not damaged:
                damage()
            return result

        def confirmation(game_pks):
            if stage == "extra" and not damaged and calls and game_pks == [102]:
                damage()
            return {pk: {"home", "away"} for pk in game_pks
                    if pk == 101 or case == "regular-link" and damaged}

        def open_descriptor(name, flags, *args, **kwargs):
            if case == "unreadable" and damaged and not restored and Path(name) == path:
                raise PermissionError("blocked pick descriptor in fixture")
            return real_open(name, flags, *args, **kwargs)

        def open_path(file, *args, **kwargs):
            if case == "unreadable" and damaged and not restored and file == path:
                raise PermissionError("blocked legacy pick read in fixture")
            return real_path_open(file, *args, **kwargs)

        def alarm(*_args):
            raise OpenDeadline("FIFO open exceeded one second")

        world.predictions = forecast
        mp.setattr(scheduler, "run_single_check", check)
        mp.setattr(scheduler, "check_confirmed_lineups", confirmation)
        mp.setattr(os, "open", open_descriptor)
        mp.setattr(Path, "open", open_path)
        exception = None
        previous = signal.signal(signal.SIGALRM, alarm)
        if case == "fifo":
            signal.alarm(3)
        try:
            scheduler.run_day(DAY, world.config)
        except (OpenDeadline, OSError, TypeError, ValueError) as error:
            exception = type(error).__name__
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
        expected_errors = {"fifo": "OpenDeadline", "directory": "IsADirectoryError",
                           "bad-shape": "TypeError", "bad-json": "JSONDecodeError",
                           "unreadable": "PermissionError", "bad-time": "ValueError"}
        expected_error = expected_errors.get(case) if stage == "initial" else None
        status = scheduler.load_state(DAY, world.picks)
        record = daily_decision.load_decision(DAY, world.picks)
        facts = {"stage": stage, "case": case, "error": exception, "evidence": evidence,
                 "unchanged_if_error": describe() == evidence if exception else None,
                 "reached_later_check": restored,
                 "sent": len(world.sends("dm")), "committed_game": record and record["primary"]["game_pk"],
                 "fallback": status.fallback_refreshes, "health": health(world)}
        print("PICK_FILE", json.dumps(facts), flush=True)
        assert exception == expected_error
        if exception:
            assert describe() == evidence and not facts["sent"]
        elif case == "regular-link":
            assert facts["sent"] == 1 and facts["committed_game"] == 101 and not restored
            assert status.fallback_refreshes[0]["reused_check_cascade"] is not IS_BASE
            assert target.exists()
        else:
            times = [clock.strftime("%H:%M") for clock in calls]
            assert times == (["18:25", "18:50", "18:50"] if IS_BASE else ["18:25", "18:50"])
            assert facts["sent"] == 1 and facts["committed_game"] == 102
            assert not list((world.picks / DAY).glob("deferred_fallback*.json"))


if __name__ == "__main__":
    stage = sys.argv[1]
    cases = ["absence", "dangling", "directory", "fifo", "bad-json", "bad-shape", "unreadable"]
    cases += ["regular-link"]
    if stage == "initial":
        cases += ["bad-time"]
    for name in sys.argv[2:] or cases:
        run(stage, name)
