"""Compare hostile on-disk inputs, receipt errors, and clock changes at flock."""

from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
from threading import Event, Thread
from unittest.mock import patch

from probe_common import ET, RUN, examine, receipt, run_against_archives, validate


TODAY = "2027-04-20"
PAST = "2027-04-16"
READ_ERROR = "pick file unreadable or not valid JSON"
MARKED = "today's result is applied to streak.json but not yet saved on the pick"
PENDING = "no final result on the pick"


def form(path, shape):
    kind, value = shape
    if kind == "missing":
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "json":
        path.write_text(json.dumps(value))
    elif kind == "raw":
        path.write_bytes(value)
    elif kind == "directory":
        path.mkdir()
    elif kind == "dangling":
        path.symlink_to("missing-target")
    elif kind == "symlink":
        target = path.with_name("indirect.data")
        target.write_text(json.dumps(value))
        path.symlink_to(target.name)
    elif kind == "unreadable":
        path.write_text(json.dumps(value))
        path.chmod(0)
    else:
        raise AssertionError(kind)


def root_for(scratch, name, streak=None):
    root = scratch / name / "picks"
    root.mkdir(parents=True)
    if streak is not None:
        form(root / "streak.json", streak)
    return root


def matrix(scratch, results):
    good = {"streak": 9, "season": 2027, "saver_available": False,
            "updated": "2027-01-01T06:00:00+00:00"}
    scalar = [("null", None), ("array", []), ("number", 3), ("float", 0.5),
              ("string", "hello"), ("boolean", True)]
    special = {"missing": ("missing", None), "malformed": ("raw", b"{"),
               "nonutf8": ("raw", b"\xff"), "blank": ("raw", b""),
               "directory": ("directory", None), "dangling": ("dangling", None)}
    pick_shapes = special | {label: ("json", value) for label, value in scalar} | {
        "empty": ("json", {}), "pending": ("json", {"result": "suspended"}),
        "hit_missing_slots": ("json", {"result": "hit"}), "miss": ("json", {"result": "miss"}),
        "void": ("json", {"result": "void"}), "result_array": ("json", {"result": []}),
        "result_dict": ("json", {"result": {}}), "result_number": ("json", {"result": 5}),
        "result_boolean": ("json", {"result": True}), "result_null": ("json", {"result": None}),
        "symlink_pending": ("symlink", {}), "no_permission": ("unreadable", {}),
    }
    for day in (PAST, TODAY, "2027-04-19"):
        for label, spec in pick_shapes.items():
            name = f"pick-{day}-{label}"
            root = root_for(scratch, name, ("json", good | {"applied_dates": [TODAY]}))
            form(root / f"{day}.json", spec)
            results[name] = examine(root, lookback=1 if day == "2027-04-19" else 0)

    marker_values = scalar + [("object", {TODAY: True}), ("plain_date", TODAY),
                              ("mixed", [None, 3, {}, TODAY, TODAY]),
                              ("past", [PAST]), ("numbered", [100]), ("trailing_space", [TODAY + " "])]
    streak_shapes = special | {label: ("json", value) for label, value in scalar} | {
        f"marker_{label}": ("json", good | {"applied_dates": value}) for label, value in marker_values
    } | {
        "object": ("json", good), "minimal": ("json", {}),
        "foreign": ("json", good | {"season": 2026, "applied_dates": [TODAY]}),
        "float_tag": ("json", good | {"season": 2027.0, "applied_dates": [TODAY]}),
        "text_tag": ("json", good | {"season": "2027", "applied_dates": [TODAY]}),
        "tagless": ("json", {"applied_dates": [TODAY], "streak": 11}),
        "bool_tag": ("json", good | {"season": True, "applied_dates": [TODAY]}),
        "weird_streak": ("json", good | {"streak": [], "applied_dates": [TODAY]}),
        "symlink_to_object": ("symlink", good | {"applied_dates": [TODAY]}),
    }
    for label, spec in streak_shapes.items():
        for scenario, day, pick in (("saved", PAST, {"result": "void"}),
                                    ("past-blocker", PAST, {}), ("today", TODAY, {})):
            name = f"streak-{scenario}-{label}"
            root = root_for(scratch, name, spec)
            form(root / f"{day}.json", ("json", pick))
            results[name] = examine(root)

    decision_shapes = special | {label: ("json", value) for label, value in scalar} | {
        "empty": ("json", {}), "skip": ("json", {"action": "skip", "scoreable": False}),
        "delivered": ("json", {"action": "single", "scoreable": True, "delivery_status": "delivered"}),
        "odd": ("json", {"schema_version": [], "date": False, "scoreable": "true"}),
        "symlink": ("symlink", {"action": "skip", "scoreable": False}),
    }
    for label, spec in decision_shapes.items():
        for scenario, value in (("orphan", None), ("paired", {}), ("terminal", {"result": "void"})):
            name = f"decision-{scenario}-{label}"
            root = root_for(scratch, name, ("json", good))
            if value is not None:
                form(root / f"{PAST}.json", ("json", value))
            form(root / PAST / "decision.json", spec)
            results[name] = examine(root)


@contextmanager
def cross_while_waiting(root, start, after):
    acquired = Event()
    contender = Event()
    state = {"now": start}
    real_flock = fcntl.flock
    lock = root / ".scoring.lock"

    def hold():
        with lock.open("w") as peer:
            real_flock(peer, fcntl.LOCK_EX)
            acquired.set()
            try:
                if contender.wait(3):
                    state["now"] = after
            finally:
                real_flock(peer, fcntl.LOCK_UN)

    def observed_flock(fd, operation):
        if operation == fcntl.LOCK_EX:
            try:
                real_flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                contender.set()
            else:
                raise AssertionError("peer did not actually hold the scoring lock")
        return real_flock(fd, operation)

    peer = Thread(target=hold, daemon=True)
    peer.start()
    try:
        assert acquired.wait(3)
        with patch.object(fcntl, "flock", observed_flock):
            yield lambda: state["now"]
        assert contender.is_set() and state["now"] == after
    finally:
        contender.set()
        peer.join(timeout=3)
        assert not peer.is_alive()


@contextmanager
def alarm(seconds):
    class ReadTimedOut(KeyboardInterrupt):
        pass

    def interrupt(*args):
        raise ReadTimedOut("bounded FIFO read")

    previous = signal.signal(signal.SIGALRM, interrupt)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def probes(scratch, revision):
    from bts import receipt_io

    results = {}
    matrix(scratch, results)
    for label, another in (("first-of-mixed", "{"), ("first-of-pending", "{}"),
                           ("first-before-exception", "[]")):
        root = root_for(scratch, label, ("json", {"streak": 9, "season": 2027}))
        (root / "2027-04-17.json").write_text(another)
        (root / f"{PAST}.json").write_text("{}")
        (root / "2027-04-21.json").write_text("{")
        results[label] = examine(root)

    for label, before, after, day in (
        ("midnight", "2027-04-19T23:59", "2027-04-20T00:01", "2027-04-19"),
        ("new-year", "2027-12-31T23:59", "2028-01-01T00:01", "2028-01-01"),
    ):
        start, end = [datetime.fromisoformat(x).replace(tzinfo=ET) for x in (before, after)]
        root = root_for(scratch, label, ("json", {"streak": 9, "season": end.year, "applied_dates": [day]}))
        (root / f"{day}.json").write_text("{}")
        if start.year != end.year:
            (root / f"{start.date()}.json").write_text("{")
        with cross_while_waiting(root, start, end) as clock:
            results[label] = examine(root, clock=clock, fold_day=end.date())

    for label, kind in (("publish-fails-blocked", "{}"), ("publish-fails-saved", '{"result": "void"}')):
        root = root_for(scratch, label, ("json", {"streak": 9, "season": 2027}))
        (root / f"{PAST}.json").write_text(kind)
        with patch.object(receipt_io, "publish", side_effect=OSError("synthetic disk write failed")):
            results[label] = examine(root)

    root = root_for(scratch, "interrupt-replay", ("json", {"streak": 9, "season": 2027}))
    path = root / f"{PAST}.json"
    path.write_text("{}")
    read = Path.read_text

    def interrupt(path_read, *args, **kwargs):
        if path_read == path:
            raise KeyboardInterrupt
        return read(path_read, *args, **kwargs)

    with patch.object(Path, "read_text", interrupt):
        results["interrupt-replay"] = examine(root)

    root = root_for(scratch, "fifo", ("json", {"streak": 9, "season": 2027}))
    os.mkfifo(root / f"{PAST}.json")
    with alarm(1):
        results["fifo"] = examine(root, direct=False)

    root = root_for(scratch, "lost-hit", ("json", {"streak": 9, "season": 2027}))
    path = root / f"{PAST}.json"
    path.write_text('{"result":"hit"}')

    def remove_after_read(path_read, *args, **kwargs):
        content = read(path_read, *args, **kwargs)
        if path_read == path:
            path_read.unlink()
        return content

    with patch.object(Path, "read_text", remove_after_read):
        results["lost-hit"] = examine(root, direct=False)

    root = root_for(scratch, "newline-name", ("json", {"streak": 9, "season": 2027}))
    (root / f"{PAST}\n.json").write_text("{}")
    results["newline-name"] = examine(root)
    return results


def check(base, current):
    old_newline = base.pop("newline-name")
    new_newline = current.pop("newline-name")
    counts = validate(base, current)
    for name, day, reason in (("midnight", "2027-04-19", PENDING), ("new-year", "2028-01-01", MARKED),
                              ("lost-hit", PAST, "hit pick could not be loaded"),
                              ("first-of-mixed", PAST, PENDING), ("first-of-pending", PAST, PENDING),
                              ("first-before-exception", PAST, PENDING)):
        assert receipt(current[name])["replay"] == {"state": "unavailable", "blocked_date": day, "reason": reason}, name
        assert current[name]["strict"][1:3] == [0, None], name
    for name in ("publish-fails-blocked", "publish-fails-saved"):
        result = current[name]
        assert result["strict"][1] == 0 and result["receipt"] is None and "unavailable (OSError)" in result["stderr"], name
    for name in ("fifo", "interrupt-replay", "pick-2027-04-19-malformed"):
        assert current[name]["strict"][1] == 1, name
        assert receipt(current[name])["replay"] == {"state": "not_reached"}, name
    assert receipt(current["fifo"])["error"] == "ReadTimedOut"
    assert old_newline["strict"] == new_newline["strict"]
    assert receipt(new_newline)["replay"]["blocked_date"] == PAST + "\n"
    assert len(new_newline["stderr"].splitlines()) == 2, new_newline
    return {"matching_cases": sum(counts.values()) + 1, "states_before_newline": counts,
            "newline_stdout": new_newline["stdout"], "newline_stderr": new_newline["stderr"],
            "newline_receipt": receipt(new_newline)["replay"]}


if __name__ == "__main__":
    run_against_archives(__file__, probes, check)
