"""Probe the real contest reader and compare preserved alerts to the base."""

import datetime as dt
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import bts

from review_support import DAY, DECEMBER, deadline, eod, flag, health, observe, select, write

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEEP = b'{"ignored":' + b"[" * 200_000 + b"]" * 200_000 + b"}"
HUGE = b'{"season":' + b"1" * 6_000 + b"}"
SPECIAL = {"utf8": b"\xff", "list": b'["active"]', "broken": b"{",
           "huge": HUGE, "deep": DEEP, "object-state": b'{"season":2027,"state":[]}'}
FLAGS = ("absent", "active", "not_earned", "used", "uninitialized", "other-season", "boolean",
         "unreadable", "directory", "fifo", "long-link", "private-link", *SPECIAL)


def bad_fixture(picks, name):
    path = picks / "account_state/saver_state.json"
    if name in SPECIAL:
        path.write_bytes(SPECIAL[name])
    elif name == "directory":
        path.mkdir()
    elif name == "fifo":
        os.mkfifo(path)
    elif name == "long-link":
        path.symlink_to("x" * 300)
    elif name == "private-link":
        target = write(picks.parent / "private/flag.json", {"state": "active", "season": DAY.year})
        path.symlink_to(target)
        target.parent.chmod(0)
        return target.parent
    elif name == "unreadable":
        flag(picks, "active").chmod(0)
    elif name != "absent":
        flag(picks, "active" if name in {"other-season", "boolean"} else name,
             {"other-season": DAY.year - 1, "boolean": True}.get(name, DAY.year))
    return None


def serialized(rows):
    return [asdict(row) for row in rows if row.incident_key != "saver_state:dated_used_mismatch"]


def grid(root, candidate):
    snapshots = {}
    for label, invalid in (("huge", ValueError), ("deep", RecursionError)):
        try:
            json.loads(SPECIAL[label])
        except invalid:
            print(f"{label} payload reached {invalid.__name__}")
        else:
            raise AssertionError(f"{label} payload did not reach decoder limit")
    for state in ("used", "active", "not_earned"):
        for streak in (0, 9, *range(10, 16), 16, 57):
            for name in FLAGS:
                picks = root / state / str(streak) / name / "picks"
                observe(picks, saver=state, streak=streak)
                for offset in (1, 2):
                    write(picks / f"{DAY.date() + dt.timedelta(days=offset)}.json", {"result": "hit"})
                private = bad_fixture(picks, name)
                try:
                    for historical in (False, True):
                        with deadline():
                            rows = health.check(picks, now=DAY, expected=True,
                                                historical_day=DAY.date() if historical else None)
                        assert rows[0].source == "contest_state" and rows[0].level == "WARN"
                        mismatch = select(rows)
                        assert bool(mismatch) == (candidate and state == "used"
                                                  and name in {"active", "not_earned"})
                        if mismatch:
                            item, = mismatch
                            assert (item.level, item.source) == ("WARN", "saver_state") and item is rows[-1]
                        snapshots[f"{state}/{streak}/{name}/{historical}"] = serialized(rows)
                        if state == "used" and streak == 14 and name in {"long-link", "private-link"}:
                            raw, sends = eod(picks, wall=DAY + dt.timedelta(days=historical))
                            assert raw == rows and not sends, (raw, sends)
                finally:
                    if private is not None:
                        private.chmod(0o700)
    for streak in range(401):
        picks = root / "sweep" / str(streak) / "picks"
        observe(picks, saver="not_earned", streak=streak)
        bad_fixture(picks, "fifo")
        for historical in (None, DAY.date()):
            with deadline():
                rows = health.check(picks, now=DAY, historical_day=historical)
            assert sum(row.source == "saver_state" for row in rows) == int(10 <= streak <= 15)
            snapshots[f"sweep/{streak}/{historical}"] = serialized(rows)
    if candidate:
        jan = (DECEMBER + dt.timedelta(days=1)).replace(hour=0, minute=4)
        for day in (DECEMBER, jan):
            picks = root / str(day.date()) / "picks"
            observe(picks, day)
            flag(picks, "active", day.year)
            raw, sends = eod(picks, day)
            assert len(select(raw)) == len(sends) == 1 and f"season {day.year}" in sends[0][1]
        picks = root / "historical" / "picks"
        observe(picks, DECEMBER)
        flag(picks, "active", DECEMBER.year)
        raw, sends = eod(picks, DECEMBER, wall=jan)
        assert len(select(raw)) == len(sends) == 1 and "season 2027" in sends[0][1]
        flag(picks, "active", jan.year)
        assert not select(eod(picks, DECEMBER, wall=jan)[0])
        flag(picks, "active", DECEMBER.year)
        observe(picks, jan)
        assert not select(eod(picks, DECEMBER, wall=jan)[0])
        print("ET year boundary, prior-slate clock, and both current-file replacements verified")
    return snapshots


if __name__ == "__main__":
    if len(sys.argv) == 4:
        print("bts.__file__ =", bts.__file__)
        Path(sys.argv[3]).write_text(json.dumps(grid(Path(sys.argv[2]), sys.argv[1] == "candidate")))
    else:
        with TemporaryDirectory(prefix="f2-compare-", dir="/tmp") as directory:
            work = Path(directory)
            for label, source in (("base", HERE / "base/src"), ("candidate", ROOT / "src")):
                subprocess.run(
                    [sys.executable, __file__, label, str(work / "fixture"), str(work / f"{label}.json")],
                    env={**os.environ, "PYTHONPATH": f"{source}:{ROOT}"}, timeout=16, check=True,
                )
                (work / "fixture").rename(work / label)
            base, candidate = (json.loads((work / f"{side}.json").read_text()) for side in ("base", "candidate"))
            assert base.keys() == candidate.keys()
            assert all(base[case] == candidate[case] for case in base)
            print(f"{len(base)} paired comparisons preserved every earlier alert and existing dated WARN")
