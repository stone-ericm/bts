import datetime as dt
import os
import sys
from pathlib import Path

from probe_support import DAY, DECEMBER, ET, INCIDENT, bts, eod, flag, health, observed, put, scratch

FLAGS = ("active", "not_earned", "used", "uninitialized", "absent", "other-season", "boolean-season",
         "unreadable-mode", "invalid-utf8", "json-list", "json-syntax", "directory", "fifo",
         "huge-int", "deep", "linked-unreadable", "linked-too-long")
UNREADABLE_LINKS = {"linked-unreadable", "linked-too-long"}


def arrange_flag(picks, name):
    path = picks / "account_state/saver_state.json"
    if name == "absent":
        pass
    elif name in {"active", "not_earned", "used", "uninitialized"}:
        flag(picks, name)
    elif name in {"other-season", "boolean-season"}:
        flag(picks, "active", DAY.year - 1 if name == "other-season" else True)
    elif name == "unreadable-mode":
        flag(picks, "active").chmod(0)
    elif name == "directory":
        path.mkdir()
    elif name == "fifo":
        os.mkfifo(path)
    elif name == "linked-unreadable":
        private = picks.parent / "restored-private"
        private.mkdir()
        put(private, "saver.json", {"season": DAY.year, "state": "active"})
        private.chmod(0)
        path.symlink_to(private / "saver.json")
        return private
    elif name == "linked-too-long":
        path.symlink_to("s" * 300)
    else:
        path.write_bytes({
            "invalid-utf8": b"\xff\x80", "json-list": b'[{"state":"active"}]',
            "json-syntax": b"{]", "huge-int": b'{"season":' + b"9" * 5100 + b"}",
            "deep": b'{"x":' + b"[" * 2600 + b"]" * 2600 + b"}",
        }[name])
    return None


def mismatch(rows):
    return [row for row in rows if row.incident_key == INCIDENT]


def run_grid(root, base):
    rows = 0
    for state in ("used", "active", "not_earned"):
        for streak in (0, 9, 10, 12, 15, 16, 57):
            for name in FLAGS:
                picks = root / f"{state}-{streak}-{name}" / "picks"
                observed(picks, state=state, streak=streak)
                # Supply a settled file after coverage to exercise preservation of an already-collected alert.
                put(picks, f"{DAY.date() + dt.timedelta(days=1)}.json", {"result": "hit"})
                modes = (None, DAY.date())
                reference = [health.check(picks, now=DAY, expected=True, historical_day=mode) for mode in modes]
                assert all(check[0].source == "contest_state" and check[0].level == "INFO" for check in reference)
                private = arrange_flag(picks, name)
                try:
                    for mode, old in zip(modes, reference):
                        try:
                            raw = health.check(picks, now=DAY, expected=True, historical_day=mode)
                        except OSError as error:
                            assert not base and state == "used" and name in UNREADABLE_LINKS
                            if streak == 12:
                                wall = DAY if mode is None else (DAY + dt.timedelta(days=1)).replace(hour=0, minute=3)
                                real_raw, sends = eod(picks, wall=wall)
                                print(f"{name} historical={mode is not None}: {type(error).__name__} "
                                      f"errno={error.errno}; base coverage={old[0]}; candidate runner={real_raw}; DM count={len(sends)}")
                                assert any(item.incident_key == "health_runner:contest_state" for item in real_raw)
                                assert not any(item.source == "contest_state" for item in real_raw)
                        else:
                            expected = not base and state == "used" and name in {"active", "not_earned"}
                            assert len(mismatch(raw)) == int(expected), (state, streak, name, mode, raw)
                            assert raw == old + mismatch(raw), (state, streak, name, mode, raw, old)
                            if expected:
                                warn, = mismatch(raw)
                                assert (warn.level, warn.source) == ("WARN", "saver_state")
                                assert str(DAY.year) in warn.message and name in warn.message
                        rows += 1
                finally:
                    if private is not None:
                        private.chmod(0o700)
    print(f"{rows} raw checks across {len(FLAGS)} flag cases, seven streaks, three dated states and both modes")


def year_boundary(root, base):
    january = (DECEMBER + dt.timedelta(days=1)).replace(hour=0, minute=4)
    for day in (DECEMBER, january):
        picks = root / str(day.date()) / "picks"
        observed(picks, day)
        flag(picks, "active", day.year)
        for now in (day, day.astimezone(dt.timezone.utc)):
            assert len(mismatch(health.check(picks, now=now))) == int(not base)
        flag(picks, "active", day.year - 1)
        assert not mismatch(health.check(picks, now=day))
    picks = root / str(DECEMBER.date()) / "picks"
    flag(picks, "active", DECEMBER.year)
    old_alerts, old_dms = eod(picks, DECEMBER, wall=january)
    assert len(mismatch(old_alerts)) == len(old_dms) == int(not base)
    for change in ("flag", "observation"):
        if change == "flag":
            flag(picks, "active", january.year)
        else:
            flag(picks, "active", DECEMBER.year)
            observed(picks, january)
        raw, _ = eod(picks, DECEMBER, wall=january)
        assert not mismatch(raw)
    print("December 31 23:52 ET (already January UTC), January 1 00:04 ET and December rerun in January verified")


if __name__ == "__main__":
    base = sys.argv[1:] == ["base"]
    print(f"bts.__file__ = {bts.__file__}")
    with scratch() as directory:
        run_grid(directory, base)
        year_boundary(directory, base)
