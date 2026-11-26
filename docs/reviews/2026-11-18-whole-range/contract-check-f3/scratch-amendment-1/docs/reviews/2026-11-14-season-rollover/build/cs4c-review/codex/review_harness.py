"""Run the targeted tests, archived-base controls and scratch mutants offline."""

import datetime as calendar
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch


SCRATCH = Path(__file__).resolve().parent
ROOT = SCRATCH.parent.parent
TESTS = [
    "tests/test_automatic_reading_cs4c.py",
    "tests/test_stale_delivery_cs4c.py",
    "tests/test_pick_chosen_on_cs5a.py",
    "tests/test_saved_pick_paths_cs5a.py",
]
EXTRAS = ["tests/test_lightgbm_optional.py", "tests/test_web_saver.py"]
SCHEDULER = "bts/scheduler.py"
CONTEST = "bts/contest_state.py"
STAMP = "_stamp_reading(now)\n"
WARN = "_warn_stale_contest_reading(daily.automatic_reading, config, picks_dir)"
GUARD = """    if now >= cutoff:
        _refuse_late_delivery(daily, config, picks_dir, state, date, label, cutoff, now)
        return False
"""


def edit(old, new, *, times=1):
    def transform(text):
        assert text.count(old) == times, (old, text.count(old), times)
        return text.replace(old, new)
    return transform


def move_stamp(indent):
    block = "".join(indent + line if line.strip() else line
                    for line in GUARD.splitlines(keepends=True))
    stamp = indent + "    " + STAMP
    return edit(block + stamp, stamp + block)


def move_warning(text):
    for indent in ("        ", "    "):
        old = f"\n{indent}if required:\n{indent}    {WARN}\n{indent}return True"
        assert text.count(old) == 1
        text = text.replace(old, f"\n{indent}return True")
        line = f"\n{indent}daily.delivery_attempted = True  # persist BEFORE the network call (E2 idempotency)"
        assert text.count(line) == 1
        text = text.replace(line, f"\n{indent}if required:\n{indent}    {WARN}" + line)
    return text


def failed_warning(text):
    marker = "daily.delivery_attempted = False  # known failure → clear so a later cycle retries"
    return edit(marker, WARN + "\n            " + marker, times=2)(text).replace(
        "\n            " + marker + "\n        save_pick", "\n        " + marker + "\n        save_pick")


def recover_warning(text):
    old = '        print(f"  LOCKED ({label}) — pick already delivered.", file=sys.stderr)'
    return edit(old, old + "\n        " + WARN)(text)


MUTANTS = {
    "no-stamp-dm": (SCHEDULER, edit("\n        " + STAMP, "\n"),
                    "attempt_snapshot or dated_attempt or send_failure or restart_preserves"),
    "no-stamp-public": (SCHEDULER, edit("\n    " + STAMP, "\n"),
                        "attempt_snapshot or dated_attempt or send_failure or restart_preserves"),
    "stamp-before-dm-cutoff": (SCHEDULER, move_stamp("    "), "final_cutoff"),
    "stamp-before-public-cutoff": (SCHEDULER, move_stamp(""), "final_cutoff"),
    "warning-before-send": (SCHEDULER, move_warning, "attempt_snapshot"),
    "warning-after-failed-send": (SCHEDULER, failed_warning, "send_failure"),
    "shared-status": (SCHEDULER, edit('"contest_stale_reading_dm_delivery_status.json"',
                                      '"health_dm_delivery_status.json"'), ""),
    "implicit-warning-date": (SCHEDULER, edit("                now_et_date=warning_day,\n", ""), ""),
    "no-incident-key": (SCHEDULER, edit(', incident_key="contest_state:stale_reading"', ""),
                        "dedup or status"),
    "old-at-13": (CONTEST, edit("AUTOMATIC_READING_STALE_HOURS = 14",
                               "AUTOMATIC_READING_STALE_HOURS = 13"), "strict_elapsed or all_pick_loaders"),
    "old-at-or-above-14": (CONTEST, edit("elif elapsed > timedelta", "elif elapsed >= timedelta"),
                          "strict_elapsed"),
    "future-at-six-minutes": (CONTEST, edit("timedelta(minutes=-5)", "timedelta(minutes=-6)"),
                              "strict_elapsed"),
    "accept-age-int": (CONTEST, edit("type(age) is not float", "type(age) not in (float, int)"),
                       "all_pick_loaders"),
    "accept-age-bool": (CONTEST, edit("type(age) is not float", "type(age) not in (float, bool)"),
                        "all_pick_loaders"),
    "stamp-other-kinds": (CONTEST, edit('    if s.observation_kind != "automatic":\n        return None\n', ""),
                          "pure_kinds or dated_attempt"),
    "warning-exception-escapes": (SCHEDULER,
                                  edit("    except Exception as error:\n        _report_stale_reading_warning_failure(error)\n",
                                       "    except Exception:\n        raise\n"),
                                  "warning_failures"),
    "stamp-private": (SCHEDULER, edit('    if mode == "private":\n        save_pick',
                                     '    if mode == "private":\n'
                                     '        automatic_reading_record(None, now)\n        save_pick'),
                      "never_compute or guards"),
    "stamp-not-required": (SCHEDULER, edit("        if required:\n            try:\n                daily.automatic_reading",
                                          "        if True:\n            try:\n                daily.automatic_reading"),
                           "never_compute or guards"),
    "warning-in-recovery": (SCHEDULER, recover_warning,
                            "restart_preserves or attempt_snapshot"),
}


@contextmanager
def wall_clock(when):
    if not when:
        yield
        return
    import pandas  # Import datetime-dependent extensions before replacing the clock.

    native_datetime = calendar.datetime
    native_date = calendar.date
    pinned = native_datetime.fromisoformat(when)

    class DateTimeType(type):
        def __instancecheck__(cls, value):
            return isinstance(value, native_datetime)

    class DateType(type):
        def __instancecheck__(cls, value):
            return isinstance(value, native_date)

    class PinnedDateTime(native_datetime, metaclass=DateTimeType):
        @classmethod
        def now(cls, tz=None):
            return pinned.astimezone(tz) if tz is not None else pinned.replace(tzinfo=None)

        @classmethod
        def today(cls):
            return cls.now()

        @classmethod
        def utcnow(cls):
            return pinned.astimezone(calendar.timezone.utc).replace(tzinfo=None)

    class PinnedDate(native_date, metaclass=DateType):
        @classmethod
        def today(cls):
            return pinned.date()

    with patch.object(calendar, "datetime", PinnedDateTime), patch.object(calendar, "date", PinnedDate):
        print("FROZEN", calendar.datetime.now(), calendar.date.today(), flush=True)
        yield


def run_pytest(src, script_root, files, logfile, *, clock=None, expression=""):
    env = os.environ.copy()
    env.pop("NOTIFY_SOCKET", None)
    env.update(
        CS4C_OFFLINE="1", TZ="America/New_York", PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=os.pathsep.join(map(str, (SCRATCH, src, script_root, ROOT))),
        CS4C_CLOCK=clock or "",
    )
    args = [sys.executable, str(Path(__file__).resolve()), "pytest",
            "-p", "no:cacheprovider", "-q", "--tb=line", "-rA", "--color=no", *files]
    if expression:
        args.extend(["-k", expression])
    destination = SCRATCH / logfile
    with destination.open("w") as output:
        completed = subprocess.run(args, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
    lines = destination.read_text().splitlines()
    summary = next((line for line in reversed(lines) if line.startswith("===")), "")
    print(f"{logfile} rc={completed.returncode} {summary}", flush=True)
    assert not any("OFFLINE GUARD BLOCKED" in line for line in lines)
    return completed.returncode


def main():
    mode = sys.argv[1]
    if mode == "pytest":
        import pytest
        import sitecustomize
        from bts import contest_state
        import scripts

        assert os.environ.get("CS4C_OFFLINE") == "1"
        assert sitecustomize.__file__ == str(SCRATCH / "sitecustomize.py")
        print(f"SOURCE {contest_state.__file__}\nSCRIPTS {scripts.__file__}", flush=True)
        # Do not preload contest_state before freezing the clock it imports.
        del sys.modules["bts.contest_state"]
        import bts
        del bts.contest_state
        with wall_clock(os.environ.get("CS4C_CLOCK")):
            return pytest.main(sys.argv[2:])
    if mode == "verification":
        runs = [run_pytest(ROOT / "src", ROOT, TESTS + EXTRAS, "combined.log")]
        for year in (2027, 2028):
            runs.append(run_pytest(ROOT / "src", ROOT, TESTS, f"clock-{year}.log",
                                   clock=f"{year}-07-15T12:00:00-04:00"))
        base = SCRATCH / "base"
        runs.append(run_pytest(base / "src", base, TESTS, "red-base.log"))
        return int(runs != [0, 0, 0, 1])
    if mode != "mutations":
        raise ValueError(mode)
    selection = sys.argv[2:] or MUTANTS.keys()
    with tempfile.TemporaryDirectory(dir=SCRATCH, prefix="mutations-") as folder:
        src = Path(folder) / "src"
        shutil.copytree(ROOT / "src", src)
        for name in selection:
            filename, transform, expression = MUTANTS[name]
            source = src / filename
            original = source.read_text()
            changed = transform(original)
            compile(changed, str(source), "exec")
            try:
                source.write_text(changed)
                run_pytest(src, ROOT, TESTS[:2], f"mutant-{name}.log", expression=expression)
            finally:
                source.write_text(original)
    return 0


if __name__ == "__main__":
    sys.exit(main())
