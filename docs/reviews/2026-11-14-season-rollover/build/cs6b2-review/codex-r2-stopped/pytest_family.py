import importlib.util
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from review_common import ET, HERE, ROOT, SOURCE, archive, offline

AUTHORED = "tests/health/test_alert_status_cs6b2.py"
LEAD = "tests/health/test_alert_status_cs6b2_review.py"
FEATURE = [
    AUTHORED,
    LEAD,
    "tests/test_alert_status_callers_cs6b2.py",
    "tests/test_alert_status_cs6b2_carbon.py",
]
NEIGHBORS = [
    "tests/test_lightgbm_optional.py",
    "tests/test_web_saver.py",
    "tests/test_morning_state_cs6b1.py",
    "tests/test_saved_pick_paths_cs5a.py",
    "tests/health/test_alert.py",
]
HEAD = str(ROOT / "src")
DYLIB = str(HERE / "libos_probe.dylib")


def blocked_dm(*_args, **_kwargs):
    raise AssertionError("test attempted an uncaptured DM")


def state():
    import bts.strategy as strategy

    mask = os.umask(0)
    os.umask(mask)
    return {
        "cwd": os.getcwd(),
        "umask": mask,
        "stat": Path.stat,
        "open": os.open,
        "alarm handler": signal.getsignal(signal.SIGALRM),
        "alarm timer": signal.getitimer(signal.ITIMER_REAL),
        "recursion limit": sys.getrecursionlimit(),
        "integer digits": sys.get_int_max_str_digits(),
        "decision cache identity": id(strategy._mdp_cache),
    }


def pytest_child(src, *args):
    import pytest

    offline()
    sys.path[:0] = [src, str(ROOT)]
    from bts import dm

    dm.send_dm = blocked_dm
    from bts.health import alert

    assert Path(alert.__file__).resolve() == (Path(src) / "bts/health/alert.py").resolve()
    assert alert.send_dm is blocked_dm
    clock = os.environ.get("BTS_REVIEW_CLOCK_YEAR")
    if clock:
        expected = (int(clock), 7, 15)
        wall = datetime.now(ET)
        print(f"native wall={wall.isoformat()} time.time={time.time():.3f} "
              f"monotonic={time.monotonic():.3f}", flush=True)
        assert (wall.year, wall.month, wall.day) == expected
        assert abs(wall.timestamp() - time.time()) < 2

    class Audit:
        @pytest.hookimpl(wrapper=True, tryfirst=True)
        def pytest_runtest_protocol(self, item, nextitem):
            before = state()
            try:
                return (yield)
            finally:
                after = state()
                changed = {key: (before[key], after[key]) for key in before
                           if before[key] != after[key]}
                assert not changed, (item.nodeid, changed)

    sys.exit(pytest.main(["-p", "no:cacheprovider", "-q", "--tb=short", *args],
                         plugins=[Audit()]))


def pytest_run(name, src, args, expected=0, *, year=None):
    print(f"=== {name} ===", flush=True)
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               PYTHONDONTWRITEBYTECODE="1", TZ="America/New_York")
    if year:
        env.update(
            DYLD_INSERT_LIBRARIES=DYLIB,
            BTS_REVIEW_CLOCK_YEAR=str(year),
            BTS_REVIEW_WALL_SHIFT=str(int(datetime(year, 7, 15, 12, tzinfo=ET).timestamp()
                                          - time.time())),
        )
    result = subprocess.run([sys.executable, __file__, "child", str(src), *args],
                            cwd=ROOT, env=env, timeout=180, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end="", flush=True)
    print(f"{name} exit={result.returncode} expected={expected}", flush=True)
    assert result.returncode == expected, name
    return result.stdout


def variants():
    archive_root = HERE / "src-38d707bd"
    old_src = archive_root / "src"
    if not (archive_root / SOURCE).exists():
        archive("38d707bd", archive_root)

    baseline = subprocess.check_output(["git", "show", "HEAD:" + SOURCE], cwd=ROOT, text=True)
    write_start = baseline.index("def _write_regular_file_text")
    read_start = baseline.index("def _read_regular_file_text")
    helper_end = baseline.index("def _persist_health_dm_delivery_status")
    write = baseline[write_start:read_start]
    read = baseline[read_start:helper_end]
    check = ('        if not stat.S_ISREG(os.fstat(fd).st_mode):\n'
             '            raise OSError("not a regular file")\n')
    truncate = "        os.ftruncate(fd, 0)\n"
    close = "        os.close(fd)\n"
    tests = [
        ("read_blocking", write, read.replace(" | os.O_NONBLOCK", ""),
         "test_fifo_swapped_in_before_the_read_cannot_block and empty", 1),
        ("write_blocking", write.replace(" | os.O_NONBLOCK", ""), read,
         "test_fifo_swapped_in_before_the_write_cannot_block and critical-False", 1),
        ("read_no_fstat", write, read.replace(check, ""),
         "test_bytes_waiting_in_a_fifo", 1),
        ("write_no_fstat", write.replace(check, ""), read,
         "test_nothing_is_truncated", 1),
        ("early_truncate", write.replace(check + truncate, truncate + check), read,
         "test_nothing_is_truncated", 1),
        ("no_truncate", write.replace(truncate, ""), read,
         "test_regular_and_dangling_writes", 1),
        ("read_leak", write, read.replace(close, "        pass\n"),
         "test_no_descriptor_is_left_open and fifo_read", 1),
        ("write_leak", write.replace(close, "        pass\n"), read,
         "test_no_descriptor_is_left_open and fifo_write", 1),
        ("extra_otrunc", write.replace(" | os.O_NONBLOCK", " | os.O_NONBLOCK | os.O_TRUNC"), read,
         "test_nothing_is_truncated", 0),
    ]
    yield "head-lead", HEAD, [LEAD], 0
    yield "round1-lead", old_src, [LEAD], 1
    scratch = HERE / "mutation-src"
    if not (scratch / SOURCE).exists():
        archive("HEAD", scratch)
    for name, w, r, selection, expected in tests:
        changed = baseline[:write_start] + w + r + baseline[helper_end:]
        assert changed != baseline, name
        (scratch / SOURCE).write_text(changed)
        yield name, scratch / "src", [LEAD, "-k", selection], expected


if __name__ == "__main__":
    action, *args = sys.argv[1:]
    if action == "child":
        pytest_child(*args)
    elif action == "mutations":
        for name, src, nodes, expected in variants():
            output = pytest_run(name, src, nodes, expected)
            if name == "round1-lead":
                assert output.count("dispatcher blocked on a replaced status path") >= 5
                assert "test_fifo_swapped_in_before_the_write_cannot_block" in output
            if name in ("read_blocking", "write_blocking"):
                assert "dispatcher blocked on a replaced status path" in output
            if expected:
                assert " failed" in output and "INTERNALERROR" not in output
    elif action == "combined":
        print("host clocks:", datetime.now(ET).isoformat(), time.monotonic(), flush=True)
        for year in (None, 2027, 2028):
            for order, files in (("feature-first", FEATURE + NEIGHBORS),
                                 ("neighbors-first", list(reversed(FEATURE + NEIGHBORS)))):
                pytest_run(f"{year or 'host'}-{order}", HEAD, files, year=year)
    else:
        raise ValueError(action)
