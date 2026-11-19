"""Run focused pytest jobs serially through the shared yield wrapper."""

from datetime import datetime
import shutil
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE / "round1"))
import probe_support
from probe_support import archive, ET

probe_support.REPO = REPO
WRAPPER = Path.home() / "bts-runs" / "yield_run.sh"
NEW = sorted((REPO / "tests").glob("test_wr_c3_*.py"))
COMBINED = sorted(set(NEW + [
    *list((REPO / "tests").glob("test_streak_*.py")),
    *list((REPO / "tests").glob("test_reconcile_*.py")),
    *(REPO / "tests" / path for path in ("test_scoring_lock.py", "test_lightgbm_optional.py",
        "test_web_saver.py", "test_web_streak_subtitle.py", "health/test_streak_validation.py")),
]))

MUTATIONS = {
    "no-lock": ('            if state["pick_locked"]:', '            if False:'),
    "missing-preview": ('(daily.chosen_on is None or daily.chosen_on["required"])',
                        '(daily.chosen_on is not None and daily.chosen_on["required"])'),
    "skip-over-delivery": ('    if delivery_signal:', '    if delivery_signal and decision is None:'),
    "strict-terminal": ('                if result == "hit":\n                    daily = _daily_pick_from_data(data)',
                        '                if result in _TERMINAL_RESULTS:\n'
                        '                    _validate_reconcile_pick(data.copy(), day)\n'
                        '                if result == "hit":\n                    daily = _daily_pick_from_data(data)'),
    "ignore-invalid-decision": ('            return None, (day, "unreadable or invalid decision record")',
                                '            decision = None'),
}


def wait_for_wrapper():
    while True:
        result = subprocess.run(["pgrep", "-f", str(WRAPPER)], capture_output=True, text=True)
        if result.returncode == 1:
            return
        if result.returncode != 0:
            print(result.stderr.strip(), flush=True)
            raise SystemExit("Unable to check the shared test queue")
        print("Waiting for the shared test wrapper", flush=True)
        time.sleep(15)


def run(name, root, files, extra_env=(), initializer=()):
    wait_for_wrapper()
    log = HERE / f"{name}.log"
    command = [str(WRAPPER), str(log), "env", "BTS_YIELD_SECONDS=2", "TZ=America/New_York",
               "PYTHONDONTWRITEBYTECODE=1", f"PYTHONPATH={root / 'src'}", *extra_env, sys.executable,
               *initializer, "-p", "no:cacheprovider", *[str(f.relative_to(root)) for f in files],
               "-q", "--tb=short", "-o", "console_output_style=count"]
    # Set the wrapper's polling interval in its own process environment.
    import os
    completed = subprocess.run(command, cwd=root, env=os.environ | {"BTS_YIELD_SECONDS": "2"}, timeout=1200)
    print(name, completed.returncode, flush=True)


def tests_on_revision(name, revision, mutation=None):
    with TemporaryDirectory(dir=HERE, prefix=f"{name}-") as tmp:
        root = archive(revision, Path(tmp), "src", "tests", "pyproject.toml", "config")
        for test in NEW:
            shutil.copyfile(test, root / "tests" / test.name)
        if mutation:
            source = root / "src/bts/picks.py"
            before, after = mutation
            original = source.read_text()
            assert original.count(before) == 1
            source.write_text(original.replace(before, after))
        run(name, root, [root / "tests" / test.name for test in NEW], initializer=(str(HERE / "offline_pytest.py"),))


def wall(year):
    target = datetime(year, 4, 20, 2, tzinfo=ET)
    offset = round(target.timestamp() - time.time())
    dylib = HERE / "wall_clock.dylib"
    run(f"wall-{year}", REPO, COMBINED, (f"DYLD_INSERT_LIBRARIES={dylib}", f"WR_C3_WALL_OFFSET={offset}",
                                        f"WR_C3_WALL_YEAR={year}"), (str(HERE / "offline_pytest.py"),))


if __name__ == "__main__":
    for mode in sys.argv[1:]:
        if mode == "red":
            for revision in ("c70effe9", "c2039541"):
                tests_on_revision(f"red-{revision}", revision)
        elif mode == "mutations":
            for name, mutation in MUTATIONS.items():
                tests_on_revision(f"mutant-{name}", "aa61bb43", mutation)
        elif mode == "native":
            run("native", REPO, COMBINED, initializer=(str(HERE / "offline_pytest.py"),))
        else:
            wall(int(mode))
