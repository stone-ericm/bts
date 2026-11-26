import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from probe_mutations import HERE, REPO, mutation


def deep_child(expect_error):
    from probe_support import DAY, bts, health, observed, put, scratch

    print(f"bts.__file__ = {bts.__file__}")
    original = "[" * 3000 + "]" * 3000
    try:
        json.loads(original)
    except RecursionError:
        print("Installed decoder raises at the test depth (3000)")
    else:
        print("Installed decoder accepts the test's 3000 levels")
    deep = '{"state":"active","season":2027,"extra":' + "[" * 20000 + "]" * 20000 + "}"
    try:
        json.loads(deep)
    except RecursionError:
        print("Installed decoder raises at 20000 levels")
    else:
        raise AssertionError("fixture did not reach decoder recursion limit")
    with scratch() as directory:
        picks = directory / "picks"
        observed(picks)
        put(picks, "2027-07-16.json", {"result": "hit"})
        modes = (None, DAY.date())
        prior = [health.check(picks, now=DAY, expected=True, historical_day=mode) for mode in modes]
        assert all(rows and rows[0].source == "contest_state" for rows in prior)
        (picks / "account_state/saver_state.json").write_text(deep)
        for mode, expected in zip(modes, prior):
            try:
                actual = health.check(picks, now=DAY, expected=True, historical_day=mode)
            except RecursionError:
                assert expect_error
                print(f"historical={mode is not None}: mutant raised RecursionError")
            else:
                assert not expect_error and actual == expected
                print(f"historical={mode is not None}: unchanged alerts with real RecursionError")


def use_source(source):
    return {**os.environ, "PYTHONPATH": f"{source}:{REPO}"}


def fifo_child(source):
    invoke = ("import bts, pytest, sys; print('bts.__file__ =', bts.__file__, flush=True); "
              "raise SystemExit(pytest.main(sys.argv[1:]))")
    args = [sys.executable, "-c", invoke, "-p", "no:cacheprovider", "-q", "--tb=short",
            "--durations=2", "-rA",
            "tests/test_wr_f2_saver_flag.py::test_k2_k3_unusable_flag_preserves_existing_alerts[fifo]",
            "tests/test_contest_dated_cs4a.py::test_health_observation_saver_without_flag_read"]
    test = subprocess.run(args, cwd=REPO, env=use_source(source), timeout=45, check=False)
    assert test.returncode == 1, test.returncode


if __name__ == "__main__":
    case, = sys.argv[1:]
    if case in {"deep-original", "deep-mutant"}:
        deep_child(case == "deep-mutant")
    else:
        assert case in {"fifo", "deep"}
        with TemporaryDirectory(prefix=f"false-green-{case}-", dir=HERE) as directory:
            source = Path(directory) / "src"
            shutil.copytree(REPO / "src", source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            mutation(source, "live-reader" if case == "fifo" else "no-recursion-catch")
            if case == "fifo":
                fifo_child(source)
            else:
                for mode, path in (("original", REPO / "src"), ("mutant", source)):
                    subprocess.run([sys.executable, __file__, f"deep-{mode}"], cwd=REPO,
                                   env=use_source(path), timeout=10, check=True)
