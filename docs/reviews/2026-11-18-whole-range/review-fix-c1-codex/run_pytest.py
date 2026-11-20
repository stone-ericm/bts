"""Run the requested pytest scope or both new test files against one verified mutant."""

from pathlib import Path
import os
import runpy
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
C1 = ["tests/test_wr_c1_clock.py", "tests/test_wr_c1_history.py"]
FOCUSED = C1 + [
    "tests/health/test_runner.py", "tests/health/test_contest_state.py", "tests/test_scheduler.py",
    "tests/test_wr_c2_recovery.py", "tests/test_wr_b_contract.py", "tests/test_wr_b_edges.py",
    "tests/test_lightgbm_optional.py", "tests/test_web_saver.py",
]


def main():
    (name,) = sys.argv[1:]
    if name == "focused":
        source = ROOT / "src"
        selection = FOCUSED
    else:
        from prepare_mutants import MUTANTS

        assert name in MUTANTS, name
        source = HERE / "mutants" / name / "src"
        selection = C1
    os.chdir(ROOT)
    os.environ["PYTHONPATH"] = os.pathsep.join([str(HERE / "offline"), str(source)])
    runpy.run_path(str(HERE / "offline/sitecustomize.py"))
    sys.path.insert(0, str(source))
    import bts
    from bts.health import contest_state, runner
    import pytest

    for module in (bts, contest_state, runner):
        assert Path(module.__file__).resolve().is_relative_to(source)
    print(f"\n{name}: {bts.__file__}", flush=True)
    return pytest.main([
        "-p", "no:cacheprovider", "-q", "--tb=short", f"--junitxml={HERE / (name + '.xml')}", *selection,
    ])


if __name__ == "__main__":
    raise SystemExit(main())
