"""Run each review family inside its own yield wrapper, with isolated pytest children."""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import time
from zoneinfo import ZoneInfo


REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parents[1]
CURRENT = ROOT / "src"
REVIEW_TESTS = ["tests/test_stale_delivery_cs4c_review.py", "tests/test_stale_delivery_cs4c_carbon.py"]
COMBINED_TESTS = [
    "tests/test_automatic_reading_cs4c.py", "tests/test_stale_delivery_cs4c.py",
    *REVIEW_TESTS, "tests/test_pick_chosen_on_cs5a.py", "tests/test_saved_pick_paths_cs5a.py",
    "tests/test_lightgbm_optional.py", "tests/test_web_saver.py",
]
ET = ZoneInfo("America/New_York")


def archive(revision, target):
    import io
    import tarfile

    target.mkdir(parents=True, exist_ok=True)
    raw = subprocess.check_output(["git", "archive", revision, "src"], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(raw)) as snapshot:
        snapshot.extractall(target, filter="data")
    return target / "src"


def pytest_child():
    import builtins
    import sitecustomize

    import pytest
    from bts import scheduler

    assert Path(scheduler.__file__).resolve().is_relative_to(Path(os.environ["CS4CR2_SRC"]))
    now = datetime.now(ET)
    wall = os.environ.get("CS4CR2_WALL_DAY")
    if wall:
        assert now.date().isoformat() == wall
        assert datetime.fromtimestamp(time(), ET).date().isoformat() == wall
    print(f"source={scheduler.__file__} wall_ET={now.isoformat()}", flush=True)
    original_print = builtins.print
    original_stderr = sys.stderr
    original_trace = sys.gettrace()
    code = pytest.main(["-p", "no:cacheprovider", "-q", "--tb=short", *sys.argv[2:]])
    assert builtins.print is original_print and sys.stderr is original_stderr
    assert sys.gettrace() is original_trace
    assert not sitecustomize.blocked
    return code


def run(name, src, tests, day=None):
    env = os.environ | {
        "PYTHONPATH": os.pathsep.join(map(str, [REVIEW, src, ROOT])),
        "CS4CR2_SRC": str(src),
    }
    if day:
        noon = datetime.fromisoformat(f"{day}T12:00:00").replace(tzinfo=ET)
        env.update({
            "CS4CR2_WALL_DAY": day,
            "CS4CR2_WALL_OFFSET": str(int(noon.timestamp() - time())),
            "DYLD_INSERT_LIBRARIES": str(REVIEW / "wall_clock.dylib"),
        })
    print(f"\n=== {name} ===", flush=True)
    result = subprocess.run([sys.executable, __file__, "child", *tests], cwd=ROOT, env=env)
    print(f"=== {name}: exit {result.returncode} ===", flush=True)
    return result.returncode


def mutation_tests():
    import ast
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory(dir=REVIEW, prefix="pytest-src-") as tmp:
        area = Path(tmp)
        old = archive("9fc2ee6d", area / "round1")
        codes = {"round1": run("round1", old, REVIEW_TESTS)}
        scheduler = (CURRENT / "bts" / "scheduler.py").read_text()
        lines = scheduler.splitlines(keepends=True)
        handlers = [node for node in ast.walk(ast.parse(scheduler))
                    if isinstance(node, ast.ExceptHandler)
                    and isinstance(node.type, ast.Name) and node.type.id == "BaseException"]
        assert len(handlers) == 4
        edits = {
            f"handler-line-{handler.lineno}": (handler.lineno, "BaseException", "Exception")
            for handler in sorted(handlers, key=lambda node: node.lineno)
        }
        for identifier in ("msg_id", "uri"):
            condition = f"required and isinstance({identifier}, str) and {identifier}"
            hits = [i for i, line in enumerate(lines, 1) if condition in line]
            assert len(hits) == 1
            edits[f"{identifier}-drop"] = (hits[0], condition, "required")
            edits[f"{identifier}-is-not-none"] = (hits[0], condition, f"required and {identifier} is not None")
        src = area / "mutant" / "src"
        shutil.copytree(CURRENT, src, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name, (line, before, after) in edits.items():
            mutant = lines.copy()
            mutant[line - 1] = mutant[line - 1].replace(before, after, 1)
            (src / "bts" / "scheduler.py").write_text("".join(mutant))
            codes[name] = run(name, src, REVIEW_TESTS)
    print(f"Mutation exits: {codes!r}")
    return int(any(code not in {0, 1} for code in codes.values()) or codes["round1"] != 1)


def combined_tests():
    codes = [run("combined-real", CURRENT, COMBINED_TESTS)]
    for day in ("2027-07-15", "2028-07-15"):
        codes.append(run(f"combined-{day}", CURRENT, COMBINED_TESTS, day))
    return int(any(codes))


if __name__ == "__main__":
    raise SystemExit({"child": pytest_child, "mutations": mutation_tests,
                      "combined": combined_tests}[sys.argv[1]]())
