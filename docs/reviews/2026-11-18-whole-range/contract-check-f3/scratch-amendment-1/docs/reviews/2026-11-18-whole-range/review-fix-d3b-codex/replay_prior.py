"""Replay the earlier reviewers' exact probe inputs with D3b expectations."""
from pathlib import Path
from subprocess import run
import sys
from tempfile import TemporaryDirectory

from review_box import HERE, ROOT, replace

HISTORY = HERE / "main-docs/docs/reviews/2026-11-18-whole-range"
CODEX = HISTORY / "review-fix-d3-codex-r2"
CARBON = HISTORY / "review-fix-d3-carbon-r2"


def replay(path, arguments, cap):
    command = [sys.executable, str(path), *arguments]
    print("Replay", path.name, *arguments, flush=True)
    result = run(command, cwd=ROOT, timeout=cap)
    return result.returncode


def make_harness(path):
    original = (CODEX / "harness.py").read_text()
    (path / "harness.py").write_text(replace(original, "REPO = HERE.parents[2]", f"REPO = Path({str(ROOT)!r})"))


def boundary(path, args):
    make_harness(path)
    original = (CODEX / "boundary_probe.py").read_text()
    original = replace(original, '''label in {"r2-existing-commit-stage-unpublished", "r2-existing-commit-rename-unpublished",
                                    "r2-actual-begin-published-signal"}''', 'label == "r2-actual-begin-published-signal"')
    original = replace(original, '("legacy-loaded-then-absent", 1, True)', '("legacy-loaded-then-absent", 1, False)')
    original = replace(original, 'assert checked == (scenario == "legacy-loaded-still-visible")', 'assert checked')
    original = replace(original, '''    old = '    if op == "read" and scenario in {"commit-read-124", "commit-read-137"}:'
    shim = replace_one(tests.SHIM, old, old.replace('"commit-read-137"', '"commit-read-137", "commit-read-126"'))''',
                       '    shim = tests.SHIM')
    original = replace(original, '((126, "2"),', '((126, "126"),')
    script = path / "boundary_probe.py"
    script.write_text(original)
    return replay(script, args, 48)


def budget(path, args):
    make_harness(path)
    original = (CODEX / "budget_probe.py").read_text()
    first = original.index("runner_source = ")
    last = original.index("\nwith TemporaryDirectory", first)
    original = original[:first] + '''
from functools import partial
namespace = {"run_box_script": partial(tests.run_box_script, limit=limit)}
''' + original[last:]
    original = replace(original, '''        assert any(label == "system-jobs-proof" for label, _, _ in timeline)
        assert all(call[0] != "pgrep" for call in result["calls"]), result''',
                       '''        assert any(call[0] == "pgrep" for call in result["calls"]), result
        assert "Scheduler stopped: unverified." in result["stdout"]''')
    script = path / "budget_probe.py"
    script.write_text(original)
    return replay(script, args, 315 if args == ["slow-proof"] else 52)


def carbon(path, args):
    name = "probe_r2_audit_new_logic.py"
    original = replace((CARBON / name).read_text(), "REPO = Path(__file__).resolve().parents[3]", f"REPO = Path({str(ROOT)!r})")
    original = original.replace("tempfile.TemporaryDirectory()", "tempfile.TemporaryDirectory(dir=Path(__file__).parent)")
    script = path / name
    script.write_text(original)
    return replay(script, args, 55)


if __name__ == "__main__":
    with TemporaryDirectory(prefix="r2-replay-", dir=HERE) as folder:
        raise SystemExit(globals()[sys.argv[1]](Path(folder), sys.argv[2:]))
