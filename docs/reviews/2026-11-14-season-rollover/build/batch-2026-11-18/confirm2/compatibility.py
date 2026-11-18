"""Check the withdrawal diff and a malformed legacy verifier input."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from review_runner import BASE, REVIEW, ROOT, environment

WITHDRAWN = [
    "src/bts/health/contest_state.py", "src/bts/health/park_drag_freshness.py",
    "src/bts/strategy.py", "tests/test_contest_dated_cs4a.py",
    "tests/health/test_park_drag_freshness.py", "docs/ops/2027-season-start.md",
    "scripts/live_forward_capture_once.py", "src/bts/health/analytics_artifacts_missing.py",
]
LAST_TWO = sorted(WITHDRAWN[:-2] + ["src/bts/experiment/artifacts.py", "tests/experiment/test_artifacts.py"])


def command(*args):
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True)


def malformed_legacy(tree, output):
    from sitecustomize import check
    from bts.experiment import artifacts
    from tests.experiment.test_artifacts import _write_live_preoutcome_artifact

    check()
    assert Path(artifacts.__file__).is_relative_to(tree / "src")
    manifest = _write_live_preoutcome_artifact(output)
    assert "effective_top_n" not in manifest
    manifest["run_kind"] = ["live_forward_preoutcome", "live_forward_preoutcome"]
    (output / "manifest.json").write_text(json.dumps(manifest))
    try:
        report = artifacts.verify_candidate_artifact_pair(
            artifact_dir=output, expected_run_kind="live_forward_preoutcome",
            expected_top_n=2, require_live_preoutcome=True,
        )
    except TypeError as exc:
        result = {"exception": f"{type(exc).__name__}: {exc}"}
    else:
        assert not report["ok"]
        result = {"ok": report["ok"], "failed": [c["name"] for c in report["checks"] if c["status"] == "fail"]}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(tree.name, result, flush=True)


def main():
    if len(sys.argv) > 1:
        malformed_legacy(*map(Path, sys.argv[1:]))
        return
    diff = command("git", "diff", "3d49308d", "HEAD", "--", *WITHDRAWN)
    assert diff.returncode == 0 and not diff.stdout
    print("Empty withdrawal diff for all eight specified paths.")
    matches = command("rg", "-n", "SEASON_OPENER_DATES|season_opener_date|OPENING_NIGHT_2027", "src", "scripts", "tests")
    assert matches.returncode == 1 and not matches.stdout and not matches.stderr
    print("None of the three retired symbols appear in src, scripts, or tests.")
    changed = command("git", "diff", "--name-only", "8cac32ea", "HEAD")
    assert changed.returncode == 0 and sorted(changed.stdout.splitlines()) == LAST_TWO
    stat = command("git", "diff", "8cac32ea", "HEAD", "--stat")
    assert stat.returncode == 0
    print(stat.stdout, end="")
    whitespace = command("git", "diff", "--check", "3d49308d", "HEAD")
    assert whitespace.returncode == 0 and not whitespace.stdout and not whitespace.stderr
    print("Full change set passed git diff --check.")
    scratch = Path(tempfile.mkdtemp(prefix="probe-legacy-", dir=REVIEW))
    print(f"Legacy compatibility evidence: {scratch.name}", flush=True)
    for label, tree in (("base", BASE), ("head", ROOT)):
        subprocess.run([sys.executable, __file__, str(tree), str(scratch / label)],
                       cwd=tree, env=environment(tree), check=True)


if __name__ == "__main__":
    main()
