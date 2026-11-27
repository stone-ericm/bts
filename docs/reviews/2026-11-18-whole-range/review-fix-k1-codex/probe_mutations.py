"""Build and run each single change against a scratch source tree."""

import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
NEEDLES = re.compile(
    r"_alert_contest_state_failure|_alert_missed_pick|_refuse_late_delivery|"
    r"_refuse_void_game_delivery|health_dm_delivery_status|dispatch_dm_for_health_alerts|"
    r"_resurface_note|Previous health alert DM FAILED|contest_state.*CRITICAL|"
    r"CONTEST STATE ERROR|FALLBACK BLOCKED"
)
MUTATIONS = {
    "full-reason": ("scheduler.py", "at this check. Reason: {reason}", "at this check. Reason: {str(error)}"),
    "no-cut": ("scheduler.py", '    if len(reason) > 400:\n        reason = reason[:397] + "..."\n', ""),
    "no-date-label": ("scheduler.py", 'else "this slate"', "else slate_date"),
    "saved-wall-date": ("scheduler.py", "ContestStateError(refusal), slate_date=date)",
                        "ContestStateError(refusal), slate_date=None)"),
    "unbounded-count": ("health/alert.py", "{n if n <= 999 else '999+'}", "{n}"),
    "no-date-try": ("scheduler.py",
                    "    try:\n        parsed_date = date_type.fromisoformat(slate_date)\n"
                    "    except (TypeError, ValueError):\n        parsed_date = None\n",
                    "    parsed_date = date_type.fromisoformat(slate_date)\n"),
}


def main():
    mutation, scratch, scope = sys.argv[1:]
    filename, before, after = MUTATIONS[mutation]
    target = Path(scratch) / "src" / "bts" / filename
    original = target.read_text()
    assert original == (ROOT / "src/bts" / filename).read_text()
    assert before in original
    files = ["tests/test_wr_k1_refusal_dm.py"]
    if scope == "all":
        files = sorted(str(p.relative_to(ROOT)) for folder in ("tests", "scripts")
                       for p in (ROOT / folder).rglob("*.py")
                       if NEEDLES.search(p.read_text(errors="replace")))
        print(f"Matched files: {len(files)}", flush=True)
    env = os.environ | {"PYTHONPATH": str(Path(scratch) / "src") + os.pathsep + str(ROOT)}
    with target.open("w") as handle:
        handle.write(original.replace(before, after, 1))
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import bts; print(bts.__file__)"], env=env, cwd=ROOT,
            capture_output=True, text=True, check=True, timeout=5,
        )
        assert str(Path(scratch) / "src") in result.stdout, result.stdout
        print(f"Mutation {mutation} at {result.stdout.strip()}", flush=True)
        run = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=short", *files],
            cwd=ROOT, env=env, timeout=108,
        )
        sys.exit(run.returncode)
    finally:
        target.write_text(original)


if __name__ == "__main__":
    main()
