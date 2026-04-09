"""Re-run diff_scorecards over all phase1 experiment results.

Usage: uv run python scripts/redo_diffs.py [results_dir]
Default results_dir: experiments/results/phase1

Re-reads each experiment's scorecard.json, re-runs diff against the cached
baseline, and overwrites diff.json + summary.txt with corrected values.
Idempotent — safe to run repeatedly.
"""

import json
import sys
from pathlib import Path

from bts.experiment.runner import evaluate_pass_fail
from bts.validate.scorecard import diff_scorecards


def _json_default(obj):
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


def redo_diffs(phase1_dir: Path) -> None:
    baseline_path = phase1_dir / "baseline_scorecard.json"
    if not baseline_path.exists():
        print(f"No baseline scorecard at {baseline_path}", file=sys.stderr)
        return

    baseline = json.loads(baseline_path.read_text())
    print(f"Loaded baseline from {baseline_path}")

    for exp_dir in sorted(phase1_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        scorecard_path = exp_dir / "scorecard.json"
        if not scorecard_path.exists():
            print(f"  {exp_dir.name}: no scorecard.json — skip")
            continue

        variant = json.loads(scorecard_path.read_text())
        diff = diff_scorecards(baseline, variant)
        passed, reason = evaluate_pass_fail(diff)

        diff_path = exp_dir / "diff.json"
        summary_path = exp_dir / "summary.txt"
        diff_path.write_text(json.dumps(diff, indent=2, default=_json_default))
        status = "PASS" if passed else "FAIL"
        summary_path.write_text(f"{status} | {reason}")

        # One-line summary
        p1_2024 = diff.get("p_at_1_by_season", {}).get("2024", {}).get("delta")
        if p1_2024 is None:
            p1_2024 = diff.get("p_at_1_by_season", {}).get(2024, {}).get("delta")
        p1_2025 = diff.get("p_at_1_by_season", {}).get("2025", {}).get("delta")
        if p1_2025 is None:
            p1_2025 = diff.get("p_at_1_by_season", {}).get(2025, {}).get("delta")
        p57 = diff.get("p_57_mdp", {}).get("delta")

        def _fmt(d):
            if d is None:
                return "  N/A"
            return f"{d:+.4f}"

        print(f"  {exp_dir.name:<24} {status} | "
              f"P@1 2024={_fmt(p1_2024)} 2025={_fmt(p1_2025)} P(57)={_fmt(p57)}")


if __name__ == "__main__":
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/results/phase1")
    redo_diffs(base)
