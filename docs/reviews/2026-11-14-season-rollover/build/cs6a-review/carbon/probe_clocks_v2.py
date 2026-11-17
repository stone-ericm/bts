"""Adversarial Probe 4B (v2): Run all 9 test files under faked wall clocks in 2027, 2028, and NYE UTC/ET split."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
HARNESS_SITE = REVIEW_DIR / "harness_site"
ADOPT_TEST = REVIEW_DIR / "adopt" / "test_cs6a_adversarial_adopt.py"

TARGETS = [
    "tests/test_live_gate_cs6a.py",
    "tests/test_entry_intent.py",
    "tests/test_cli_integration.py",
    "tests/test_cli_schedule_date.py",
    "tests/test_season_contract_cs4b2_edges.py",
    "tests/test_day_outcome.py",
    "tests/test_day_outcome_entry_sequence.py",
    "tests/test_orchestrator.py",
    str(ADOPT_TEST),
]

CLOCKS = [
    ("2027_07_15_et", "2027-07-15T12:00:00-04:00"),
    ("2028_07_15_et", "2028-07-15T12:00:00-04:00"),
    ("2027_12_31_et_2028_utc", "2027-12-31T22:30:00-05:00"),
]


def main() -> None:
    results = {}
    for label, iso in CLOCKS:
        env = dict(
            os.environ,
            TZ="America/New_York",
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=str(HARNESS_SITE),
            CS6A_PYTEST_CLOCK_ISO=iso,
        )
        cmd = [
            sys.executable, "-m", "pytest",
            "-p", "no:cacheprovider",
            "-p", "cs6a_clock_plugin_v2",
            "-q",
            *TARGETS,
        ]
        proc = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=180)
        summary_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        results[label] = {
            "clock_iso": iso,
            "returncode": proc.returncode,
            "summary": summary_line,
            "stderr": proc.stderr.strip(),
        }
        if proc.returncode != 0:
            results[label]["stdout_tail"] = "\n".join(proc.stdout.splitlines()[-30:])

    out_path = REVIEW_DIR / "clock_runs_summary.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    for label, r in results.items():
        assert r["returncode"] == 0, f"Clock run {label} failed:\n{r}"


if __name__ == "__main__":
    main()
