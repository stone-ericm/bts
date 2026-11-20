"""Item 6 mutation runner (part 2): remaining read_text and reraise mutants."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HEAD_SRC = REPO / "src"

MUTANTS = [
    (
        "m7a_read_text_in_primary_state_reader",
        "bts/contest_state.py",
        '    malformed = f"The {label} contest streak file is malformed or unreadable."\n    try:\n        raw = read_regular_file_text(path) if historical else path.read_text(encoding="utf-8")',
        '    malformed = f"The {label} contest streak file is malformed or unreadable."\n    try:\n        raw = path.read_text(encoding="utf-8")',
    ),
    (
        "m7b_read_text_in_second_manual_reread",
        "bts/health/contest_state.py",
        "raw = read_regular_file_text(manual_path) if historical else manual_path.read_text()",
        "raw = manual_path.read_text()",
    ),
    (
        "m7c_read_text_in_coverage_scan",
        "bts/contest_state.py",
        '        except ValueError:        # ISO-shaped but invalid (e.g. 2026-99-99)\n            continue\n        try:\n            raw = read_regular_file_text(path) if historical else path.read_text(encoding="utf-8")',
        '        except ValueError:        # ISO-shaped but invalid (e.g. 2026-99-99)\n            continue\n        try:\n            raw = path.read_text(encoding="utf-8")',
    ),
    (
        "m7d_read_text_in_switch_reader",
        "bts/season_contract.py",
        '        raw = read_regular_file_text(path) if historical else path.read_text(encoding="utf-8")',
        '        raw = path.read_text(encoding="utf-8")',
    ),
    (
        "m7e_read_text_in_saver_reader",
        "bts/saver_state.py",
        "        raw = read_regular_file_text(path) if historical else path.read_text()",
        "        raw = path.read_text()",
    ),
    (
        "m8_reraise_historical_diagnostic_as_crash",
        "bts/health/contest_state.py",
        "if historical and not isinstance(exc, _HistoricalContestStateError):",
        "if historical:",
    ),
]


def main() -> None:
    results = []
    with TemporaryDirectory(dir=HERE) as tmp:
        root = Path(tmp)
        for name, rel_path, target, replacement in MUTANTS:
            msrc = root / name / "src"
            shutil.copytree(HEAD_SRC, msrc)
            fpath = msrc / rel_path
            text = fpath.read_text(encoding="utf-8")
            assert target in text, f"Target not found for {name} in {rel_path}"
            assert text.count(target) == 1, f"Ambiguous target for {name}"
            fpath.write_text(text.replace(target, replacement, 1), encoding="utf-8")

            env = {
                **os.environ,
                "TZ": "America/New_York",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(msrc),
            }
            try:
                proc = subprocess.run(
                    [
                        sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "-x",
                        "tests/test_wr_c1_clock.py", "tests/test_wr_c1_history.py",
                    ],
                    cwd=REPO, env=env, capture_output=True, text=True, timeout=45,
                )
                status = "SURVIVED" if proc.returncode == 0 else "KILLED"
                summary = (proc.stdout + "\n" + proc.stderr).strip().splitlines()[-3:]
            except subprocess.TimeoutExpired:
                status = "KILLED_TIMEOUT"
                summary = ["Timed out at 45s"]
            print(f"[{status}] {name}: {' | '.join(summary)}", flush=True)
            results.append({"mutant": name, "file": rel_path, "status": status, "tail": summary})

    (HERE / "mutations_part2_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
