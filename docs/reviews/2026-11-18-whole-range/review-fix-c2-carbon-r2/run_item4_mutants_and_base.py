"""Run base RED checks and all 12 scratch mutants sequentially ONE AT A TIME via /usr/bin/nice -n 19."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path("/Users/eric/projects/bts-work/wr-c2-r2-carbon")
REVIEW_DIR = ROOT / ".codex-review" / "wr-c2-r2"
BASE_SRC = REVIEW_DIR / "base_386aa846" / "src"
MUTANTS_DIR = REVIEW_DIR / "mutants"
LOGS_DIR = REVIEW_DIR / "mutant_logs"
PYTHON = ROOT / ".venv" / "bin" / "python"

RECOVERY = "tests/test_wr_c2_recovery.py"
CARBON_V2 = "tests/test_wr_c2_carbon_v2.py"
R2_ADOPT = ".codex-review/wr-c2-r2/adopt/test_wr_c2_r2_carbon.py"
CALLERS = "tests/test_alert_status_callers_cs6b2.py"


def run_serial_pytest(label: str, src_path: Path, test_files: list[str], log_path: Path) -> dict:
    Path("/tmp/mplconfig").mkdir(exist_ok=True)
    env = dict(
        os.environ,
        PYTHONPATH=str(src_path),
        PYTHONDONTWRITEBYTECODE="1",
        TZ="America/New_York",
        MPLCONFIGDIR="/tmp/mplconfig",
    )
    cmd = [
        "/usr/bin/nice", "-n", "19",
        str(PYTHON), "-m", "pytest", "-p", "no:cacheprovider", "-v", "--tb=no", "-rN",
        *test_files,
    ]
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
    combined = proc.stdout + ("\n--- STDERR ---\n" + proc.stderr if proc.stderr else "")
    log_path.write_text(combined, encoding="utf-8")

    failed_by_file = {tf: [] for tf in test_files}
    passed_by_file = {tf: 0 for tf in test_files}
    for line in proc.stdout.splitlines():
        for tf in test_files:
            if line.startswith(f"{tf}::"):
                if " FAILED" in line or " ERROR" in line:
                    node = line.split(" ")[0]
                    failed_by_file[tf].append(node.split("::", 1)[1])
                elif " PASSED" in line:
                    passed_by_file[tf] += 1

    summary_match = re.search(r"=+ ([^=]+ in [0-9.]+s) =+", proc.stdout)
    summary_line = summary_match.group(1) if summary_match else f"rc={proc.returncode}"
    return {
        "label": label,
        "returncode": proc.returncode,
        "summary": summary_line,
        "failed_counts": {k: len(v) for k, v in failed_by_file.items()},
        "passed_counts": passed_by_file,
        "sample_failures": {k: v[:5] for k, v in failed_by_file.items() if v},
    }


def main() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict = {"base": {}, "mutants": {}}

    # 1. Base 386aa846 RED check
    results["base"] = run_serial_pytest(
        "base_386aa846",
        BASE_SRC,
        [RECOVERY, CARBON_V2, R2_ADOPT, CALLERS],
        LOGS_DIR / "base_386aa846.log",
    )
    print(f"base_386aa846: {results['base']['summary']} failed={results['base']['failed_counts']}", flush=True)

    # 2. All 12 scratch mutants sequentially ONE AT A TIME
    for mut_dir in sorted(MUTANTS_DIR.iterdir()):
        if not mut_dir.is_dir():
            continue
        mname = mut_dir.name
        res = run_serial_pytest(
            mname,
            mut_dir / "src",
            [RECOVERY, CARBON_V2, R2_ADOPT],
            LOGS_DIR / f"{mname}.log",
        )
        committed_kills = res["failed_counts"][RECOVERY] + res["failed_counts"][CARBON_V2]
        res["survived_recovery_alone"] = (res["failed_counts"][RECOVERY] == 0)
        res["survived_carbon_v2_alone"] = (res["failed_counts"][CARBON_V2] == 0)
        res["survived_committed_suite"] = (committed_kills == 0)
        results["mutants"][mname] = res
        print(
            f"{mname}: recovery_fail={res['failed_counts'][RECOVERY]}, "
            f"carbon_v2_fail={res['failed_counts'][CARBON_V2]}, "
            f"r2_adopt_fail={res['failed_counts'][R2_ADOPT]}, "
            f"survived_committed={res['survived_committed_suite']}",
            flush=True,
        )

    out_path = REVIEW_DIR / "item4_mutants_and_base.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    survivors = [m for m, r in results["mutants"].items() if r["survived_committed_suite"]]
    print(f"DONE: {len(results['mutants'])} mutants tested sequentially; committed-suite survivors={survivors}")


if __name__ == "__main__":
    main()
