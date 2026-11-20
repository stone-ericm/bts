"""Sequential one-at-a-time runner for Item 7: base RED/GREEN breakdown and 13 scratch mutants."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PYTHON = ROOT / ".venv/bin/python"
NICE = "/usr/bin/nice"
BASE_SRC = HERE / "base/src"
MUTANTS_ROOT = HERE / "mutants"
AUTHOR_TEST = "tests/test_wr_c2_recovery.py"
ADOPT_TEST = ".codex-review/wr-c2-review/adopt/test_wr_c2_carbon_adopted.py"


def verify_import_origin(src_dir: Path) -> None:
    env = dict(os.environ, PYTHONPATH=f"{src_dir}:{ROOT}", PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.check_output(
        [str(PYTHON), "-B", "-c", "import bts.health.alert as a, bts.scheduler as s; print(a.__file__); print(s.__file__)"],
        cwd=str(ROOT), env=env, text=True, timeout=15,
    ).splitlines()
    for line in out:
        assert Path(line).resolve().is_relative_to(src_dir.resolve()), (src_dir, line)


def run_pytest_once(src_dir: Path, test_files: list[str], log_path: Path) -> dict:
    verify_import_origin(src_dir)
    env = dict(os.environ, PYTHONPATH=f"{src_dir}:{ROOT}", PYTHONDONTWRITEBYTECODE="1", TZ="America/New_York")
    cmd = [
        NICE, "-n", "19", str(PYTHON), "-B", "-m", "pytest",
        "-p", "no:cacheprovider", "-o", "pythonpath=", "-ra", "-q", *test_files,
    ]
    proc = subprocess.run(
        cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=120,
    )
    combined = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    log_path.write_text(combined, encoding="utf-8")
    failed_nodes = re.findall(r"^FAILED (\S+)", proc.stdout, flags=re.MULTILINE)
    summary = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    return {
        "returncode": proc.returncode,
        "summary": summary,
        "failed_count": len(failed_nodes),
        "failed_nodes": failed_nodes,
    }


def main() -> None:
    report: dict = {"base": {}, "mutants": {}}

    # 1. Run author's test file against BASE (386aa846)
    base_res = run_pytest_once(BASE_SRC, [AUTHOR_TEST], HERE / "pytest-base-author.log")
    report["base"]["author_test"] = base_res
    base_adopt = run_pytest_once(BASE_SRC, [ADOPT_TEST], HERE / "pytest-base-adopt.log")
    report["base"]["adopt_test"] = base_adopt
    print(f"BASE author: rc={base_res['returncode']} ({base_res['summary']})")
    print(f"BASE adopt:  rc={base_adopt['returncode']} ({base_adopt['summary']})")

    # 2. Run each mutant sequentially (one process at a time!)
    mutant_dirs = sorted(p for p in MUTANTS_ROOT.iterdir() if p.is_dir())
    for mdir in mutant_dirs:
        mname = mdir.name
        msrc = mdir / "src"
        author_m = run_pytest_once(msrc, [AUTHOR_TEST], HERE / f"pytest-mutant-{mname}-author.log")
        survived_author = (author_m["returncode"] == 0)
        adopt_m = run_pytest_once(msrc, [ADOPT_TEST], HERE / f"pytest-mutant-{mname}-adopt.log")
        survived_both = (author_m["returncode"] == 0 and adopt_m["returncode"] == 0)
        report["mutants"][mname] = {
            "survived_author_suite": survived_author,
            "survived_with_adopt": survived_both,
            "author": author_m,
            "adopt": adopt_m,
        }
        status_str = "SURVIVED AUTHOR" if survived_author else f"killed ({author_m['failed_count']} failed)"
        adopt_str = "SURVIVED ADOPT" if adopt_m["returncode"] == 0 else f"killed by adopt ({adopt_m['failed_count']} failed)"
        print(f"  {mname}: {status_str} | {adopt_str}")

    out_json = HERE / "item7_results.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
