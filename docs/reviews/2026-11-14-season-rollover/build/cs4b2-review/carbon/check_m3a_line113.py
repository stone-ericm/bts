"""Verify M3a at season_contract.py:113 (inside live_automatic_refusal) is killed by test_season_contract_cs4b2.py,
and record exact GREEN-on-base node breakdown of test_season_contract_cs4b2.py."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True

from run_item10_suite import REPO, WORK, _run_pytest, _verify_import_path

def main() -> None:
    mut_dir = WORK / "mutated_src_m3a"
    if mut_dir.exists():
        shutil.rmtree(mut_dir)
    shutil.copytree(REPO / "src", mut_dir)
    _verify_import_path(mut_dir)
    sc_file = mut_dir / "bts/season_contract.py"
    txt = sc_file.read_text(encoding="utf-8")
    old = '    if state.state == "corrupt":\n        kind = 1'
    new = '    if False:\n        kind = 1'
    assert old in txt
    sc_file.write_text(txt.replace(old, new, 1), encoding="utf-8")
    r_m3a = _run_pytest(["tests/test_season_contract_cs4b2.py"], src_override=mut_dir, extra_args=["-x", "--tb=no"])
    shutil.rmtree(mut_dir)
    assert r_m3a["returncode"] != 0

    # Also get exact per-function breakdown of the 592 passed on base in test_season_contract_cs4b2.py
    r_base_v = _run_pytest(
        ["tests/test_season_contract_cs4b2.py"],
        src_override=WORK / "base/src",
        extra_args=["-v", "--tb=no"],
    )
    by_func: dict[str, dict[str, int]] = {}
    for line in r_base_v["output"].splitlines():
        if "::" in line and (" PASSED" in line or " FAILED" in line):
            node = line.split(" ")[0]
            fn = node.split("::")[1].split("[")[0]
            bucket = "passed" if " PASSED" in line else "failed"
            by_func.setdefault(fn, {"passed": 0, "failed": 0})[bucket] += 1

    out = {
        "m3a_line113_killed": True,
        "m3a_summary": r_m3a["summary_lines"][-1],
        "cs4b2_base_by_function": by_func,
    }
    (WORK / "item10_m3a_and_base_breakdown.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
