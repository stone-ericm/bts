"""Adversarial Probe 4C: Verify mutant m13c (`orchestrate` re-opens `config_path` directly when `config` is passed,
while keeping `load_config(config_path)` for the 2-arg call):
passes `tests/test_live_gate_cs6a.py` (false green) and fails `adopt/test_cs6a_adversarial_adopt.py` (killed).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REVIEW_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REVIEW_DIR))

import probe_mutations_and_clocks as pmc


def main() -> None:
    pmc.ensure_wrapper_executable()
    try:
        msrc = pmc.make_mutant_src(
            "m13c_orchestrate_else_direct_open",
            "bts/orchestrator.py",
            "    if config is None:\n        config = load_config(config_path)",
            '    if config is None:\n        config = load_config(config_path)\n    else:\n        with open(config_path, "rb") as _f:\n            config = tomllib.load(_f)',
        )
        r_author = pmc.run_pytest_under_harness(["tests/test_live_gate_cs6a.py"], override_src=msrc)
        r_adopt = pmc.run_pytest_under_harness([str(pmc.ADOPT_TEST)], override_src=msrc)
        res = {
            "m13c_killed_by_test_live_gate_cs6a": r_author["returncode"] != 0,
            "m13c_author_returncode": r_author["returncode"],
            "m13c_killed_by_adopt": r_adopt["returncode"] != 0,
            "m13c_adopt_failures": r_adopt["failed"] + r_adopt["errors"],
        }
        (REVIEW_DIR / "m13c_summary.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        assert res["m13c_killed_by_test_live_gate_cs6a"] is False
        assert res["m13c_killed_by_adopt"] is True
    finally:
        if pmc.HARNESS_CFG.exists():
            pmc.HARNESS_CFG.unlink()


if __name__ == "__main__":
    main()
