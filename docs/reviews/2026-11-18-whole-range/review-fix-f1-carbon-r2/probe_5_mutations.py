"""Item 8: Scratch mutation checks on an isolated copy of src/ (never touching tracked files)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRATCH_ROOT = HERE / "scratch_mut_pkg"
SCRATCH_SRC = SCRATCH_ROOT / "src"

MUTATIONS = [
    # The 6 required scratch changes from BRIEF item 8:
    (
        "brief_1_drop_int_game_pk",
        'if result["state"] == "wrong_game" and type(slot["game_pk"]) is int]',
        'if result["state"] == "wrong_game"]',
    ),
    (
        "brief_2_drop_shared_batter",
        'if len({slot["batter_id"] for slot in slots}) != len(slots):',
        "if False:",
    ),
    (
        "brief_3_drop_case_b",
        'if not affected and prior_wrong_game and not qualification["all_confirmed"]:',
        "if False:",
    ),
    (
        "brief_4_first_alert_consumes_initial",
        '        if reason != "wrong_game":\n            consumed.add("initial")\n        elif tier == "wrong_game":\n            consumed.add("wrong_game")',
        '        consumed.add("initial")\n        if tier == "wrong_game":\n            consumed.add("wrong_game")',
    ),
    (
        "brief_5_confirm_when_any_leg_confirmed",
        'if not affected and prior_wrong_game and not qualification["all_confirmed"]:',
        'if not affected and prior_wrong_game and not any(r["state"] == "confirmed" for _, r in results):',
    ),
    (
        "brief_6_swap_texts",
        '        elif tier == "wrong_game":',
        '        elif tier != "wrong_game":',
    ),
    # Additional adversarial sub-mutations on Round 1 & Round 2 guards:
    (
        "extra_7_allow_bool_game_pk_isinstance",
        'type(slot["game_pk"]) is int',
        'isinstance(slot["game_pk"], int)',
    ),
    (
        "extra_8_allow_non_int_game_pk_not_none",
        'type(slot["game_pk"]) is int',
        'slot["game_pk"] is not None',
    ),
    (
        "extra_9_legs_game_id_only_checks_none",
        'game = game_pk if type(game_pk) is int else "ID unavailable"',
        'game = game_pk if game_pk is not None else "ID unavailable"',
    ),
    (
        "extra_10_omit_game_unresolved_on_present_unverified",
        'if marker["status"] != "confirmed" and (reason == "wrong_game" or prior_wrong_game):',
        'if marker["status"] not in ("confirmed", "present_unverified") and (reason == "wrong_game" or prior_wrong_game):',
    ),
    (
        "extra_11_omit_game_unresolved_on_dm_failed",
        'if marker["status"] != "confirmed" and (reason == "wrong_game" or prior_wrong_game):',
        'if marker["status"] not in ("confirmed", "dm_failed") and (reason == "wrong_game" or prior_wrong_game):',
    ),
]


def run_mutations():
    shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)
    shutil.copytree(REPO / "src", SCRATCH_SRC)
    cli_path = SCRATCH_SRC / "bts" / "cli.py"
    original_cli = cli_path.read_text()

    results = {}
    try:
        for name, target, replacement in MUTATIONS:
            assert target in original_cli, f"Target not found for {name}: {target}"
            mutated = original_cli.replace(target, replacement, 1)
            cli_path.write_text(mutated)

            env = {**os.environ, "PYTHONPATH": str(SCRATCH_SRC), "TZ": "America/New_York"}
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "--tb=no",
                    "-k",
                    "not fifo",
                    "tests/test_wr_f1_wrong_game.py",
                    "tests/test_wr_f1_review.py",
                ],
                cwd=str(REPO),
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            killed_by_author = proc.returncode != 0

            # If author's tests missed it, check whether our probe_2_rows_and_texts.py catches it!
            killed_by_probe2 = None
            if not killed_by_author:
                env2 = {**os.environ, "PYTHONPATH": f"{SCRATCH_SRC}:{HERE}", "TZ": "America/New_York"}
                p2 = subprocess.run(
                    [sys.executable, str(HERE / "probe_2c_rows_and_texts.py")],
                    cwd=str(REPO),
                    env=env2,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                killed_by_probe2 = p2.returncode != 0

            results[name] = {
                "killed_by_author_tests": killed_by_author,
                "pytest_summary": proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "",
                "killed_by_our_probe2": killed_by_probe2,
            }
            cli_path.write_text(original_cli)
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)

    (HERE / "log_5_mutations.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run_mutations()
