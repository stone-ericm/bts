"""Probe 05 (Brief Item 8): Code-change sensitivity checks on an isolated scratch copy of src/.

Copies HEAD src/ into a temporary directory, applies one small code change at a time (without touching
any tracked file), runs tests/test_wr_f1_wrong_game.py (with FIFO sleep shortened via env/arg? No — runs
pytest -k 'not fifo' in ~1.5s first, and if 'not fifo' passes, runs test_fifo_capture_blocks_before_match_marker_only too!)
plus tests/test_entry_receipt.py and tests/test_cli_integration.py -k TestCheckPickEntered to see whether
the existing + new tests catch or miss each change.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
HEAD_SRC = REPO_ROOT / "src"

MUTATIONS = [
    (
        "1a_drop_int_game_pk_check",
        "src/bts/cli.py",
        'if result["state"] == "wrong_game" and type(slot["game_pk"]) is int',
        'if result["state"] == "wrong_game"',
    ),
    (
        "1b_allow_bool_game_pk_via_isinstance",
        "src/bts/cli.py",
        'type(slot["game_pk"]) is int',
        'isinstance(slot["game_pk"], int)',
    ),
    (
        "2_drop_shared_batter_condition",
        "src/bts/cli.py",
        'if len({slot["batter_id"] for slot in slots}) != len(slots):',
        "if False:",
    ),
    (
        "3a_drop_case_b",
        "src/bts/cli.py",
        'if not affected and prior_wrong_game and not qualification["all_confirmed"]:',
        "if False:",
    ),
    (
        "3b_drop_reason_from_prior_wrong_game",
        "src/bts/cli.py",
        'prior_wrong_game = prior.get("reason") == "wrong_game" or "wrong_game" in prior_escalations',
        'prior_wrong_game = "wrong_game" in prior_escalations',
    ),
    (
        "4_first_alert_consumes_initial",
        "src/bts/cli.py",
        '        if reason != "wrong_game":\n            consumed.add("initial")\n        elif tier == "wrong_game":\n            consumed.add("wrong_game")',
        '        consumed.add("initial")\n        if tier == "wrong_game":\n            consumed.add("wrong_game")',
    ),
    (
        "5a_confirm_when_any_leg_confirmed_case_a",
        "src/bts/cli.py",
        '                affected = [(slot, result) for slot, result in results\n                            if result["state"] == "wrong_game" and type(slot["game_pk"]) is int]',
        '                affected = ([] if any(result["state"] == "confirmed" for _, result in results) else\n                            [(slot, result) for slot, result in results\n                             if result["state"] == "wrong_game" and type(slot["game_pk"]) is int])',
    ),
    (
        "5b_confirm_when_any_leg_confirmed_case_b",
        "src/bts/cli.py",
        'if not affected and prior_wrong_game and not qualification["all_confirmed"]:',
        'if not affected and prior_wrong_game and not any(r["state"] == "confirmed" for _, r in results):',
    ),
    (
        "6_swap_first_and_reminder_texts",
        "src/bts/cli.py",
        '        elif tier == "wrong_game":\n            msg = (f"⚠️ BTS entry game needs verification. {legs}. "\n                   "Check the delivered pick and the game shown in MLB for each listed leg. "\n                   "Change only a leg whose correct game you can verify and that MLB still lets you edit. "\n                   "Keep any correct leg as entered.")\n        else:\n            msg = (f"⏰ BTS entry game verification is still needed. {legs}. "\n                   "Check the delivered pick against the game shown in MLB.")',
        '        elif tier != "wrong_game":\n            msg = (f"⚠️ BTS entry game needs verification. {legs}. "\n                   "Check the delivered pick and the game shown in MLB for each listed leg. "\n                   "Change only a leg whose correct game you can verify and that MLB still lets you edit. "\n                   "Keep any correct leg as entered.")\n        else:\n            msg = (f"⏰ BTS entry game verification is still needed. {legs}. "\n                   "Check the delivered pick against the game shown in MLB.")',
    ),
    (
        "7_drop_wrong_game_in_all_clear_guard",
        "src/bts/cli.py",
        'if dm_recipient and (was_alerted or "initial" in prior_escalations or "wrong_game" in prior_escalations):',
        'if dm_recipient and (was_alerted or "initial" in prior_escalations):',
    ),
    (
        "8_drop_legacy_wrong_game_exclusion",
        "src/bts/cli.py",
        'or (["initial"] if was_alerted and prior.get("reason") != "wrong_game" else [])',
        'or (["initial"] if was_alerted else [])',
    ),
]


def run_suite_against(scratch_src: Path) -> tuple[bool, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(scratch_src)
    env["TZ"] = "America/New_York"
    # Run fast non-FIFO tests first (~2.5s) across the F1 suite, receipt suite, and CLI CheckPickEntered suite
    cmd_fast = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--tb=no",
        "-k",
        "not test_fifo_capture_blocks_before_match_marker_only",
        "tests/test_wr_f1_wrong_game.py",
        "tests/test_entry_receipt.py",
        "tests/health/test_pick_entry_source.py",
        "tests/test_day_outcome.py",
        "tests/test_day_outcome_entry_sequence.py",
    ]
    res = subprocess.run(cmd_fast, cwd=REPO_ROOT, env=env, timeout=30, capture_output=True, text=True)
    if res.returncode != 0:
        last_line = [ln for ln in res.stdout.splitlines() if ln.strip()][-1] if res.stdout else "failed"
        return True, last_line

    # Also check TestCheckPickEntered in test_cli_integration.py (~2s)
    cmd_cli = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--tb=no",
        "-k",
        "TestCheckPickEntered",
        "tests/test_cli_integration.py",
    ]
    res2 = subprocess.run(cmd_cli, cwd=REPO_ROOT, env=env, timeout=30, capture_output=True, text=True)
    if res2.returncode != 0:
        last_line = [ln for ln in res2.stdout.splitlines() if ln.strip()][-1] if res2.stdout else "failed"
        return True, last_line
    return False, "SURVIVED (all tests passed)"


def main():
    results = []
    with TemporaryDirectory() as tmp:
        scratch_root = Path(tmp) / "scratch"
        for name, rel_path, target, replacement in MUTATIONS:
            if scratch_root.exists():
                shutil.rmtree(scratch_root)
            shutil.copytree(HEAD_SRC, scratch_root / "src")
            fpath = scratch_root / rel_path
            text = fpath.read_text()
            assert target in text, f"Target not found for {name}"
            assert text.count(target) == 1, f"Target matched multiple times for {name}"
            fpath.write_text(text.replace(target, replacement, 1))
            caught, detail = run_suite_against(scratch_root / "src")
            results.append({"mutation": name, "caught": caught, "detail": detail})

    survived = [r["mutation"] for r in results if not r["caught"]]
    summary = {
        "total_mutations": len(results),
        "caught_count": sum(1 for r in results if r["caught"]),
        "survived_count": len(survived),
        "survived": survived,
        "results": results,
    }
    (HERE / "probe_05_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
