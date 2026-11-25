"""Probe item 6 & C14: Run each of the 19 C14 scratch mutations plus 7 adversarial
mutations one at a time on an isolated scratch copy of src/, verifying bts.__file__
points to the scratch copy and checking whether tests notice every mutation.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


MUTATIONS: list[tuple[str, str, str, str, int]] = [
    # --- 16 scratch changes from round 2 ---
    (
        "r2_01_top_prediction",
        "scheduler.py",
        """            summary.update(
                best_batter=name if isinstance(name, str) else "?",
                best_team=team if isinstance(team, str) else "?",
                best_p=probability,
                candidate_verified=True,
            )""",
        '            summary["candidate_verified"] = True',
        0,
    ),
    (
        "r2_02_below_equality",
        "scheduler.py",
        "if probability < bar:",
        "if probability <= bar:",
        0,
    ),
    (
        "r2_03_drop_label_k2",
        "scheduler.py",
        'sentence = f"{description}, below {bar_text} using unrounded values."',
        'sentence = f"{description}, below {bar_text} using unrounded values.".replace(" (eligibility unverified)", "")',
        0,
    ),
    (
        "r2_04_swap_k1_k5",
        "scheduler.py",
        "    if reason:\n        ending",
        '    if reason and fields.get("basis") != "no_decision":\n        ending',
        0,
    ),
    (
        "r2_05_swap_k1_k2",
        "scheduler.py",
        "    if reason:\n        ending",
        "    if reason and not (probability is not None and bar is not None and probability < bar):\n        ending",
        0,
    ),
    (
        "r2_06_old_journal_k1",
        "scheduler.py",
        '                print(f"  SKIP — {format_skip_summary(skip_summary)}", file=sys.stderr)',
        '                body = format_skip_summary(skip_summary)\n'
        '                if ". Reason:" in body:\n'
        '                    body += " Will pick if a confirmed lineup clears it."\n'
        '                print(f"  SKIP — {body}", file=sys.stderr)',
        0,
    ),
    (
        "r2_07_old_journal_k3",
        "scheduler.py",
        '                print(f"  SKIP — {format_skip_summary(skip_summary)}", file=sys.stderr)',
        '                body = format_skip_summary(skip_summary)\n'
        '                if ", at or above " in body:\n'
        '                    body += " Will pick if a confirmed lineup clears it."\n'
        '                print(f"  SKIP — {body}", file=sys.stderr)',
        0,
    ),
    (
        "r2_08_old_journal_k4",
        "scheduler.py",
        '                print(f"  SKIP — {format_skip_summary(skip_summary)}", file=sys.stderr)',
        '                body = format_skip_summary(skip_summary)\n'
        '                if "No probability comparison" in body:\n'
        '                    body += " Will pick if a confirmed lineup clears it."\n'
        '                print(f"  SKIP — {body}", file=sys.stderr)',
        0,
    ),
    (
        "r2_09_unescaped_name",
        "web.py",
        "    body = html_lib.escape(format_skip_summary(summary))",
        "    body = html_lib.escape(format_skip_summary(summary))\n"
        "    raw = summary.get('best_batter')\n"
        "    if isinstance(raw, str) and raw:\n"
        "        norm = ' '.join(raw.splitlines())\n"
        "        body = body.replace(html_lib.escape(norm), norm, 1)",
        0,
    ),
    (
        "r2_10_unescaped_team",
        "web.py",
        "    body = html_lib.escape(format_skip_summary(summary))",
        "    body = html_lib.escape(format_skip_summary(summary))\n"
        "    raw = summary.get('best_team')\n"
        "    if isinstance(raw, str) and raw:\n"
        "        norm = ' '.join(raw.splitlines())\n"
        "        body = body.replace(html_lib.escape(norm), norm, 1)",
        0,
    ),
    (
        "r2_11_unescaped_reason",
        "web.py",
        "    body = html_lib.escape(format_skip_summary(summary))",
        "    body = html_lib.escape(format_skip_summary(summary))\n"
        "    raw = summary.get('reason')\n"
        "    if isinstance(raw, str) and raw:\n"
        "        norm = ' '.join(raw.splitlines())\n"
        "        body = body.replace(html_lib.escape(norm), norm, 1)",
        0,
    ),
    (
        "r2_12_drop_delivery_saver",
        "scheduler.py",
        '"saver used, check bts saver-state --picks-dir DIR. For active or not_earned, run bts "\n'
        '            "saver-state --use --picks-dir DIR. For uninitialized, see B8a. Check the latest fetch status."',
        '"saver used, check bts saver-state --picks-dir DIR. For uninitialized, see B8a. Check the latest fetch status."',
        0,
    ),
    (
        "r2_13_drop_morning_saver",
        "scheduler.py",
        '"saver-state --picks-dir DIR. For active or not_earned, run bts saver-state --use --picks-dir "\n'
        '            "DIR. For uninitialized, see B8a. Check the latest fetch status."',
        '"saver-state --picks-dir DIR. For uninitialized, see B8a. Check the latest fetch status."',
        0,
    ),
    (
        "r2_14_old_refusal_late",
        "scheduler.py",
        '"The entry check may report a mismatch if it verifies the entry in time.")',
        '"Enter a pick manually if you want one now.")',
        0,
    ),
    (
        "r2_15_old_refusal_void",
        "scheduler.py",
        '"The entry check may report a mismatch if it verifies the entry in time.")',
        '"Enter a pick manually if you want one today.")',
        1,
    ),
    (
        "r2_16_string_probability",
        "scheduler.py",
        '    probability = _skip_display_number(fields.get("best_p"))',
        '    if isinstance(fields.get("best_p"), str):\n'
        '        raise TypeError("string probability")\n'
        '    probability = _skip_display_number(fields.get("best_p"))',
        0,
    ),
    # --- 3 Ea-specific mutations from C14 ---
    (
        "ea_17_restore_no_pick_at_last_check",
        "web.py",
        "LAST RECORDED NO-PICK SUMMARY",
        "NO PICK AT LAST CHECK",
        0,
    ),
    (
        "ea_18_restore_follow_b8b1",
        "scheduler.py",
        "see B8a. On",
        "follow B8b(1). On",
        0,
    ),
    (
        "ea_19_drop_before_first_check",
        "scheduler.py",
        "Before the first check, if prior entries are settled or absent",
        "If prior entries are settled or absent",
        0,
    ),
    # --- 7 additional adversarial mutations of our own ---
    (
        "own_20_isinstance_outside_try",
        "scheduler.py",
        "    if decision is not None:\n        try:\n            if not isinstance(candidate, Mapping):\n                return summary",
        "    if decision is not None:\n        if not isinstance(candidate, Mapping):\n            return summary\n        try:",
        0,
    ),
    (
        "own_21_allow_out_of_range_numbers",
        "scheduler.py",
        "return number if 0 <= number <= 1 else None",
        "return number if math.isfinite(number) else None",
        0,
    ),
    (
        "own_22_reject_zero_and_one_edges",
        "scheduler.py",
        "return number if 0 <= number <= 1 else None",
        "return number if 0 < number < 1 else None",
        0,
    ),
    (
        "own_23_no_strip_blank_reason",
        "scheduler.py",
        'reason = display_string(raw_reason).strip() if isinstance(raw_reason, str) else ""',
        'reason = display_string(raw_reason) if isinstance(raw_reason, str) else ""',
        0,
    ),
    (
        "own_24_truthy_candidate_verified",
        "scheduler.py",
        'if fields.get("candidate_verified") is True:',
        'if fields.get("candidate_verified"):',
        0,
    ),
    (
        "own_25_bool_streak_accepted",
        "scheduler.py",
        "if type(streak) is not int or streak < 0:",
        "if not isinstance(streak, int) or streak < 0:",
        0,
    ),
    (
        "own_26_drop_settled_condition",
        "scheduler.py",
        "if prior entries are settled or absent ",
        "if ",
        0,
    ),
]


RUNNER_CODE = """
import os, sys
from pathlib import Path
scratch = Path(os.environ["SCRATCH_SRC"]).resolve()
sys.path.insert(0, str(scratch))
import bts
actual = Path(bts.__file__).resolve()
assert actual.is_relative_to(scratch), f"Expected scratch bts, got {actual}"
import pytest
raise SystemExit(pytest.main(["-q", "-x", "--no-header", "tests/test_wr_e_texts.py"]))
"""


def main() -> None:
    outcomes = []
    survivors = []
    for name, rel_file, old, new, occurrence in MUTATIONS:
        with tempfile.TemporaryDirectory(dir=HERE, prefix=f"mut-{name}-") as tmp:
            scratch_src = Path(tmp) / "src"
            shutil.copytree(
                REPO / "src/bts",
                scratch_src / "bts",
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            target = scratch_src / "bts" / rel_file
            content = target.read_text()
            parts = content.split(old)
            assert len(parts) > occurrence + 1, f"Pattern not found for {name} in {rel_file}"
            rebuilt = old.join(parts[:occurrence + 1]) + new + old.join(parts[occurrence + 1:])
            target.write_text(rebuilt)

            env = dict(os.environ)
            env["SCRATCH_SRC"] = str(scratch_src)
            env["PYTHONPATH"] = f"{scratch_src}{os.pathsep}{REPO}"
            res = subprocess.run(
                [sys.executable, "-c", RUNNER_CODE],
                cwd=REPO,
                env=env,
                capture_output=True,
                text=True,
                timeout=45,
            )
            failed_line = next(
                (ln for ln in res.stdout.splitlines() if ln.startswith("FAILED ")),
                "",
            )
            detected = res.returncode == 1 and bool(failed_line) and "ERROR " not in res.stdout
            rec = {
                "mutation": name,
                "file": rel_file,
                "returncode": res.returncode,
                "detected": detected,
                "failed_test": failed_line,
            }
            outcomes.append(rec)
            if not detected:
                survivors.append(rec)
            print(f"{name}: {'DETECTED' if detected else 'SURVIVED'} ({failed_line})", flush=True)

    summary = {
        "total_mutations": len(MUTATIONS),
        "c14_count": 19,
        "own_count": len(MUTATIONS) - 19,
        "detected_count": len(outcomes) - len(survivors),
        "survivors": survivors,
        "outcomes": outcomes,
    }
    (HERE / "probe_mutations_summary.json").write_text(json.dumps(summary, indent=2))
    assert not survivors, f"Surviving mutations: {survivors}"
    print(f"probe_mutations: ALL {len(MUTATIONS)} MUTATIONS DETECTED")


if __name__ == "__main__":
    main()
