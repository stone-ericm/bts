"""Probe 6: Run the required 10-target test suite in one process on HEAD, then mutate a scratch
copy of `src` in `.carbon-review/wr-e/mut_src` in each of the 9 ways named in brief item 6
(without ever modifying tracked files) and record which mutations are caught vs unnoticed.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MUT_ROOT = REPO / ".carbon-review/wr-e/mut_src"

SUITE_TARGETS = [
    "tests/test_wr_e_texts.py",
    "tests/test_scheduler_skip_visibility.py",
    "tests/test_scheduler_tail_objective.py",
    "tests/test_stale_delivery_cs4c.py",
    "tests/test_morning_state_cs6b1.py",
    "tests/test_late_delivery_guard.py",
    "tests/test_postponed_delivery_guard.py",
    "tests/test_wr_b_contract.py",
    "tests/scripts/season_walk/",
    "tests/test_lightgbm_optional.py",
    "tests/test_web_saver.py",
]

FOCUSED_MUT_TARGETS = [
    "tests/test_wr_e_texts.py",
    "tests/test_scheduler_skip_visibility.py",
    "tests/test_scheduler_tail_objective.py",
    "tests/test_stale_delivery_cs4c.py",
    "tests/test_morning_state_cs6b1.py",
    "tests/test_late_delivery_guard.py",
    "tests/test_postponed_delivery_guard.py",
    "tests/scripts/season_walk/test_season_walk.py",
]


def reset_scratch() -> None:
    if MUT_ROOT.exists():
        shutil.rmtree(MUT_ROOT)
    shutil.copytree(REPO / "src", MUT_ROOT / "src")


def run_pytest(targets: list[str], extra_src: Path | None = None, stop_first: bool = False) -> tuple[int, str]:
    env = os.environ.copy()
    if extra_src is not None:
        env["PYTHONPATH"] = f"{extra_src}:{REPO}"
    else:
        env["PYTHONPATH"] = str(REPO)
    cmd = [str(REPO / ".venv/bin/python"), "-m", "pytest", *targets, "-q", "--tb=line"]
    if stop_first:
        cmd.append("-x")
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=180)
    return proc.returncode, (proc.stdout + "\n" + proc.stderr).strip()


def apply_replace(rel_path: str, old: str, new: str) -> None:
    p = MUT_ROOT / "src" / rel_path
    txt = p.read_text()
    assert old in txt, f"Anchor not found in {rel_path}: {old!r}"
    assert txt.count(old) == 1, f"Anchor not unique in {rel_path}: {old!r}"
    p.write_text(txt.replace(old, new, 1))


def main() -> None:
    # 1. Run full required combined suite in one process on HEAD
    rc_head, out_head = run_pytest(SUITE_TARGETS, extra_src=None, stop_first=False)
    head_summary = out_head.splitlines()[-1] if out_head else ""
    print("Combined HEAD suite rc:", rc_head, "|", head_summary)
    assert rc_head == 0, f"Combined HEAD suite failed:\n{out_head}"

    # 2. Define small mutations on scratch copy covering all 9 classes from brief item 6
    mutations = [
        (
            "1_highest_prediction_instead_of_candidate",
            "bts/scheduler.py",
            "    if decision is not None and isinstance(candidate, Mapping):",
            "    if False and decision is not None and isinstance(candidate, Mapping):",
        ),
        (
            "2_say_below_when_at_bar",
            "bts/scheduler.py",
            "        if probability < bar:",
            "        if probability <= bar:",
        ),
        (
            "3a_drop_eligibility_wording_global",
            "bts/scheduler.py",
            '    label = "Top model prediction (eligibility unverified)"',
            '    label = "Top model prediction"',
        ),
        (
            "3b_drop_eligibility_wording_k1_to_k4_only",
            "bts/scheduler.py",
            "    else:\n        description = f\"{label}: {candidate}\"",
            "    else:\n        description = f\"Top model prediction: {candidate}\"",
        ),
        (
            "4a_swap_k5_and_k2k3_order",
            "bts/scheduler.py",
            '    elif fields.get("basis") == "no_decision":\n'
            '        sentence = f"No candidate was chosen. {label}: {candidate}."\n'
            '    elif bar is not None and probability is not None:\n'
            '        bar_label = f"the streak-{streak} bar" if streak is not None else "the bar"\n'
            '        bar_text = f"{bar_label} ({bar:.1%})"\n'
            '        if probability < bar:\n'
            '            sentence = f"{description}, below {bar_text} using unrounded values."\n'
            '        else:\n'
            '            sentence = f"{description}, at or above {bar_text}."',
            '    elif bar is not None and probability is not None:\n'
            '        bar_label = f"the streak-{streak} bar" if streak is not None else "the bar"\n'
            '        bar_text = f"{bar_label} ({bar:.1%})"\n'
            '        if probability < bar:\n'
            '            sentence = f"{description}, below {bar_text} using unrounded values."\n'
            '        else:\n'
            '            sentence = f"{description}, at or above {bar_text}."\n'
            '    elif fields.get("basis") == "no_decision":\n'
            '        sentence = f"No candidate was chosen. {label}: {candidate}."',
        ),
        (
            "4b_swap_k1_and_k5_order",
            "bts/scheduler.py",
            '    if isinstance(reason, str) and reason:\n'
            '        reason = display_string(reason)\n'
            '        ending = "" if reason.endswith((".", "!", "?")) else "."\n'
            '        sentence = f"{description}. Reason: {reason}{ending}"\n'
            '    elif fields.get("basis") == "no_decision":\n'
            '        sentence = f"No candidate was chosen. {label}: {candidate}."',
            '    if fields.get("basis") == "no_decision":\n'
            '        sentence = f"No candidate was chosen. {label}: {candidate}."\n'
            '    elif isinstance(reason, str) and reason:\n'
            '        reason = display_string(reason)\n'
            '        ending = "" if reason.endswith((".", "!", "?")) else "."\n'
            '        sentence = f"{description}. Reason: {reason}{ending}"',
        ),
        (
            "4c_swap_k1_and_k2k3_order",
            "bts/scheduler.py",
            '    if isinstance(reason, str) and reason:\n'
            '        reason = display_string(reason)\n'
            '        ending = "" if reason.endswith((".", "!", "?")) else "."\n'
            '        sentence = f"{description}. Reason: {reason}{ending}"\n'
            '    elif fields.get("basis") == "no_decision":\n'
            '        sentence = f"No candidate was chosen. {label}: {candidate}."\n'
            '    elif bar is not None and probability is not None:\n'
            '        bar_label = f"the streak-{streak} bar" if streak is not None else "the bar"\n'
            '        bar_text = f"{bar_label} ({bar:.1%})"\n'
            '        if probability < bar:\n'
            '            sentence = f"{description}, below {bar_text} using unrounded values."\n'
            '        else:\n'
            '            sentence = f"{description}, at or above {bar_text}."',
            '    if bar is not None and probability is not None and fields.get("basis") != "no_decision":\n'
            '        bar_label = f"the streak-{streak} bar" if streak is not None else "the bar"\n'
            '        bar_text = f"{bar_label} ({bar:.1%})"\n'
            '        if probability < bar:\n'
            '            sentence = f"{description}, below {bar_text} using unrounded values."\n'
            '        else:\n'
            '            sentence = f"{description}, at or above {bar_text}."\n'
            '    elif fields.get("basis") == "no_decision":\n'
            '        sentence = f"No candidate was chosen. {label}: {candidate}."\n'
            '    elif isinstance(reason, str) and reason:\n'
            '        reason = display_string(reason)\n'
            '        ending = "" if reason.endswith((".", "!", "?")) else "."\n'
            '        sentence = f"{description}. Reason: {reason}{ending}"',
        ),
        (
            "5a_leave_old_sentence_in_dm_surface_k4",
            "bts/scheduler.py",
            'def format_skip_dm(date: str, summary: object) -> str:\n'
            '    return f"BTS {date}: No pick at this check. {format_skip_summary(summary)}"',
            'def format_skip_dm(date: str, summary: object) -> str:\n'
            '    if isinstance(summary, dict) and not summary.get("reason") and summary.get("basis") != "no_decision" and summary.get("bar") is None:\n'
            '        return f"BTS {date}: No pick yet — model\'s best is {summary.get(\'best_batter\')} ({summary.get(\'best_team\')})."\n'
            '    return f"BTS {date}: No pick at this check. {format_skip_summary(summary)}"',
        ),
        (
            "5b_leave_old_sentence_in_journal_surface_k1_k3_k4",
            "bts/scheduler.py",
            '            if skip_summary is not None:\n'
            '                print(f"  SKIP — {format_skip_summary(skip_summary)}", file=sys.stderr)',
            '            if skip_summary is not None:\n'
            '                if skip_summary.get("reason"):\n'
            '                    print(f"  SKIP — best {skip_summary[\'best_batter\']} ({skip_summary[\'best_team\']}) {skip_summary[\'best_p\']:.1%} {skip_summary[\'reason\']}; streak holds at {skip_summary[\'streak\']}.", file=sys.stderr)\n'
            '                else:\n'
            '                    print(f"  SKIP — {format_skip_summary(skip_summary)}", file=sys.stderr)',
        ),
        (
            "6a_drop_banner_escaping_entirely",
            "bts/web.py",
            "    body = html_lib.escape(format_skip_summary(summary))",
            "    body = format_skip_summary(summary)",
        ),
        (
            "6b_drop_banner_escaping_of_team_or_reason_value",
            "bts/web.py",
            "    body = html_lib.escape(format_skip_summary(summary))",
            '    raw_team = str(summary.get("best_team")) if isinstance(summary.get("best_team"), str) else None\n'
            '    s_copy = dict(summary, best_team="__TEAM_TOKEN__") if raw_team is not None else summary\n'
            '    body = html_lib.escape(format_skip_summary(s_copy))\n'
            '    if raw_team is not None:\n'
            '        body = body.replace("__TEAM_TOKEN__", " ".join(raw_team.splitlines()))',
        ),
        (
            "7_drop_saver_sentence_from_morning_warning",
            "bts/scheduler.py",
            '            f"unchanged. On {date} ET away from midnight, if MLB shows this season\'s saver used, check bts "\n'
            '            "saver-state --picks-dir DIR. For active or not_earned, run bts saver-state --use --picks-dir "\n'
            '            "DIR. For uninitialized, see B8a. Check the latest fetch status."',
            '            "unchanged. Check the latest fetch status."',
        ),
        (
            "8a_restore_old_last_sentence_of_late_refusal_with_scan_phrase",
            "bts/scheduler.py",
            '           "A manual entry can differ from a later recommendation. "\n'
            '           "The entry check may report a mismatch if it verifies the entry in time.")',
            '           "A manual entry can differ from a later recommendation. "\n'
            '           "Enter a pick manually if you want one now.")',
        ),
        (
            "9a_renderer_raises_on_numeric_string_probability",
            "bts/scheduler.py",
            "    if type(value) not in (int, float):\n        return None",
            "    if isinstance(value, str):\n        raise ValueError('string prob')\n    if type(value) not in (int, float):\n        return None",
        ),
        (
            "9b_renderer_accepts_numeric_string_raises_on_non_numeric_string",
            "bts/scheduler.py",
            "    if type(value) not in (int, float):\n        return None",
            "    if isinstance(value, str):\n        return float(value)\n    if type(value) not in (int, float):\n        return None",
        ),
    ]

    results = []
    for name, rel_path, old, new in mutations:
        reset_scratch()
        apply_replace(rel_path, old, new)
        # First test against focused suite with -x for speed; if it survives focused, test against full 11-target suite!
        rc, out = run_pytest(FOCUSED_MUT_TARGETS, extra_src=MUT_ROOT / "src", stop_first=True)
        if rc == 0:
            rc, out = run_pytest(SUITE_TARGETS, extra_src=MUT_ROOT / "src", stop_first=True)
        killed = rc != 0
        fail_line = next((ln for ln in out.splitlines() if "FAILED" in ln or "AssertionError" in ln or "Error" in ln), out.splitlines()[-1] if out else "")
        results.append({"mutation": name, "killed": killed, "detail": fail_line[:220]})
        print(f"  [{ 'KILLED' if killed else 'SURVIVED' }] {name}: {fail_line[:140]}")

    shutil.rmtree(MUT_ROOT, ignore_errors=True)
    out_file = REPO / ".carbon-review/wr-e/mutation_results.json"
    out_file.write_text(json.dumps({"head_summary": head_summary, "mutations": results}, indent=2))


if __name__ == "__main__":
    main()
