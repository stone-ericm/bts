"""Probe 6b: Mutation testing on a scratch copy of `src` in `.carbon-review/wr-e/mut_src`
against the 11-target test suite (excluding only the single sandbox-blocked loopback socket test).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
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


def reset_scratch() -> None:
    if MUT_ROOT.exists():
        shutil.rmtree(MUT_ROOT)
    shutil.copytree(REPO / "src", MUT_ROOT / "src")


def run_pytest(extra_src: Path | None, fast_skip_bars: bool = False) -> tuple[int, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{extra_src}:{REPO}" if extra_src else str(REPO)
    k_expr = "not test_do_post_saver_transition_socket_wiring"
    if fast_skip_bars:
        k_expr += " and not test_shipped_bars_equal_base"
    cmd = [
        str(REPO / ".venv/bin/python"), "-m", "pytest", *SUITE_TARGETS,
        "-k", k_expr, "-x", "-q", "--tb=line",
    ]
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=180)
    return proc.returncode, (proc.stdout + "\n" + proc.stderr).strip()


def apply_replace(rel_path: str, old: str, new: str) -> None:
    p = MUT_ROOT / "src" / rel_path
    txt = p.read_text()
    assert txt.count(old) == 1, f"Anchor count {txt.count(old)} != 1 in {rel_path}"
    p.write_text(txt.replace(old, new, 1))


def main() -> None:
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
            "3a_drop_eligibility_wording_all_unverified",
            "bts/scheduler.py",
            '    label = "Top model prediction (eligibility unverified)"',
            '    label = "Top model prediction"',
        ),
        (
            "3b_drop_eligibility_wording_in_k2_only",
            "bts/scheduler.py",
            '        if probability < bar:\n            sentence = f"{description}, below {bar_text} using unrounded values."',
            '        if probability < bar:\n            sentence = f"{description.replace(\' (eligibility unverified)\', \'\')}, below {bar_text} using unrounded values."',
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
            "5b_leave_old_sentence_in_journal_surface_k1",
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
            "5c_leave_old_sentence_in_journal_surface_k3_k4",
            "bts/scheduler.py",
            '            if skip_summary is not None:\n'
            '                print(f"  SKIP — {format_skip_summary(skip_summary)}", file=sys.stderr)',
            '            if skip_summary is not None:\n'
            '                if skip_summary.get("basis") == "decision" and not skip_summary.get("reason") and (skip_summary.get("bar") is None or skip_summary["best_p"] >= skip_summary["bar"]):\n'
            '                    print(f"  SKIP — best {skip_summary[\'best_batter\']} ({skip_summary[\'best_team\']}) {skip_summary[\'best_p\']:.1%}; streak holds at {skip_summary[\'streak\']}.", file=sys.stderr)\n'
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
            "6b_drop_banner_escaping_of_best_team",
            "bts/web.py",
            "    body = html_lib.escape(format_skip_summary(summary))",
            '    raw_team = summary.get("best_team") if isinstance(summary.get("best_team"), str) else None\n'
            '    s_copy = dict(summary, best_team="__TEAM_TOKEN__") if raw_team is not None else summary\n'
            '    body = html_lib.escape(format_skip_summary(s_copy))\n'
            '    if raw_team is not None:\n'
            '        body = body.replace("__TEAM_TOKEN__", " ".join(raw_team.splitlines()))',
        ),
        (
            "6c_drop_banner_escaping_of_reason",
            "bts/web.py",
            "    body = html_lib.escape(format_skip_summary(summary))",
            '    raw_rsn = summary.get("reason") if isinstance(summary.get("reason"), str) and summary.get("reason") else None\n'
            '    s_copy = dict(summary, reason="__RSN_TOKEN__.") if raw_rsn is not None else summary\n'
            '    body = html_lib.escape(format_skip_summary(s_copy))\n'
            '    if raw_rsn is not None:\n'
            '        clean_r = " ".join(raw_rsn.splitlines())\n'
            '        end_r = "" if clean_r.endswith((".", "!", "?")) else "."\n'
            '        body = body.replace("__RSN_TOKEN__.", clean_r + end_r)',
        ),
        (
            "7a_drop_saver_sentence_from_morning_warning",
            "bts/scheduler.py",
            '            f"unchanged. On {date} ET away from midnight, if MLB shows this season\'s saver used, check bts "\n'
            '            "saver-state --picks-dir DIR. For active or not_earned, run bts saver-state --use --picks-dir "\n'
            '            "DIR. For uninitialized, see B8a. Check the latest fetch status."',
            '            "unchanged. Check the latest fetch status."',
        ),
        (
            "7b_drop_saver_sentence_from_delivery_warning",
            "bts/scheduler.py",
            '            f"follow B8b(1). On {warning_day.isoformat()} ET away from midnight, if MLB shows this season\'s "\n'
            '            "saver used, check bts saver-state --picks-dir DIR. For active or not_earned, run bts "\n'
            '            "saver-state --use --picks-dir DIR. For uninitialized, see B8a. Check the latest fetch status."',
            '            "follow B8b(1). Check the latest fetch status."',
        ),
        (
            "8a_restore_old_last_sentence_of_late_refusal",
            "bts/scheduler.py",
            '           "A manual entry can differ from a later recommendation. "\n'
            '           "The entry check may report a mismatch if it verifies the entry in time.")',
            '           "A manual entry can differ from a later recommendation. "\n'
            '           "Enter a pick manually if you want one now.")',
        ),
        (
            "8b_restore_old_last_sentence_of_postponed_refusal",
            "bts/scheduler.py",
            '    msg = (f"DELIVERY REFUSED for {date}: {games}. "\n'
            '           "This pick was withheld. A later check may produce another recommendation. "\n'
            '           "Use a later official pick message only while the entry window is open. "\n'
            '           "A manual entry can differ from a later recommendation. "\n'
            '           "The entry check may report a mismatch if it verifies the entry in time.")',
            '    msg = (f"DELIVERY REFUSED for {date}: {games}. "\n'
            '           "This pick was withheld. A later check may produce another recommendation. "\n'
            '           "Use a later official pick message only while the entry window is open. "\n'
            '           "A manual entry can differ from a later recommendation. "\n'
            '           "Enter a pick manually if you want one today.")',
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
        rc, out = run_pytest(MUT_ROOT / "src", fast_skip_bars=True)
        if rc == 0:
            # Verify whether even test_shipped_bars_equal_base catches it
            rc, out = run_pytest(MUT_ROOT / "src", fast_skip_bars=False)
        killed = rc != 0
        lines = [ln for ln in out.splitlines() if ln.strip()]
        fail_line = next((ln for ln in lines if "FAILED" in ln or "AssertionError" in ln or "ValueError" in ln), lines[-1] if lines else "")
        results.append({"mutation": name, "killed": killed, "detail": fail_line[:220]})
        print(f"[{'KILLED' if killed else 'SURVIVED'}] {name} -> {fail_line[:130]}", flush=True)

    shutil.rmtree(MUT_ROOT, ignore_errors=True)
    out_file = REPO / ".carbon-review/wr-e/mutation_results.json"
    out_file.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
