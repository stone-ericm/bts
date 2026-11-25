"""Probe for Item 6 (The tests and scratch-copy mutations):
Usage:
  --baseline   Run the 11 named test targets in one process + audit test diffs against 63669c42
  --batch N    Run mutation batch N (0..5, 3 mutations per batch so each command finishes well under 2 min)
  --summarize  Aggregate baseline + all batch JSONs into probe_6_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT_DIR = Path(__file__).resolve().parent
SCRATCH_PKG = OUT_DIR / "scratch_pkg"

TARGETS = [
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


def _parse_pytest_counts(output: str) -> dict:
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", output)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", output)) else 0
    errors = int(m.group(1)) if (m := re.search(r"(\d+) error", output)) else 0
    deselected = int(m.group(1)) if (m := re.search(r"(\d+) deselected", output)) else 0
    return {"passed": passed, "failed": failed, "errors": errors, "deselected": deselected}


def run_pytest(extra_args: list[str], *, pythonpath: str | None = None, timeout: int = 100) -> dict:
    env = os.environ.copy()
    env["TZ"] = "America/New_York"
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    cmd = [str(REPO / ".venv/bin/python"), "-m", "pytest", "-q", *extra_args, *TARGETS]
    res = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)
    counts = _parse_pytest_counts(res.stdout + "\n" + res.stderr)
    failed_lines = [
        ln for ln in (res.stdout + "\n" + res.stderr).splitlines()
        if ln.startswith("FAILED ") or "PermissionError" in ln
    ]
    return {
        "returncode": res.returncode,
        "counts": counts,
        "failed_lines": failed_lines[:20],
    }


def audit_existing_test_diffs() -> dict:
    existing_files = [
        "tests/test_scheduler_skip_visibility.py",
        "tests/test_scheduler_tail_objective.py",
        "tests/test_stale_delivery_cs4c.py",
        "tests/test_morning_state_cs6b1.py",
        "tests/scripts/season_walk/test_season_walk.py",
    ]
    per_file = {}
    for rel in existing_files:
        diff = subprocess.run(
            ["git", "diff", "-U0", "63669c42", "HEAD", "--", rel],
            cwd=REPO, check=True, capture_output=True, text=True, timeout=10,
        ).stdout
        removed = [ln[1:] for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---")]
        added = [ln[1:] for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
        per_file[rel] = {"removed_count": len(removed), "added_count": len(added), "removed": removed, "added": added}

    # Check no other existing test file changed
    all_test_names = subprocess.run(
        ["git", "diff", "--name-only", "63669c42", "HEAD", "--", "tests/"],
        cwd=REPO, check=True, capture_output=True, text=True, timeout=10,
    ).stdout.splitlines()
    return {"all_changed_test_files": all_test_names, "existing_diffs": per_file}


# All 18 scratch mutations (9 required categories + 9 narrow edge variants)
MUTATIONS: list[tuple[str, str, str, str, str]] = [
    # 1. Highest prediction instead of decided candidate
    (
        "M1_highest_prediction",
        "scheduler.py",
        "    if decision is not None and isinstance(candidate, Mapping):",
        "    if False and decision is not None and isinstance(candidate, Mapping):",
        "Use highest prediction instead of decided candidate",
    ),
    # 2. Say below when candidate is at the bar
    (
        "M2_below_at_bar",
        "scheduler.py",
        "        if probability < bar:",
        "        if probability <= bar:",
        "Say 'below' when candidate is at the bar (probability == bar)",
    ),
    # 3a. Drop unverified eligibility wording globally
    (
        "M3a_drop_unverified_global",
        "scheduler.py",
        '    label = "Top model prediction (eligibility unverified)"',
        '    label = "Top model prediction"',
        "Drop '(eligibility unverified)' globally",
    ),
    # 3b. Drop unverified eligibility wording ONLY in K1 (reason set, candidate_verified False)
    (
        "M3b_drop_unverified_k1_only",
        "scheduler.py",
        '        sentence = f"{description}. Reason: {reason}{ending}"',
        '        sentence = f"Candidate used in the decision: {candidate}. Reason: {reason}{ending}"',
        "Drop unverified eligibility wording in K1 (reason branch) only",
    ),
    # 4a. Swap K1 and K5 order
    (
        "M4a_swap_k1_k5",
        "scheduler.py",
        '    if reason:\n        ending = "" if reason.endswith((".", "!", "?")) else "."\n        sentence = f"{description}. Reason: {reason}{ending}"\n    elif fields.get("basis") == "no_decision":\n        sentence = f"No candidate was chosen. {label}: {candidate}."',
        '    if fields.get("basis") == "no_decision":\n        sentence = f"No candidate was chosen. {label}: {candidate}."\n    elif reason:\n        ending = "" if reason.endswith((".", "!", "?")) else "."\n        sentence = f"{description}. Reason: {reason}{ending}"',
        "Swap K1 and K5 precedence",
    ),
    # 4b. Swap K1 and K2/K3 order
    (
        "M4b_swap_k1_k2k3",
        "scheduler.py",
        '    if reason:\n        ending = "" if reason.endswith((".", "!", "?")) else "."\n        sentence = f"{description}. Reason: {reason}{ending}"\n    elif fields.get("basis") == "no_decision":\n        sentence = f"No candidate was chosen. {label}: {candidate}."\n    elif bar is not None and probability is not None:',
        '    if fields.get("basis") == "no_decision":\n        sentence = f"No candidate was chosen. {label}: {candidate}."\n    elif bar is not None and probability is not None:',
        "Drop K1 precedence before K2/K3 (move K1 to fallback)",
    ),
    # 4c. Swap K5 and K2/K3 order
    (
        "M4c_swap_k5_k2k3",
        "scheduler.py",
        '    elif fields.get("basis") == "no_decision":\n        sentence = f"No candidate was chosen. {label}: {candidate}."\n    elif bar is not None and probability is not None:\n        bar_label = f"the streak-{streak} bar" if streak is not None else "the bar"\n        bar_text = f"{bar_label} ({bar:.1%})"\n        if probability < bar:\n            sentence = f"{description}, below {bar_text} using unrounded values."\n        else:\n            sentence = f"{description}, at or above {bar_text}."',
        '    elif bar is not None and probability is not None:\n        bar_label = f"the streak-{streak} bar" if streak is not None else "the bar"\n        bar_text = f"{bar_label} ({bar:.1%})"\n        if probability < bar:\n            sentence = f"{description}, below {bar_text} using unrounded values."\n        else:\n            sentence = f"{description}, at or above {bar_text}."\n    elif fields.get("basis") == "no_decision":\n        sentence = f"No candidate was chosen. {label}: {candidate}."',
        "Swap K5 and K2/K3 precedence",
    ),
    # 5a. Leave old sentence in K3 journal surface
    (
        "M5a_old_k3_journal",
        "scheduler.py",
        '                print(f"  SKIP — {format_skip_summary(skip_summary)}", file=sys.stderr)',
        '                print(f"  SKIP — best {skip_summary[\'best_batter\']} ({skip_summary[\'best_team\']}) below the pick bar; streak holds at {skip_summary[\'streak\']}." if (skip_summary.get("bar") is not None and not skip_summary.get("reason") and skip_summary.get("best_p", 0) >= skip_summary["bar"]) else f"  SKIP — {format_skip_summary(skip_summary)}", file=sys.stderr)',
        "Leave old SKIP sentence on K3 journal surface only",
    ),
    # 5b. Leave old sentence in K4 banner surface
    (
        "M5b_old_k4_banner",
        "web.py",
        '    body = html_lib.escape(format_skip_summary(summary))',
        '    body = "No pick — below the pick bar." if (summary.get("bar") is None and not summary.get("reason") and summary.get("basis") != "no_decision") else html_lib.escape(format_skip_summary(summary))',
        "Leave old sentence in K4 banner surface only",
    ),
    # 6a. Drop HTML escaping of best_batter in banner
    (
        "M6a_unescape_banner_name",
        "web.py",
        '    body = html_lib.escape(format_skip_summary(summary))',
        '    raw_n = str(summary.get("best_batter", "")); body = html_lib.escape(format_skip_summary(summary)).replace(html_lib.escape(" ".join(raw_n.splitlines())), " ".join(raw_n.splitlines())) if isinstance(summary.get("best_batter"), str) else html_lib.escape(format_skip_summary(summary))',
        "Drop HTML escaping of best_batter in banner",
    ),
    # 6b. Drop HTML escaping of best_team in banner
    (
        "M6b_unescape_banner_team",
        "web.py",
        '    body = html_lib.escape(format_skip_summary(summary))',
        '    raw_t = str(summary.get("best_team", "")); body = html_lib.escape(format_skip_summary(summary)).replace(html_lib.escape(" ".join(raw_t.splitlines())), " ".join(raw_t.splitlines())) if isinstance(summary.get("best_team"), str) else html_lib.escape(format_skip_summary(summary))',
        "Drop HTML escaping of best_team in banner",
    ),
    # 6c. Drop HTML quote escaping in banner (quote=False)
    (
        "M6c_unescape_banner_quotes",
        "web.py",
        '    body = html_lib.escape(format_skip_summary(summary))',
        '    body = html_lib.escape(format_skip_summary(summary), quote=False)',
        "Drop HTML escaping of quotes (quote=False) in banner",
    ),
    # 7a. Drop saver sentence from delivery stale warning
    (
        "M7a_drop_saver_delivery",
        "scheduler.py",
        '            f"follow B8b(1). On {warning_day.isoformat()} ET away from midnight, if MLB shows this season\'s "\n            "saver used, check bts saver-state --picks-dir DIR. For active or not_earned, run bts "\n            "saver-state --use --picks-dir DIR. For uninitialized, see B8a. Check the latest fetch status."',
        '            "follow B8b(1). Check the latest fetch status."',
        "Drop saver sentences from delivery stale warning",
    ),
    # 7b. Drop saver sentence from morning stale warning
    (
        "M7b_drop_saver_morning",
        "scheduler.py",
        '            f"unchanged. On {date} ET away from midnight, if MLB shows this season\'s saver used, check bts "\n            "saver-state --picks-dir DIR. For active or not_earned, run bts saver-state --use --picks-dir "\n            "DIR. For uninitialized, see B8a. Check the latest fetch status."',
        '            "unchanged. Check the latest fetch status."',
        "Drop saver sentences from morning stale warning",
    ),
    # 8a. Restore old last sentence of late refusal
    (
        "M8a_old_late_refusal",
        "scheduler.py",
        '           "The entry check may report a mismatch if it verifies the entry in time.")',
        '           "Enter a pick manually if you want one now.")',
        "Restore old last sentence of late refusal",
    ),
    # 8b. Restore old last sentence of void refusal (only void, not late)
    (
        "M8b_old_void_refusal",
        "scheduler.py",
        '    msg = (f"DELIVERY REFUSED for {date}: {games}. "\n           "This pick was withheld. A later check may produce another recommendation. "\n           "Use a later official pick message only while the entry window is open. "\n           "A manual entry can differ from a later recommendation. "\n           "The entry check may report a mismatch if it verifies the entry in time.")',
        '    msg = (f"DELIVERY REFUSED for {date}: {games}. "\n           "This pick was withheld. A later check may produce another recommendation. "\n           "Use a later official pick message only while the entry window is open. "\n           "A manual entry can differ from a later recommendation. "\n           "Enter a pick manually if you want one today.")',
        "Restore old last sentence of postponed/void refusal",
    ),
    # 9a. Let renderer raise on a string probability
    (
        "M9a_raise_str_prob",
        "scheduler.py",
        "    if type(value) not in (int, float):\n        return None",
        "    if isinstance(value, str):\n        raise TypeError('string number')\n    if type(value) not in (int, float):\n        return None",
        "Raise TypeError on string probability/bar in _skip_display_number",
    ),
    # 9b. Let renderer raise on a string probability ONLY in format_skip_dm when bar is valid float
    (
        "M9b_raise_str_prob_with_valid_bar",
        "scheduler.py",
        "def format_skip_dm(date: str, summary: object) -> str:",
        "def format_skip_dm(date: str, summary: object) -> str:\n    if isinstance(summary, dict) and isinstance(summary.get('best_p'), str) and type(summary.get('bar')) is float:\n        raise TypeError('string best_p with valid float bar')",
        "Raise TypeError on string best_p only when bar is a valid float in format_skip_dm",
    ),
]


def prepare_scratch_mutate(mut_id: str, rel_file: str, target: str, replacement: str) -> str:
    if SCRATCH_PKG.exists():
        shutil.rmtree(SCRATCH_PKG)
    SCRATCH_PKG.mkdir(parents=True)
    shutil.copytree(REPO / "src/bts", SCRATCH_PKG / "bts")
    fpath = SCRATCH_PKG / "bts" / rel_file
    src = fpath.read_text()
    assert target in src, f"Target not found for {mut_id} in {rel_file}"
    # Replace only first occurrence for M8a so M8a only mutates late refusal!
    new_src = src.replace(target, replacement, 1)
    assert new_src != src
    fpath.write_text(new_src)
    return f"{SCRATCH_PKG}:{REPO}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--batch", type=int, choices=range(6), default=None)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()

    if args.baseline:
        raw_full = run_pytest([])
        clean_filter = run_pytest(["-k", "not test_do_post_saver_transition_socket_wiring"])
        diffs = audit_existing_test_diffs()
        data = {"full_11_targets": raw_full, "clean_without_loopback_bind": clean_filter, "test_diffs": diffs}
        (OUT_DIR / "probe_6_baseline.json").write_text(json.dumps(data, indent=2) + "\n")
        print(json.dumps({"full": raw_full["counts"], "clean": clean_filter["counts"], "failed": raw_full["failed_lines"]}, indent=2))
        return

    if args.batch is not None:
        chunk = MUTATIONS[args.batch * 3:(args.batch + 1) * 3]
        batch_out = []
        try:
            for mut_id, rel_file, target, replacement, desc in chunk:
                py_path = prepare_scratch_mutate(mut_id, rel_file, target, replacement)
                res = run_pytest(["-k", "not test_do_post_saver_transition_socket_wiring"], pythonpath=py_path)
                killed_count = res["counts"]["failed"] + res["counts"]["errors"]
                batch_out.append({
                    "id": mut_id,
                    "description": desc,
                    "killed_by_count": killed_count,
                    "counts": res["counts"],
                    "sample_failures": res["failed_lines"][:5],
                })
        finally:
            if SCRATCH_PKG.exists():
                shutil.rmtree(SCRATCH_PKG)
        out_f = OUT_DIR / f"probe_6_batch_{args.batch}.json"
        out_f.write_text(json.dumps(batch_out, indent=2) + "\n")
        print(json.dumps(batch_out, indent=2))
        return

    if args.summarize:
        base = json.loads((OUT_DIR / "probe_6_baseline.json").read_text())
        muts = []
        for i in range(6):
            p = OUT_DIR / f"probe_6_batch_{i}.json"
            if p.exists():
                muts.extend(json.loads(p.read_text()))
        combined = {
            "baseline_full_11_targets": base["full_11_targets"],
            "baseline_clean": base["clean_without_loopback_bind"],
            "changed_test_files": base["test_diffs"]["all_changed_test_files"],
            "mutations": muts,
            "surviving_mutations": [m["id"] for m in muts if m["killed_by_count"] == 0],
        }
        (OUT_DIR / "probe_6_results.json").write_text(json.dumps(combined, indent=2) + "\n")
        print(json.dumps({
            "full_counts": combined["baseline_full_11_targets"]["counts"],
            "surviving_mutations": combined["surviving_mutations"],
            "table": [(m["id"], m["killed_by_count"]) for m in muts],
        }, indent=2))


if __name__ == "__main__":
    main()
