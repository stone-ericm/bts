"""Probe 6c: Run remaining 4 mutations (8a, 8b, 9a, 9b) on scratch copy and clean up mut_src."""
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


def run_pytest(extra_src: Path, fast_skip_bars: bool = False) -> tuple[int, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{extra_src}:{REPO}"
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
            "8a_restore_old_last_sentence_of_late_refusal",
            "bts/scheduler.py",
            '    msg = (f"LATE DELIVERY REFUSED: {names} would have been sent {late_min:.0f} min after the "\n'
            '           f"{cutoff.strftime(\'%H:%M ET\')} cutoff. "\n'
            '           "This pick was withheld. A later check may produce another recommendation. "\n'
            '           "Use a later official pick message only while the entry window is open. "\n'
            '           "A manual entry can differ from a later recommendation. "\n'
            '           "The entry check may report a mismatch if it verifies the entry in time.")',
            '    msg = (f"LATE DELIVERY REFUSED: {names} would have been sent {late_min:.0f} min after the "\n'
            '           f"{cutoff.strftime(\'%H:%M ET\')} cutoff. "\n'
            '           "This pick was withheld. A later check may produce another recommendation. "\n'
            '           "Use a later official pick message only while the entry window is open. "\n'
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
            rc, out = run_pytest(MUT_ROOT / "src", fast_skip_bars=False)
        killed = rc != 0
        lines = [ln for ln in out.splitlines() if ln.strip()]
        fail_line = next((ln for ln in lines if "FAILED" in ln or "AssertionError" in ln or "ValueError" in ln), lines[-1] if lines else "")
        results.append({"mutation": name, "killed": killed, "detail": fail_line[:220]})
        print(f"[{'KILLED' if killed else 'SURVIVED'}] {name} -> {fail_line[:130]}", flush=True)

    shutil.rmtree(MUT_ROOT, ignore_errors=True)
    out_file = REPO / ".carbon-review/wr-e/mutation_results_tail.json"
    out_file.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
