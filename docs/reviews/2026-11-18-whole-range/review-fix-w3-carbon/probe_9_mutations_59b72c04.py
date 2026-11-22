"""Probe 9: Scratch-copy mutation runner targeting 59b72c04 (run ONLY after author's gate finishes and pgrep -f yield_run.sh is clear)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import time

REPO = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
BASE_TARGET_SRC = REVIEW_DIR / "target_59b72c04" / "src"
SCRATCH_SRC = REVIEW_DIR / "scratch_mutant_59b72c04_src"


MUTANTS: list[tuple[str, str, str, str, list[str]]] = [
    (
        "m01_drop_byte_comparison",
        "bts/scheduler.py",
        "if daily is not None and daily.date == date and raw == saved_bytes:",
        "if daily is not None and daily.date == date:",
        ["tests/test_wr_w3.py", "-k", "replacement or whitespace"],
    ),
    (
        "m02_reuse_after_sleep",
        "bts/scheduler.py",
        "if not slept_to_deadline and fresh_decision is not None and saved_bytes is not None:",
        "if fresh_decision is not None and saved_bytes is not None:",
        ["tests/test_wr_w3.py", "-k", "sleep"],
    ),
    (
        "m03_skip_post_marker_cutoff_recheck",
        "bts/scheduler.py",
        "if now < cutoff and now.date() == snapshot_day:",
        "if now.date() == snapshot_day:",
        ["tests/test_wr_w3.py", "-k", "attempt_write_reaches_cutoff"],
    ),
    (
        "m04a_skip_et_day_recheck_after_read",
        "bts/scheduler.py",
        "            else:\n                if _now_et().date() == snapshot_day:\n                    return reading",
        "            else:\n                return reading",
        ["tests/test_wr_w3.py", "-k", "live_day_at_midnight"],
    ),
    (
        "m04b_skip_et_day_recheck_before_transport",
        "bts/scheduler.py",
        "if now < cutoff and now.date() == snapshot_day:",
        "if now < cutoff:",
        ["tests/test_wr_w3.py", "-k", "live_day_at_midnight"],
    ),
    (
        "m05a_drop_both_scoreable_checks",
        "bts/scheduler.py",
        'if record is not None and record["scoreable"]:',
        "if False:",
        ["tests/test_wr_w3.py", "-k", "scoreable"],
    ),
    (
        "m05b_drop_post_poll_scoreable_check_only",
        "bts/scheduler.py",
        '                        if current_record is not None:\n                            record, _ = current_record\n                            if record is not None and record["scoreable"]:\n                                preserve_reuse_commit(record)\n                                break',
        "                        if current_record is not None:\n                            record, _ = current_record",
        ["tests/test_wr_w3.py", "tests/test_incident_2026_08_30.py", "tests/test_saved_pick_paths_cs5a.py", "tests/test_fallback_plan.py"],
    ),
    (
        "m06a_late_lookup_no_deadline_cap",
        "bts/scheduler.py",
        "deadline = time.monotonic() + 2.0",
        "deadline = time.monotonic() + 10.0",
        ["tests/test_wr_w3.py", "-k", "partial or slow_status"],
    ),
    (
        "m06b_late_worker_mutates_confirmed_sides_after_return",
        "bts/scheduler.py",
        "        lookups = (lambda pk=pk: check_confirmed_lineups([pk]) for pk in pending_pks)\n        for confirmation in _reuse_lookups(lookups, deadline):\n            confirmed_sides.update(\n                (pk, side) for pk, sides in confirmation.items() for side in sides\n            )",
        "        def _mutating_lookup(pk):\n            res = check_confirmed_lineups([pk])\n            confirmed_sides.update((k, s) for k, sides in res.items() for s in sides)\n            return res\n        lookups = (lambda pk=pk: _mutating_lookup(pk) for pk in pending_pks)\n        for _ in _reuse_lookups(lookups, deadline):\n            pass",
        ["tests/test_wr_w3.py", "-k", "pending_window or overdue"],
    ),
    (
        "m07a_pass_selection_whatever_it_matched",
        "bts/scheduler.py",
        "selection=_reuse_commit_selection(refresh.selection, daily),",
        "selection=refresh.selection,",
        ["tests/test_wr_w3.py", "-k", "warmup_cached_selection"],
    ),
    (
        "m07b_reuse_commit_selection_ignore_double_slot",
        "bts/scheduler.py",
        'and (action == "single" or selection.double_candidate == _row_from_daily(daily.double_down))):',
        "and True):",
        ["tests/test_wr_w3.py", "tests/test_saved_pick_paths_cs5a.py"],
    ),
    (
        "m08_coalesce_after_reuse_defer",
        "bts/scheduler.py",
        "                        if not reused_check_cascade:\n                            for j in range(run_idx + 1, len(runs)):",
        "                        if True:\n                            for j in range(run_idx + 1, len(runs)):",
        ["tests/test_wr_w3.py", "-k", "overdue_confirmed_check"],
    ),
    (
        "m09_post_poll_pick_blocking_read",
        "bts/picks.py",
        "    if regular_only:\n        raw = read_regular_file_bytes(path)",
        "    if regular_only:\n        raw = path.read_bytes()",
        ["tests/test_wr_w3.py", "tests/test_saved_pick_paths_cs5a.py"],
    ),
    (
        "m10_drop_prior_reading_restore_on_refusal",
        "bts/scheduler.py",
        "        daily.delivery_attempted = False\n        daily.automatic_reading = prior_reading",
        "        daily.delivery_attempted = False",
        ["tests/test_wr_w3.py", "tests/test_saved_pick_paths_cs5a.py"],
    ),
    (
        "m11_pick_matches_reuse_commit_ignore_double_leg",
        "bts/scheduler.py",
        "            and (daily.double_down is None or _matches_pick_slot(\n                record.get(\"double_down\"), daily.double_down)))",
        "            and True)",
        ["tests/test_wr_w3.py", "tests/test_scheduler_decision_record_integration.py"],
    ),
    (
        "m12_drop_coalesced_filter_in_reuse_planner",
        "bts/scheduler.py",
        '                    if reused_check_cascade:\n                        remaining_runs = [\n                            run for index, run in enumerate(future_runs, run_idx + 1)\n                            if index not in coalesced\n                        ]\n                        planner_inputs["remaining_runs"] = remaining_runs',
        "                    if False:\n                        pass",
        ["tests/test_wr_w3.py", "tests/test_fallback_plan.py", "tests/test_incident_2026_08_30.py"],
    ),
    (
        "m13_cutoff_3s_guard_relaxed_to_0s",
        "bts/scheduler.py",
        "if (cutoff - _now_et()).total_seconds() > 3:",
        "if (cutoff - _now_et()).total_seconds() > 0:",
        ["tests/test_wr_w3.py", "tests/test_fallback_plan.py"],
    ),
    (
        "m14a_observer_takes_last_write_instead_of_candidate_first",
        "bts/picks.py",
        "        if daily is candidate and not written:\n            written.append(text.encode(\"ascii\"))",
        "        written[:] = [text.encode(\"ascii\")]",
        ["tests/test_wr_w3.py", "-k", "audit_replacement"],
    ),
    (
        "m14b_observer_drops_identity_check_only",
        "bts/picks.py",
        "if daily is candidate and not written:",
        "if not written:",
        ["tests/test_wr_w3.py", "tests/test_saved_pick_paths_cs5a.py"],
    ),
]


def prepare_scratch() -> None:
    if SCRATCH_SRC.exists():
        shutil.rmtree(SCRATCH_SRC)
    shutil.copytree(BASE_TARGET_SRC, SCRATCH_SRC)


def verify_scratch_import(env: dict[str, str]) -> None:
    check = subprocess.run(
        [
            str(REPO / ".venv/bin/python"),
            "-c",
            f"import bts, pathlib; p = pathlib.Path(bts.__file__).resolve(); assert p.is_relative_to(pathlib.Path({str(SCRATCH_SRC.resolve())!r})), p",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if check.returncode != 0:
        raise RuntimeError(f"Scratch import check failed: {check.stderr}")


def main() -> None:
    prepare_scratch()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SCRATCH_SRC.resolve())
    env["TZ"] = "America/New_York"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    verify_scratch_import(env)

    originals = {
        "bts/scheduler.py": (BASE_TARGET_SRC / "bts/scheduler.py").read_text(),
        "bts/picks.py": (BASE_TARGET_SRC / "bts/picks.py").read_text(),
    }

    report = []
    t_all = time.monotonic()
    for name, rel_file, target, replacement, pytest_args in MUTANTS:
        for rf, content in originals.items():
            (SCRATCH_SRC / rf).write_text(content)
        src_text = originals[rel_file]
        if target not in src_text:
            raise RuntimeError(f"Target anchor not found for {name} in {rel_file}")
        mutated = src_text.replace(target, replacement, 1)
        assert mutated != src_text
        (SCRATCH_SRC / rel_file).write_text(mutated)

        t0 = time.monotonic()
        proc = subprocess.run(
            [str(REPO / ".venv/bin/python"), "-m", "pytest", "-q", "--tb=no", *pytest_args],
            cwd=str(REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        dt = round(time.monotonic() - t0, 2)
        killed = proc.returncode != 0
        tail = (proc.stdout.strip().splitlines() or [""])[-1]
        row = {
            "mutant": name,
            "file": rel_file,
            "killed": killed,
            "returncode": proc.returncode,
            "duration_sec": dt,
            "summary": tail,
        }
        report.append(row)
        status_word = "KILLED" if killed else "SURVIVED"
        print(f"[{status_word}] {name} ({dt}s): {tail}", flush=True)

    shutil.rmtree(SCRATCH_SRC, ignore_errors=True)
    out = {
        "total_duration_sec": round(time.monotonic() - t_all, 2),
        "killed_count": sum(1 for r in report if r["killed"]),
        "survived_count": sum(1 for r in report if not r["killed"]),
        "mutants": report,
    }
    out_path = REVIEW_DIR / "probe_9_mutations.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Done in {out['total_duration_sec']}s: {out['killed_count']} killed, {out['survived_count']} survived.")


if __name__ == "__main__":
    main()
