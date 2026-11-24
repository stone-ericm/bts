"""Slice 3 — Item 4: Mutation runner over 30 new tests in 619abda7..HEAD (weighted to pick path).

Modes:
- `--dry-check`: non-pytest validation of scratch copy, PYTHONPATH import precedence,
  unique target match + syntax compilation for all 30 mutations, and AST verification
  of all 30 target test functions.
- default (no flags): runs baseline pytest on the sampled test nodes, then applies each
  mutation one at a time in `.carbon-review/whole-range/scratch_mut/src`, runs pytest on
  the claimed test node (and if it survives, also checks the full test file), restores
  the file byte-for-byte, cleans up `scratch_mut/`, and writes `mutation_results.json`.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SCRATCH_MUT = HERE / "scratch_mut"
SCRATCH_SRC = SCRATCH_MUT / "src"
OUT_JSON = HERE / "mutation_results.json"

FORBIDDEN_TEST_IMPORTS = {
    "tests.test_wr_a_pick",
    "tests.test_wr_a_review_carbon",
    "tests.test_wr_c2_recovery",
    "tests.test_wr_b_edges",
    "tests.test_wr_c1_history",
    "tests.test_season_contract_cs4b2_edges",
}

# 30 (id, test_file, test_func, src_rel, target, replacement, description) tuples
MUTATIONS: list[tuple[str, str, str, str, str, str, str]] = [
    # --- Pick path: saved-pick refusal (CS5a) & delivery chokepoint ---
    (
        "M01",
        "tests/test_pick_chosen_on_cs5a.py",
        "test_pure_matcher",
        "bts/picks.py",
        '    elif not record["required"]:\n        chosen = "it was chosen without the live check (for example by the preview)"',
        '    elif False:\n        chosen = "it was chosen without the live check (for example by the preview)"',
        "saved_pick_refusal: drop required=False rejection (accept preview pick as live)",
    ),
    (
        "M02",
        "tests/test_pick_chosen_on_cs5a.py",
        "test_invalid_record_normalized_in_all_loaders",
        "bts/picks.py",
        "        or type(streak) is not int or streak < 0",
        "        or not isinstance(streak, int) or streak < 0",
        "_valid_chosen_on: relax exact int check on outer streak (accept bool True/False)",
    ),
    (
        "M03",
        "tests/test_saved_pick_paths_cs5a.py",
        "test_uncovered_preview_double_is_refused_then_replaced",
        "bts/scheduler.py",
        "        refusal = saved_pick_refusal(daily, decision_state)\n        if refusal is not None:",
        "        refusal = None\n        if refusal is not None:",
        "_deliver_and_lock_pick: bypass saved_pick_refusal in scheduler delivery",
    ),
    (
        "M04",
        "tests/test_saved_pick_paths_cs5a.py",
        "test_legacy_public_reuse",
        "bts/cli.py",
        "            refusal = saved_pick_refusal(result.daily, decision_state)\n            if refusal is not None:",
        "            refusal = None\n            if refusal is not None:",
        "bts run: bypass saved_pick_refusal on locked catch-up public post",
    ),
    (
        "M05",
        "tests/test_saved_pick_paths_cs5a.py",
        "test_fresh_choices_stamp_the_selection_read",
        "bts/orchestrator.py",
        "    if sel.decision is not None and sel.pick_result is not None and sel.pick_result.daily is not None:\n        sel.pick_result.daily.chosen_on = chosen_on_record(decision_state, required=required)",
        "    if sel.decision is not None and sel.pick_result is not None and sel.pick_result.daily is not None:\n        sel.pick_result.daily.chosen_on = chosen_on_record(decision_state, required=False)",
        "run_and_pick: stamp required=False instead of required=required",
    ),
    (
        "M06",
        "tests/test_saved_pick_paths_cs5a.py",
        "test_missed_pick_alert_copy",
        "bts/scheduler.py",
        '        timing = (f"~{mins_to_game:.0f} min to first pitch"\n                  if mins_to_game >= 0 else "first pitch has passed")',
        '        timing = (f"~{mins_to_game:.0f} min to first pitch"\n                  if mins_to_game > 0 else "first pitch has passed")',
        "_alert_missed_pick: boundary mins_to_game >= 0 -> > 0 at exact 0 min",
    ),
    (
        "M07",
        "tests/test_saved_pick_cs5a_review.py",
        "test_unlocked_unavailable_reuse_reaches_lineup_delivery_and_is_checked_as_saved",
        "bts/strategy.py",
        "            return SelectionResult(PickResult(daily=current, locked=False), None, None, None, None, None)",
        "            return SelectionResult(PickResult(daily=current, locked=True), None, None, None, None, None)",
        "select_pick: return locked=True instead of locked=False when available pool empty in Warmup",
    ),
    # --- Pick path: stale automatic reading stamp & warning (CS4c) ---
    (
        "M08",
        "tests/test_stale_delivery_cs4c.py",
        "test_attempt_snapshot_and_confirmed_warning",
        "bts/scheduler.py",
        "        _stamp_reading(now)\n        daily.delivery_attempted = True  # persist BEFORE the network call (E2 idempotency)",
        "        daily.delivery_attempted = True  # persist BEFORE the network call (E2 idempotency)",
        "_deliver_and_lock_pick (DM): omit _stamp_reading(now) before pre-send save",
    ),
    (
        "M09",
        "tests/test_stale_delivery_cs4c.py",
        "test_dated_attempt_clears_seeded_record",
        "bts/scheduler.py",
        "                daily.automatic_reading = automatic_reading_record(decision_state, attempt_time)",
        "                daily.automatic_reading = automatic_reading_record(decision_state, attempt_time) or daily.automatic_reading",
        "_stamp_reading: preserve seeded automatic_reading instead of clearing to None on dated obs",
    ),
    (
        "M10",
        "tests/test_stale_delivery_cs4c_carbon.py",
        "test_send_returning_no_id_and_reentry",
        "bts/scheduler.py",
        "        if required and isinstance(msg_id, str) and msg_id:\n            _warn_stale_contest_reading(daily.automatic_reading, config, picks_dir)",
        "        if required:\n            _warn_stale_contest_reading(daily.automatic_reading, config, picks_dir)",
        "_deliver_and_lock_pick (DM): warn stale reading even when send_dm returned empty/None id",
    ),
    # --- Pick path: morning state check (CS6b-1) ---
    (
        "M11",
        "tests/test_morning_state_cs6b1.py",
        "test_margin_uses_earliest_planned_target",
        "bts/scheduler.py",
        "            elif first_check_et - now < timedelta(minutes=MORNING_CHECK_MIN_LEAD_MIN):",
        "            elif first_check_et - now <= timedelta(minutes=MORNING_CHECK_MIN_LEAD_MIN):",
        "run_day morning check: < 10 min lead boundary mutated to <= 10 min (600s case)",
    ),
    (
        "M12",
        "tests/test_morning_state_cs6b1.py",
        "test_existing_commit_markers_suppress_startup_read",
        "bts/scheduler.py",
        "    if daily is not None and (pick_was_delivered(daily) or daily.delivery_attempted):\n        return",
        "    if daily is not None and pick_was_delivered(daily):\n        return",
        "_morning_state_check: drop daily.delivery_attempted commit marker suppression",
    ),
    (
        "M13",
        "tests/test_morning_state_cs6b1_carbon.py",
        "test_morning_dispatches_pass_sampled_now_et_date_not_wall_clock",
        "bts/scheduler.py",
        '                status_path=health_dir / "contest_morning_check_dm_delivery_status.json",\n                now_et_date=now.date(),',
        '                status_path=health_dir / "contest_morning_check_dm_delivery_status.json",',
        "_morning_state_check: omit explicit now_et_date=now.date() on CRITICAL dispatch",
    ),
    # --- Pick path: thin-slate checks, fallback wait, postponed delivery guard ---
    (
        "M14",
        "tests/test_thin_slate_checks.py",
        "test_early_checks_already_in_the_past_are_dropped_but_the_regular_check_is_kept",
        "bts/scheduler.py",
        "            if now is not None and t <= now:\n                continue",
        "            if now is not None and t < now:\n                continue",
        "plan_run_times: boundary t <= now -> t < now ('at or before now')",
    ),
    (
        "M15",
        "tests/test_thin_slate_checks.py",
        "test_a_two_hour_move_still_leaves_one_check_in_time",
        "bts/scheduler.py",
        "THIN_SLATE_EARLY_OFFSETS_MIN = (180, 120)",
        "THIN_SLATE_EARLY_OFFSETS_MIN = (180,)",
        "plan_run_times: drop 120-min early offset, leaving only 180-min",
    ),
    (
        "M16",
        "tests/test_thin_slate_checks.py",
        "test_a_selected_pick_is_delivered_at_the_deadline_when_the_later_checks_produce_nothing",
        "bts/scheduler.py",
        "            if (mins_to_game > fallback_min and state.final_skip_candidate is None\n                    and state.skip_summary is None):",
        "            if False:",
        "run_day step 5: disable post-checks sleep until fallback deadline",
    ),
    (
        "M17",
        "tests/test_postponed_delivery_guard.py",
        "test_a_postponed_double_down_leg_leaves_the_delivered_double_unchanged",
        "bts/scheduler.py",
        "        if len(void_games) == len(_committed_pick_game_pks(daily)):\n            _refuse_void_game_delivery(daily, config, picks_dir, state, date, label, void_games)\n            return False",
        "        if void_games:\n            _refuse_void_game_delivery(daily, config, picks_dir, state, date, label, void_games)\n            return False",
        "_deliver_and_lock_pick: refuse partial-void double instead of all-void only",
    ),
    (
        "M18",
        "tests/test_postponed_delivery_guard.py",
        "test_a_partial_void_double_still_faces_the_original_earliest_cutoff",
        "bts/scheduler.py",
        "    cutoff = submission_cutoff_et(daily)\n    now = _now_et()\n    if now >= cutoff:\n        _refuse_late_delivery(daily, config, picks_dir, state, date, label, cutoff, now)",
        "    cutoff = submission_cutoff_et(daily)\n    now = _now_et()\n    if now > cutoff:\n        _refuse_late_delivery(daily, config, picks_dir, state, date, label, cutoff, now)",
        "_deliver_and_lock_pick: cutoff boundary >= -> > in partial-void cutoff test",
    ),
    # --- Pick path: calendar & MDP season end (strategy.py & tail_policy.py) ---
    (
        "M19",
        "tests/test_season_end_date.py",
        "test_2027_last_day_and_after",
        "bts/strategy.py",
        'SEASON_END_DATES = {2026: "2026-09-28", 2027: "2027-09-27"}',
        'SEASON_END_DATES = {2026: "2026-09-28", 2027: "2027-09-26"}',
        "strategy.SEASON_END_DATES: off-by-one on 2027 exclusive end date (09-26 vs 09-27)",
    ),
    (
        "M20",
        "tests/test_wr_a_calendar_replay_carbon.py",
        "test_late_2027_shipped_policy_actions",
        "bts/strategy.py",
        "    if action == \"double\" and not ctx.allow_double:\n        action = \"single\"",
        "    if action == \"double\" and not ctx.allow_double:\n        action = \"skip\"",
        "resolve_policy_decision: clamp disallowed double to skip instead of single",
    ),
    # --- Live gate (CS6a), season contract (CS4b-1/2), dated obs (CS4a), saver (CS3) ---
    (
        "M21",
        "tests/test_live_gate_cs6a.py",
        "test_schedule_refuses_before_resource_access",
        "bts/cli.py",
        "    problem = live_gate_problem(config)\n    if problem is not None:\n        if not dry_run:\n            click.echo(f\"bts schedule: refusing to start: {problem}\", err=True)\n            raise SystemExit(2)",
        "    problem = None\n    if problem is not None:\n        if not dry_run:\n            click.echo(f\"bts schedule: refusing to start: {problem}\", err=True)\n            raise SystemExit(2)",
        "bts schedule: disable live_gate_problem startup refusal",
    ),
    (
        "M22",
        "tests/test_season_contract_cs4b1.py",
        "test_identifier_format",
        "bts/season_contract.py",
        '    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{32}", value) is not None',
        '    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{32}", value) is not None',
        "season_contract.valid_activation_id: allow uppercase hex in activation_id",
    ),
    (
        "M23",
        "tests/test_season_contract_cs4b2.py",
        "test_optional_and_dated_bypass_corrupt_switch",
        "bts/contest_state.py",
        "    if require_contest_state and not (\n            contest is not None and dated_observation_in_effect(contest, now_dt)):",
        "    if require_contest_state:",
        "load_decision_streak_state: fail to bypass corrupt season switch when today's dated obs is in effect",
    ),
    (
        "M24",
        "tests/test_contest_dated_cs4a.py",
        "test_write_and_noop_order_output_and_isolation",
        "bts/cli.py",
        '    identity_keys = ("effective_date", "active_streak", "best_streak", "saver_state", "reason")',
        '    identity_keys = ("effective_date", "active_streak", "best_streak", "saver_state")',
        "_write_dated_contest_observation: omit reason from idempotency identity_keys",
    ),
    (
        "M25",
        "tests/test_saver_cs3.py",
        "test_policy_saver_table",
        "bts/saver_state.py",
        '    return state.state == "active" or (state.state == "not_earned" and streak < 10)',
        '    return state.state == "active" or (state.state == "not_earned" and streak <= 10)',
        "policy_saver_available: boundary streak < 10 -> streak <= 10 on not_earned",
    ),
    # --- Scoring & reconcile: season-scoped streak (CS2), C3v, day-outcome, C1 clock, bullpen, alert ---
    (
        "M26",
        "tests/test_streak_season_cs2.py",
        "test_row1_readers_require_an_exact_integer_season_without_writing",
        "bts/picks.py",
        '    tag = data.get("season")\n    return data if type(tag) is int and tag == season else {}',
        '    tag = data.get("season")\n    return data if isinstance(tag, (int, float)) and tag == season else {}',
        "_load_streak_season: allow float/bool season tag instead of exact int",
    ),
    (
        "M27",
        "tests/test_day_outcome.py",
        "test_a_confirmation_that_does_not_bind_to_this_day_does_not_count",
        "bts/day_outcome.py",
        '            and rec.get("et_date") == date and rec.get("season") == int(date[:4])',
        '            and rec.get("et_date") == date and True',
        "day_outcome._binds: drop rec.get('season') == int(date[:4]) check on confirmed receipt",
    ),
    (
        "M28",
        "tests/test_wr_c1_clock.py",
        "test_mode_sample_and_last_et_microsecond",
        "bts/health/runner.py",
        "    if today is not None and today < wall_date_et:",
        "    if today is not None and today <= wall_date_et:",
        "health.runner: < wall_date_et -> <= wall_date_et (wrongly enter historical mode on same day)",
    ),
    (
        "M29",
        "tests/test_bullpen_same_day.py",
        "test_bullpen_availability_does_not_depend_on_same_day_reliever",
        "bts/features/compute.py",
        '                allow_exact_matches=False,',
        '                allow_exact_matches=True,',
        "compute.py bullpen merge_asof: allow_exact_matches=True (re-introduce same-day leakage)",
    ),
    (
        "M30",
        "tests/health/test_alert_status_cs6b2.py",
        "test_failed_count",
        "bts/health/alert.py",
        '    if data.get("status") == "failed" and type(n) is int and n > 0:',
        '    if data.get("status") == "failed" and isinstance(n, int) and n > 0:',
        "alert._resurface_note: isinstance(n, int) instead of type(n) is int (accepts bool True as 1)",
    ),
]


def _prepare_scratch() -> None:
    if SCRATCH_MUT.exists():
        shutil.rmtree(SCRATCH_MUT)
    SCRATCH_MUT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO_ROOT / "src", SCRATCH_SRC, symlinks=True)


def _verify_all_specs() -> None:
    assert len(MUTATIONS) >= 25, f"expected >= 25 mutations, got {len(MUTATIONS)}"
    seen_tests: set[tuple[str, str]] = set()
    for mid, tfile, tfunc, srel, target, replacement, _desc in MUTATIONS:
        assert (tfile, tfunc) not in seen_tests, f"{mid}: duplicate test target {(tfile, tfunc)}"
        seen_tests.add((tfile, tfunc))

        # Verify test file exists, parses, defines tfunc, and has no forbidden imports
        tpath = REPO_ROOT / tfile
        assert tpath.is_file(), f"{mid}: missing test file {tfile}"
        tree = ast.parse(tpath.read_text(encoding="utf-8"), filename=tfile)
        funcs = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        assert tfunc in funcs, f"{mid}: {tfunc} not found in {tfile}"
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in FORBIDDEN_TEST_IMPORTS:
                raise AssertionError(f"{mid}: {tfile} imports broken {node.module}")

        # Verify target matches uniquely in scratch copy, compiles when mutated, and restores cleanly
        spath = SCRATCH_SRC / srel
        assert spath.is_file(), f"{mid}: missing source file {srel}"
        orig_bytes = spath.read_bytes()
        orig_text = orig_bytes.decode("utf-8")
        count = orig_text.count(target)
        assert count == 1, f"{mid}: expected target count 1 in {srel}, got {count}"
        mutated_text = orig_text.replace(target, replacement, 1)
        assert mutated_text != orig_text, f"{mid}: replacement produced identical text"
        compile(mutated_text, str(spath), "exec")
        spath.write_bytes(orig_bytes)
        assert spath.read_bytes() == orig_bytes

    # Verify PYTHONPATH import precedence resolves `bts` from SCRATCH_SRC
    env = dict(os.environ, PYTHONPATH=str(SCRATCH_SRC))
    out = subprocess.check_output(
        [sys.executable, "-c", "import bts; print(bts.__file__)"],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        timeout=15,
    ).strip()
    assert Path(out).resolve().is_relative_to(SCRATCH_SRC.resolve()), (
        f"PYTHONPATH did not shadow editable install: got {out}"
    )


def _run_pytest_node(node: str, env: dict[str, str], timeout: int = 45) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=no", "--no-header", node],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    tail = (proc.stdout + "\n" + proc.stderr).strip()[-300:]
    return proc.returncode, tail


def main() -> int:
    dry_check = "--dry-check" in sys.argv[1:]
    try:
        _prepare_scratch()
        _verify_all_specs()
        if dry_check:
            print(f"DRY-CHECK OK: {len(MUTATIONS)} mutations verified against scratch copy and AST.")
            return 0

        env = dict(os.environ, PYTHONPATH=str(SCRATCH_SRC), LOKY_MAX_CPU_COUNT="1")

        # 1. Baseline check: verify all 30 unmutated target test nodes pass in a single pytest invocation
        all_nodes = [f"{tfile}::{tfunc}" for _, tfile, tfunc, *_ in MUTATIONS]
        base_proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=short", "--no-header", *all_nodes],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if base_proc.returncode != 0:
            print("BASELINE FAILED:\n" + base_proc.stdout + "\n" + base_proc.stderr, file=sys.stderr)
            return 2

        results: list[dict] = []
        survivors_on_target: list[str] = []
        survivors_on_whole_file: list[str] = []

        for mid, tfile, tfunc, srel, target, replacement, desc in MUTATIONS:
            spath = SCRATCH_SRC / srel
            orig_bytes = spath.read_bytes()
            mutated_text = orig_bytes.decode("utf-8").replace(target, replacement, 1)
            node = f"{tfile}::{tfunc}"
            try:
                spath.write_text(mutated_text, encoding="utf-8")
                rc, summary = _run_pytest_node(node, env)
                killed_by_target = rc != 0
                killed_by_file = None
                if not killed_by_target:
                    survivors_on_target.append(mid)
                    rc_file, _ = _run_pytest_node(tfile, env)
                    killed_by_file = rc_file != 0
                    if not killed_by_file:
                        survivors_on_whole_file.append(mid)
                results.append({
                    "id": mid,
                    "test_node": node,
                    "src_file": f"src/{srel}",
                    "description": desc,
                    "killed_by_target_test": killed_by_target,
                    "killed_by_test_file": killed_by_file,
                    "pytest_rc": rc,
                    "output_tail": summary,
                })
                status = "KILLED" if killed_by_target else (
                    "SURVIVED_TARGET_KILLED_BY_FILE" if killed_by_file else "SURVIVED_WHOLE_FILE"
                )
                print(f"{mid} [{status}] {node} <- src/{srel}: {desc}", flush=True)
            finally:
                spath.write_bytes(orig_bytes)

        payload = {
            "total_mutations": len(MUTATIONS),
            "killed_by_target_count": len(MUTATIONS) - len(survivors_on_target),
            "survivors_on_target": survivors_on_target,
            "survivors_on_whole_file": survivors_on_whole_file,
            "results": results,
        }
        OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(
            f"\nSUMMARY: {payload['killed_by_target_count']}/{len(MUTATIONS)} killed by target test; "
            f"target survivors={survivors_on_target}; whole-file survivors={survivors_on_whole_file}"
        )
        return 0
    finally:
        shutil.rmtree(SCRATCH_MUT, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
