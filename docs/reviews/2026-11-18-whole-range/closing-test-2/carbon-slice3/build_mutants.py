"""Slice 3 probe family 4: build and verify 32 isolated scratch production mutants for 'Tests that cannot fail'.

Adoptable and import-safe (no module-level bts.model.predict import, no leaked state, no network).
Run standalone to materialize .carbon-review/whole-range/mutants/m01..m32 and verify imports:
    nice -n 19 .venv/bin/python .carbon-review/whole-range/build_mutants.py
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
MUTANTS_DIR = REVIEW_DIR / "mutants"
LOGS_DIR = REVIEW_DIR / "logs"

MUTANTS: list[dict[str, str]] = [
    # --- Change set G: policy guard (5 mutants) ---
    {
        "id": "m01",
        "area": "G (policy guard donor codes)",
        "file": "src/bts/simulate/mdp.py",
        "target": "            if donor_code in (1, 2):",
        "replacement": "            if donor_code in (2,):",
        "pytest_cmd": "tests/test_wr_g_guard.py::test_lookup_edge_shapes",
    },
    {
        "id": "m02",
        "area": "G (policy guard nearest-lower-bin order)",
        "file": "src/bts/simulate/mdp.py",
        "target": "        for j in range(q - 1, -1, -1):",
        "replacement": "        for j in range(0, q):",
        "pytest_cmd": "tests/test_wr_g_guard.py::test_lookup_edge_shapes",
    },
    {
        "id": "m03",
        "area": "G (write_decision clears policy_guard on skip)",
        "file": "src/bts/daily_decision.py",
        "target": '            "policy_guard": policy_guard if action != "skip" else None,',
        "replacement": '            "policy_guard": policy_guard,',
        "pytest_cmd": "tests/test_wr_g_guard.py::test_decision_round_trip_and_skip_predicate",
    },
    {
        "id": "m04",
        "area": "G (_log_policy_decision guard journal line)",
        "file": "src/bts/scheduler.py",
        "target": "    if guard is not None:",
        "replacement": "    if False and guard is not None:",
        "pytest_cmd": "tests/test_wr_g_guard.py::test_guard_journal",
    },
    {
        "id": "m05",
        "area": "G (_commit_decision_for_pick Warmup reuse candidate)",
        "file": "src/bts/scheduler.py",
        "target": '    has_decision = getattr(selection, "decision", None) is not None',
        "replacement": "    has_decision = selection is not None",
        "pytest_cmd": 'tests/test_wr_g_guard.py -k "test_guarded_delivery_records_selected_batter and warmup"',
    },
    # --- Change set H: regime >= 57 (5 mutants) ---
    {
        "id": "m06",
        "area": "H (mdp_objective streak >= 57 boundary)",
        "file": "src/bts/simulate/tail_policy.py",
        "target": "    if streak >= LONGEST_STREAK_FROM and days_remaining > 0:",
        "replacement": "    if streak > LONGEST_STREAK_FROM and days_remaining > 0:",
        "pytest_cmd": "tests/test_wr_h_longest.py::test_h1_h2_h3_resolver_grid",
    },
    {
        "id": "m07",
        "area": "H (longest_streak_action double gain 2.0 + gain)",
        "file": "src/bts/simulate/tail_policy.py",
        "target": "        values = (p_hit * (1.0 + gain), p_both * (2.0 + gain), gain)",
        "replacement": "        values = (p_hit * (1.0 + gain), p_both * (1.0 + gain), gain)",
        "pytest_cmd": "tests/test_wr_h_longest.py::test_r1_healthy_paired_base_can_double",
    },
    {
        "id": "m08",
        "area": "H (_longest_action base-tail SHA pairing check)",
        "file": "src/bts/strategy.py",
        "target": "        if sha != tail.base_policy_sha256:",
        "replacement": "        if False and sha != tail.base_policy_sha256:",
        "pytest_cmd": 'tests/test_wr_h_longest.py -k "test_h4_bad_artifacts_force_single and tail-mismatch"',
    },
    {
        "id": "m09",
        "area": "H (_validate_longest_base masked array guard)",
        "file": "src/bts/strategy.py",
        "target": "    if np.ma.isMaskedArray(table):",
        "replacement": "    if False and np.ma.isMaskedArray(table):",
        "pytest_cmd": "tests/test_wr_h_longest.py::test_f2_masked_table_forces_single",
    },
    {
        "id": "m10",
        "area": "H (is_reach57_mdp_skip excludes longest_streak)",
        "file": "src/bts/daily_decision.py",
        "target": "        and decision_objective(rec) == OBJECTIVE_REACH57",
        "replacement": "        and decision_objective(rec) in OBJECTIVES",
        "pytest_cmd": "tests/test_wr_h_longest.py::test_h7_finalized_decision",
    },
    # --- Change set H2: raised contest-state caps (5 mutants) ---
    {
        "id": "m11",
        "area": "H2 (MAX_SEASON_STREAK = 400 cap)",
        "file": "src/bts/contest_state.py",
        "target": "MAX_SEASON_STREAK = 400",
        "replacement": "MAX_SEASON_STREAK = 57",
        "pytest_cmd": "tests/test_wr_h2_caps.py::test_m1_dated_cli_selects_delivers_and_commits",
    },
    {
        "id": "m12",
        "area": "H2 (validate_fetch best_streak <= MAX_SEASON_STREAK)",
        "file": "src/bts/contest_fetch.py",
        "target": "    if best_streak > MAX_SEASON_STREAK:",
        "replacement": "    if best_streak > MAX_SEASON_STREAK + 10:",
        "pytest_cmd": "tests/test_wr_h2_caps.py::test_k5_fetch_validation",
    },
    {
        "id": "m13",
        "area": "H2 (_valid_chosen_on dated active <= best <= MAX_SEASON_STREAK)",
        "file": "src/bts/picks.py",
        "target": "        or active != streak or not 0 <= active <= best <= MAX_SEASON_STREAK",
        "replacement": "        or active != streak or not 0 <= active <= MAX_SEASON_STREAK",
        "pytest_cmd": 'tests/test_wr_h2_caps.py -k "test_k7_chosen_on_boundary_and_identity and dated"',
    },
    {
        "id": "m14",
        "area": "H2 (classify_best_streak rejects best < active_streak)",
        "file": "src/bts/contest_state.py",
        "target": "    if best < 0 or best < contest.streak or best > MAX_SEASON_STREAK:",
        "replacement": "    if best < 0 or best > MAX_SEASON_STREAK:",
        "pytest_cmd": "tests/test_wr_h2_caps.py::test_k8_parsed_best_requires_current_provenance",
    },
    {
        "id": "m15",
        "area": "H2 (set-contest-streak CLI rejects best_streak > MAX_SEASON_STREAK)",
        "file": "src/bts/cli.py",
        "target": "        if best_streak > MAX_SEASON_STREAK:",
        "replacement": "        if best_streak > MAX_SEASON_STREAK + 10:",
        "pytest_cmd": "tests/test_wr_h2_caps.py::test_k6_dated_cli_rejects_before_writes",
    },
    # --- Change sets F1 & F1b: entry watchdog game check & marker persistence (5 mutants) ---
    {
        "id": "m16",
        "area": "F1 (entry_receipt.qualify wrong_game detection)",
        "file": "src/bts/entry_receipt.py",
        "target": '            elif ident[0] != s["game_pk"]:',
        "replacement": '            elif False and ident[0] != s["game_pk"]:',
        "pytest_cmd": "tests/test_wr_f1_wrong_game.py::test_m2_first_observation_is_wrong_game",
    },
    {
        "id": "m17",
        "area": "F1 (lost unit binding retains prior wrong_game warning)",
        "file": "src/bts/cli.py",
        "target": '                if not affected and prior_wrong_game and not qualification["all_confirmed"]:',
        "replacement": '                if False and not affected and prior_wrong_game and not qualification["all_confirmed"]:',
        "pytest_cmd": "tests/test_wr_f1_wrong_game.py::test_m9_lost_binding_does_not_clear_warning",
    },
    {
        "id": "m18",
        "area": "F1 (mark preserves game_unresolved across non-confirmed bridges)",
        "file": "src/bts/cli.py",
        "target": '        if marker["status"] != "confirmed" and (reason == "wrong_game" or prior_wrong_game):',
        "replacement": '        if marker["status"] != "confirmed" and reason == "wrong_game":',
        "pytest_cmd": "tests/test_wr_f1_review.py::test_m13_partial_double_proof",
    },
    {
        "id": "m19",
        "area": "F1 (exact-int game_pk check rejects bool game_pk)",
        "file": "src/bts/cli.py",
        "target": '                            if result["state"] == "wrong_game" and type(slot["game_pk"]) is int]',
        "replacement": '                            if result["state"] == "wrong_game" and isinstance(slot["game_pk"], int)]',
        "pytest_cmd": "tests/test_wr_f1_game_pk_types.py::test_a1_non_integer_game_id_cannot_trigger_wrong_game",
    },
    {
        "id": "m20",
        "area": "F1b (dm_failed after wrong_game + present_unverified keeps escalations)",
        "file": "src/bts/cli.py",
        "target": '        if prior_wrong_game or reason == "wrong_game":',
        "replacement": '        if reason == "wrong_game":',
        "pytest_cmd": "tests/test_wr_f1b_carbon_regressions.py::test_dm_failed_after_wrong_game_and_present_unverified_keeps_consumed_wrong_game_tier",
    },
    # --- Change set F2: saver-mismatch WARN (3 mutants) ---
    {
        "id": "m21",
        "area": "F2 (dated used warns against both active and not_earned saver flag)",
        "file": "src/bts/health/contest_state.py",
        "target": '            if flag.state in {"active", "not_earned"}:',
        "replacement": '            if flag.state in {"active"}:',
        "pytest_cmd": 'tests/test_wr_f2_saver_flag.py -k "test_m1_m2_live_warning_and_delivery and not_earned and 14"',
    },
    {
        "id": "m22",
        "area": "F2 (runner splits mismatch WARN out of primary digest)",
        "file": "src/bts/health/runner.py",
        "target": "        digest_alerts = [alert for alert in alerts if alert.incident_key != mismatch_key]",
        "replacement": "        digest_alerts = list(alerts)",
        "pytest_cmd": "tests/test_wr_f2_saver_flag.py::test_existing_digest_sends_first_and_stays_below_chat_limit",
    },
    {
        "id": "m23",
        "area": "F2 (runner requires transport message_id on mismatch receipt)",
        "file": "src/bts/health/runner.py",
        "target": "            require_message_id=True,",
        "replacement": "            require_message_id=False,",
        "pytest_cmd": "tests/test_wr_f2_saver_flag.py::test_failed_send_and_interrupt_before_status_can_retry",
    },
    # --- day-outcome & entry receipt precedence (3 mutants) ---
    {
        "id": "m24",
        "area": "day-outcome (batter-only match requires all slots confirmed/unit_unverified)",
        "file": "src/bts/day_outcome.py",
        "target": '                   and all(s.get("state") in ("confirmed", "unit_unverified")',
        "replacement": '                   and any(s.get("state") in ("confirmed", "unit_unverified")',
        "pytest_cmd": "tests/test_day_outcome.py::test_batter_match_with_other_qualification_states_stays_unverifiable",
    },
    {
        "id": "m25",
        "area": "day-outcome (partly void slate still requires decision)",
        "file": "src/bts/day_outcome.py",
        "target": "    if games > 0 and void_games == games:",
        "replacement": "    if games > 0 and void_games >= 1:",
        "pytest_cmd": "tests/test_day_outcome.py::test_partly_void_slate_still_needs_a_decision",
    },
    {
        "id": "m26",
        "area": "day-outcome (confirmed entry precedes batter-only regardless of order)",
        "file": "src/bts/day_outcome.py",
        "target": "    if in_time:",
        "replacement": "    if False and in_time:",
        "pytest_cmd": "tests/test_day_outcome_entry_sequence.py::test_receipt_precedence_is_independent_of_observation_order",
    },
    # --- Pick path & core range change sets: 5a, 4c, 6b-1, 6b-2, Ea, C2 (6 mutants) ---
    {
        "id": "m27",
        "area": "5a (saved_pick_refusal rejects required=False preview pick)",
        "file": "src/bts/picks.py",
        "target": '    elif not record["required"]:',
        "replacement": '    elif False and not record["required"]:',
        "pytest_cmd": "tests/test_saved_pick_paths_cs5a.py::test_uncovered_preview_double_is_refused_then_replaced",
    },
    {
        "id": "m28",
        "area": "4c (_deliver_and_lock_pick stamps automatic_reading at attempt instant)",
        "file": "src/bts/scheduler.py",
        "target": "        _stamp_reading(now)",
        "replacement": "        pass",
        "pytest_cmd": 'tests/test_stale_delivery_cs4c.py -k "test_attempt_snapshot_and_confirmed_warning and 20-old-True-dm"',
    },
    {
        "id": "m29",
        "area": "6b-1 (morning check 10-min lead margin uses strict <)",
        "file": "src/bts/scheduler.py",
        "target": "            elif first_check_et - now < timedelta(minutes=MORNING_CHECK_MIN_LEAD_MIN):",
        "replacement": "            elif first_check_et - now <= timedelta(minutes=MORNING_CHECK_MIN_LEAD_MIN):",
        "pytest_cmd": "tests/test_morning_state_cs6b1.py::test_margin_uses_earliest_planned_target",
    },
    {
        "id": "m30",
        "area": "6b-2 (_resurface_note requires exact int critical_count, rejecting bool True)",
        "file": "src/bts/health/alert.py",
        "target": '    if data.get("status") == "failed" and type(n) is int and n > 0:',
        "replacement": '    if data.get("status") == "failed" and isinstance(n, int) and n > 0:',
        "pytest_cmd": "tests/health/test_alert_status_cs6b2.py::test_failed_count",
    },
    {
        "id": "m31",
        "area": "Ea (maybe_notify_skip passes formatted skip DM body to transport)",
        "file": "src/bts/scheduler.py",
        "target": "        send(recipient, format_skip_dm(state.date, summary))",
        "replacement": '        send(recipient, "")',
        "pytest_cmd": "tests/test_wr_e_texts.py::test_skip_notice_restart_and_retry",
    },
    {
        "id": "m32",
        "area": "6b-2/C2 (_alert_delivery_outcome_unknown passes slate date to now_et_date)",
        "file": "src/bts/scheduler.py",
        "target": "            now_et_date=slate,",
        "replacement": "            now_et_date=None,",
        "pytest_cmd": 'tests/test_alert_status_callers_cs6b2.py -k "test_shared_alert_callers and unconfirmed and green_missing"',
    },
]


def test_mutant_targets_are_unique_and_valid() -> None:
    """Adoptable regression test: every mutant target matches once in HEAD and compiles cleanly."""
    assert len(MUTANTS) >= 25
    seen_ids = set()
    for spec in MUTANTS:
        mid = spec["id"]
        assert mid not in seen_ids, f"duplicate mutant id {mid}"
        seen_ids.add(mid)
        src_file = REPO_ROOT / spec["file"]
        text = src_file.read_text(encoding="utf-8")
        occurrences = text.count(spec["target"])
        assert occurrences == 1, f"{mid}: expected 1 occurrence in {spec['file']}, found {occurrences}"
        mutated = text.replace(spec["target"], spec["replacement"], 1)
        assert mutated != text
        compile(mutated, str(src_file), "exec")


def materialize_all_mutants() -> list[dict[str, str]]:
    if MUTANTS_DIR.exists():
        shutil.rmtree(MUTANTS_DIR)
    MUTANTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    src_bts = REPO_ROOT / "src" / "bts"
    for spec in MUTANTS:
        mid = spec["id"]
        dest_bts = MUTANTS_DIR / mid / "src" / "bts"
        shutil.copytree(src_bts, dest_bts, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        target_path = MUTANTS_DIR / mid / spec["file"]
        original = target_path.read_text(encoding="utf-8")
        count = original.count(spec["target"])
        if count != 1:
            raise RuntimeError(f"{mid}: target matched {count} times in {spec['file']}")
        mutated = original.replace(spec["target"], spec["replacement"], 1)
        compile(mutated, str(target_path), "exec")
        target_path.write_text(mutated, encoding="utf-8")

    # Verify PYTHONPATH shadowing in a subprocess
    m01_src = MUTANTS_DIR / "m01" / "src"
    env = dict(os.environ, PYTHONPATH=f"{m01_src}:{REPO_ROOT}")
    check = subprocess.run(
        [sys.executable, "-c", "import bts; print(bts.__file__)"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=True,
    )
    resolved = Path(check.stdout.strip()).resolve()
    expected = (m01_src / "bts" / "__init__.py").resolve()
    assert resolved == expected, f"PYTHONPATH failed to shadow src/bts: {resolved} != {expected}"

    # Verify all target test modules import cleanly in this copy (without running pytest)
    test_files = sorted({spec["pytest_cmd"].split()[0].split("::")[0] for spec in MUTANTS})
    import_code = (
        "import importlib.util, sys\n"
        "for rel in sys.argv[1:]:\n"
        "    spec = importlib.util.spec_from_file_location(rel.replace('/', '_').removesuffix('.py'), rel)\n"
        "    mod = importlib.util.module_from_spec(spec)\n"
        "    sys.modules[spec.name] = mod\n"
        "    spec.loader.exec_module(mod)\n"
        "print('OK', len(sys.argv) - 1)\n"
    )
    imp = subprocess.run(
        [sys.executable, "-c", import_code, *test_files],
        cwd=REPO_ROOT, env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)),
        capture_output=True, text=True, check=True,
    )
    assert imp.stdout.strip().startswith("OK "), imp.stderr

    # Verify our own probe modules do not import bts.model.predict or lightgbm at module level
    with tempfile.TemporaryDirectory() as td:
        Path(td, "lightgbm.py").write_text("raise ImportError('no lightgbm')\n", encoding="utf-8")
        probe_files = [
            str(REVIEW_DIR / "test_probe_interactions.py"),
            str(REVIEW_DIR / "test_probe_clocks_and_years.py"),
            str(REVIEW_DIR / "test_probe_fail_closed.py"),
            str(REVIEW_DIR / "build_mutants.py"),
        ]
        guard_code = (
            "import importlib.util, sys\n"
            "for p in sys.argv[1:]:\n"
            "    spec = importlib.util.spec_from_file_location(p, p)\n"
            "    mod = importlib.util.module_from_spec(spec)\n"
            "    spec.loader.exec_module(mod)\n"
            "assert 'bts.model.predict' not in sys.modules, 'bts.model.predict imported at module level!'\n"
            "print('CLEAN_IMPORTS')\n"
        )
        res = subprocess.run(
            [sys.executable, "-c", guard_code, *probe_files],
            cwd=REPO_ROOT,
            env=dict(os.environ, PYTHONPATH=f"{td}:{REPO_ROOT}"),
            capture_output=True, text=True, check=True,
        )
        assert "CLEAN_IMPORTS" in res.stdout

    manifest_path = LOGS_DIR / "mutants_manifest.json"
    manifest_path.write_text(json.dumps(MUTANTS, indent=2), encoding="utf-8")
    return MUTANTS


if __name__ == "__main__":
    test_mutant_targets_are_unique_and_valid()
    built = materialize_all_mutants()
    print(f"build_mutants: {len(built)} mutants materialized and verified PASS")
