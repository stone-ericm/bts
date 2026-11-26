#!/bin/bash
set -u
output=.codex-review/wr-f5b-r1
export TZ=America/New_York PYTHONDONTWRITEBYTECODE=1
run() {
    label=$1
    shift
    nice -n 19 timeout 110 .venv/bin/python -m pytest -p no:cacheprovider -q "$@" > "$output/$label.log" 2>&1
    status=$?
    printf '%s %s\n' "$label" "$status"
}
run existing-intent-lock tests/test_entry_intent.py tests/test_wr_f1b_lock.py
run existing-picks tests/test_pick_chosen_on_cs5a.py tests/test_saved_pick_paths_cs5a.py tests/test_scheduler_decision_record_integration.py tests/test_submission_cutoff.py
run existing-readers tests/test_entry_receipt.py tests/test_day_outcome.py tests/test_day_outcome_entry_sequence.py tests/health/test_attention.py tests/health/test_pick_entry_source.py
run existing-entry tests/test_contest_fetch.py tests/test_cli_integration.py tests/test_wr_f1_review.py tests/test_automatic_reading_cs4c.py tests/test_live_gate_cs6a_review.py tests/leaderboard/test_scraper_stealth.py
run existing-policy tests/test_wr_a_review_carbon.py tests/test_wr_a_schedule.py tests/test_wr_b_contract.py tests/test_wr_b_edges.py tests/test_wr_c2_carbon_v2.py tests/test_wr_c2_r2_carbon.py
run existing-ops tests/test_audit_f6_review_probes.py tests/test_season_contract_cs4b2_edges.py tests/test_wr_f2_saver_flag.py tests/test_wr_f3_skip_state_review.py tests/test_wr_f4_streak_file.py tests/scripts/season_ledger/test_contest.py
run existing-audit tests/test_audit_driver.py
