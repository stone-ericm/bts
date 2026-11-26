# B2 + B3 handover
B2: run_day resets attempt_returned_pick before every iteration, including early continues; only the current executed check can set it. All protected functions are unchanged.
The fixture now writes account_state/contest_streak.json (auto schema/source, active/best 57, source_date and recorded_at in 2027) and pins contest_state.datetime alongside the other clocks. No test seeds streak.json; the skip test asserts contest provenance and absence of a model-streak file.
B2 baseline: 10 passed. RED: 3 distinct cases (contest provenance, refused/archived pick followed by an early extra sleep, no prior executed check/NameError); the fixture also failed with companion 66f3992 before correction. Final companion scratch tree (66f3992 plus our scheduler/tests, PYTHONPATH=src): 12 passed.
Mutations: reverting the generator fix failed both early-sleep tests; reverting the wording failed both B3 contract tests. Source restored byte-for-byte each time; restored wording/incident checks: 2 passed.
B3: the empty-day CRITICAL missed_pick now says: "Today's checks and remaining attempts are over. No pick is on file for today." Same transport, source, level, and one day-end alert.
The incident test searches every dispatch, preserves no pick DM, CRITICAL late_delivery, archive, unlocked state, Steven Kwan and stderr assertions, and additionally requires exactly one later day-end dispatch with truthful wording. Added the wording transport test; updated our two wording assertions. No other existing tests changed or failed.
B3 RED: original incident test failed once; strengthened incident and new wording test both failed before the message change (2 RED).
Operator order after a late refusal and no later selection: CRITICAL late_delivery refusal notice first; after remaining attempts, one CRITICAL missed_pick day-end notice; no pick DM.

First-handover focused command (B2 result: 287 passed, 8 skipped):
`TZ=America/New_York .venv/bin/python -m pytest tests/test_thin_slate_checks.py tests/test_late_delivery_guard.py tests/test_postponed_delivery_guard.py tests/test_pickpath_review_pp1.py tests/test_pickpath_review_pp2.py tests/test_incident_register_2026.py tests/test_scheduler.py tests/test_scheduler_decision_record_integration.py tests/test_cold_start_integration.py tests/test_scheduler_last_attempt.py -q -p no:cacheprovider`
Expanded discovery used the brief's literal pattern with rg over Python files. Ran every match in two batches using the same command form: 560 passed/8 skipped plus 63 passed/1 skipped = 623 passed, 9 skipped, no failures. Exact commands: b3-command.txt and b3-additional-command.txt beside this handover; results: b3-focused.xml and b3-additional.xml.
Full per-file counts below are passed/skipped (conftest has no test items):
tests/conftest.py 0/0; tests/health/test_analytics_artifacts_missing.py 13/0; tests/health/test_runner.py 21/0
tests/health/test_unit_drift.py 9/0; tests/scripts/incident_register/test_deploy_runs.py 17/0; tests/scripts/incident_register/test_records.py 0/1
tests/scripts/season_ledger/test_acquire.py 2/0; tests/test_cli_schedule_date.py 1/0; tests/test_cold_start_integration.py 4/0
tests/test_cutoff_candidate_exclusion.py 9/0; tests/test_day_outcome.py 56/0; tests/test_day_outcome_entry_sequence.py 35/0
tests/test_e2_delivery_idempotency.py 3/0; tests/test_e3_missed_pick_alert.py 8/0; tests/test_fallback_plan.py 13/0
tests/test_incident_2026_08_30.py 4/0; tests/test_incident_register_2026.py 62/0; tests/test_late_delivery_guard.py 8/0
tests/test_lock_decision.py 7/0; tests/test_london_series_tz.py 4/0; tests/test_pickpath_review_pp1.py 4/0
tests/test_pickpath_review_pp2.py 59/8; tests/test_postponed_delivery_guard.py 24/0; tests/test_private_mode_transport.py 8/0
tests/test_scheduler.py 76/0; tests/test_scheduler_decision_record.py 9/0; tests/test_scheduler_decision_record_integration.py 24/0
tests/test_scheduler_eve_of_break.py 10/0; tests/test_scheduler_fallback_deadline.py 7/0; tests/test_scheduler_last_attempt.py 12/0
tests/test_scheduler_no_games.py 1/0; tests/test_scheduler_skip_visibility.py 39/0; tests/test_scheduler_state_integrity.py 7/0
tests/test_scheduler_tail_objective.py 9/0; tests/test_scheduler_watchdog_discipline.py 1/0; tests/test_scoring_lock.py 7/0
tests/test_shadow_predict_once.py 4/0; tests/test_shadow_scheduler.py 8/0; tests/test_submission_cutoff.py 5/0
tests/test_tail_policy_r3_fixes.py 19/0; tests/test_thin_slate_checks.py 14/0
Skips: eight intentional private-mode parameter cases; test_records collection requires unavailable jsonschema.

git diff --stat 75994d3:
```text
 src/bts/scheduler.py                 |  6 ++-
 tests/test_e3_missed_pick_alert.py   | 10 +++++
 tests/test_incident_2026_08_30.py    | 10 ++++-
 tests/test_scheduler_last_attempt.py | 71 ++++++++++++++++++++++++++++++------
 4 files changed, 82 insertions(+), 15 deletions(-)
```
git diff --check passed. No commit, push, other-worktree edits, or production account actions.
