# F2b author handover

Commit `980f65ca` on `fix/wr-f2`. The three authorized files are in the single commit. The branch was not pushed.

| Row | Before the fixes on `a875f655` | After the fixes |
| --- | --- | --- |
| E1 | Failed. The first mismatch receipt replaced the attention state and removed the `sources` map. | Passed. The mismatch DM sends on July 15 and again on a same-day rerun. On July 16 the repeated calibration WARN sends with streak 2. |
| E2 | Failed for the dotted and relative spellings with the same loss of the `sources` map. Existing alias refusal preserved attention state for symlink and hard-link cases. | Passed for both spellings and both aliases. The second dispatch receives no receipt path in every collision case. The dispatch argument assertion was added after the first run. |
| E3 | Passed with the default receipt and same-day deduplication. | Passed with identical receipt bytes on rerun. |
| E4 | Passed with an attention override elsewhere. | Passed with the receipt and attention state stored separately. |
| E5 | Passed the existing delivery cases. | Passed both incident orders, same-day reruns, digest first and near the chat limit, consecutive days, post-midnight delivery, nameless override fallback, alias refusal, and retries after a blank transport ID. |
| E6 | Failed both the authorized exact-bullet assertion and the new dashboard replay after the status command showed both records as used. | Passed. The new wording is exact and the old sentence is absent. The replay reconciled the flag after the dashboard change and the next automatic read counted the saver as available. |
| E7 | Passed the B8a row checksum and existing live and historical message checks. | Passed the B8a row checksum after substituting only the authorized sentence. The added full-message comparisons passed in live and historical mode. |

The first F2 test run yielded 5 failures and 59 passes. The final run yielded 66 passes in that file. The only existing assertion edited was the F2-owned `test_b8a_saver_reconciliation_text` under Amendment 3 item 4. No assertion from base `a4e8278d` changed.

The earlier stop on the exact-bullet assertion was resolved by Amendment 3 item 4. One socket test in batch 6 hit the sandbox loopback restriction. The complete `tests/test_web_saver.py` file passed all 12 cases when rerun with the approved localhost bind. The sandbox declined the priority change in the other batches. The 110-second timeouts remained in force and all batches completed within 32 seconds.

## Regression test files

All 35 tracked source files returned by the brief's test search ran, along with the explicitly named dated-contest file and the remaining round-two indexed test files. Search hits under `__pycache__` are interpreter-generated copies of test files. The 81 distinct files yielded 5,548 passes, 12 skips, and one expected failure, counting the socket retry once.

### Batch 1 — 1,204 passed

- `tests/test_wr_f2_saver_flag.py`
- `tests/test_f2_b1_stat_oserror_regression.py`
- `tests/health/test_runner.py`
- `tests/health/test_alert.py`
- `tests/health/test_alert_status_cs6b2.py`
- `tests/test_alert_status_callers_cs6b2.py`
- `tests/test_wr_c1_history.py`
- `tests/test_contest_dated_cs4a.py`

### Batch 2 — 306 passed, 8 skipped

- `tests/health/test_alert_status_cs6b2_carbon_r2.py`
- `tests/health/test_h6_resurface.py`
- `tests/test_alert_status_cs6b2_carbon.py`
- `tests/test_contest_dated_cs4a_review.py`
- `tests/test_e3_missed_pick_alert.py`
- `tests/test_incident_2026_08_30.py`
- `tests/test_late_delivery_guard.py`
- `tests/test_morning_state_cs6b1.py`
- `tests/test_morning_state_cs6b1_review.py`
- `tests/test_pickpath_review_pp1.py`
- `tests/test_pickpath_review_pp2.py`
- `tests/test_postponed_delivery_guard.py`
- `tests/test_private_mode_transport.py`
- `tests/test_saver_cs3_review.py`

### Batch 3 — 716 passed

- `tests/test_scheduler.py`
- `tests/test_stale_delivery_cs4c.py`
- `tests/test_stale_delivery_cs4c_carbon.py`
- `tests/test_stale_delivery_cs4c_review.py`
- `tests/test_wr_b_contract.py`
- `tests/test_wr_b_edges.py`
- `tests/test_wr_c1_clock.py`
- `tests/test_wr_c2_r2_carbon.py`
- `tests/test_wr_c2_receipt_identity.py`
- `tests/test_wr_c2_recovery.py`
- `tests/test_wr_g_guard.py`
- `tests/test_wr_w15_health.py`
- `tests/test_wr_w15b.py`

### Batch 4 — 65 passed

- `tests/test_wr_e_texts.py`

### Batch 5 — 792 passed, 4 skipped

- `tests/health/test_alert_status_cs6b2_review.py`
- `tests/health/test_contest_state.py`
- `tests/scripts/test_state_segment_policy_candidate_screen.py`
- `tests/test_backup.py`
- `tests/test_backup_cli.py`
- `tests/test_contest_fetch.py`
- `tests/test_contest_state.py`
- `tests/test_contest_state_decoder_limits.py`
- `tests/test_decision_saver_fallback.py`
- `tests/test_health_endpoint.py`
- `tests/test_live_gate_cs6a_review.py`
- `tests/test_pick_chosen_on_cs5a.py`
- `tests/test_saved_pick_cs5a_review.py`
- `tests/test_saved_pick_paths_cs5a.py`
- `tests/test_saver_cs3.py`
- `tests/test_saver_state.py`
- `tests/test_season_contract_cs4b1.py`

### Batch 6 — 2,052 passed across the batch and retry

- `tests/test_season_contract_cs4b1_edges.py`
- `tests/test_season_contract_cs4b1_review.py`
- `tests/test_season_contract_cs4b2.py`
- `tests/test_season_contract_cs4b2_edges.py`
- `tests/test_season_contract_cs4b2_review.py`
- `tests/test_streak_season_cs2.py`
- `tests/test_streak_season_cs2_review.py`
- `tests/test_web_saver.py`
- `tests/test_wr_a_calendar.py`
- `tests/test_wr_b_r2_review_adopt.py`
- `tests/test_wr_c2_carbon_v2.py`
- `tests/test_wr_c3_r2_entered_pick_blockers.py`
- `tests/test_wr_h2_caps.py`
- `tests/validate/test_ope.py`
- `tests/health/test_f4_exception_propagation.py`

### Batch 7 — 413 passed, 1 expected failure

- `tests/health/test_late_delivery.py`
- `tests/test_cli_integration.py`
- `tests/test_cli_schedule_date.py`
- `tests/test_cold_start_integration.py`
- `tests/test_contest_state_defensive_reads.py`
- `tests/test_entry_intent.py`
- `tests/test_incident_register_2026.py`
- `tests/test_live_gate_cs6a.py`
- `tests/test_scheduler_decision_record_integration.py`
- `tests/test_scheduler_eve_of_break.py`
- `tests/test_thin_slate_checks.py`
- `tests/test_wr_a_pick.py`
- `tests/test_wr_a_review_carbon.py`
