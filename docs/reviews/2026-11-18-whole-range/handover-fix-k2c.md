# K2 Amendment 7 handover

Commit `ef948b99` on `fix/wr-k2`, based on `82cd60d1`.

## K2 tests before and after

| Test | Base scheduler with updated tests | After the move |
| --- | --- | --- |
| `test_boundary_journal_follows_last_lineup_check_and_fallback` | Failed. The warning appeared before the first lineup check and the final fallback. | Passed. The recorded order was last lineup check, final fallback, boundary journal line, next-day lookahead. The delivery stub recorded the final fallback. |
| `test_shared_journal_with_690_bytes_free_preserves_restart_cutoff` | Failed. A pipe with 690 free bytes and a simulated two-second reader stall moved the first check from 18:59:59 to 19:00:01 ET. Game 123, scheduled for 19:05 ET, was unavailable. | Passed. The first check ran at 18:59:59 ET with no unavailable game IDs. One boundary line was written. |
| `test_qualifying_start_sends_no_dm_and_writes_no_receipt` (3 modes) | 3 passed | 3 passed |
| `test_configured_boundary_journal_counts_each_start` | Passed | Passed |
| `test_stalled_boundary_journal_preserves_first_check` (3 start times) | 3 passed | 3 passed |
| `test_fallback_warning_and_unclassifiable_rows` | Passed | Passed |
| `test_startup_without_warning` (8 cases) | 8 passed | 8 passed |

The journal fixture now completes the day where the schedule allows it. The stalled-writer cases previously stopped at the first check and now record its time and enterable games while continuing through the journal phase. All 52 existing K2 cases passed on both runs, including the unchanged floor cases.

## Matched test files after the last code change

| Test files | Batch result |
| --- | --- |
| `tests/test_wr_k2_season_boundary.py` | 54 passed (8.81s) |
| `tests/experiment/test_factored_runner_bitexact.py`, `tests/experiment/test_integration.py`, `tests/experiment/test_registry.py`, `tests/experiment/test_run_selection_feature_cols.py` | 25 passed, 5 skipped in 1.22s |
| `tests/experiment/test_runner.py`, `tests/experiment/test_runner_nested_split.py`, `tests/experiment/test_strategies.py`, `tests/model/test_calibrate.py` | 30 passed in 0.57s |
| `tests/scripts/season_walk/test_season_walk.py`, `tests/simulate/test_exact.py`, `tests/simulate/test_mdp.py`, `tests/simulate/test_mdp_save_atomic.py` | 61 passed in 1.28s |
| `tests/simulate/test_monte_carlo.py`, `tests/simulate/test_rare_event_mc.py`, `tests/simulate/test_strategies.py`, `tests/test_alert_status_callers_cs6b2.py` | 56 passed in 6.15s |
| `tests/test_alert_status_cs6b2_carbon.py`, `tests/test_audit_driver.py`, `tests/test_cli_integration.py`, `tests/test_cli_schedule_date.py` | 33 failed, 99 passed, 16 errors in 2.78s |
| `tests/test_cold_start_integration.py`, `tests/test_contest_dated_cs4a_review.py`, `tests/test_cutoff_candidate_exclusion.py`, `tests/test_decide_action.py` | 74 passed, 1 xfailed in 1.40s |
| `tests/test_incident_2026_08_30.py`, `tests/test_incident_register_2026.py`, `tests/test_live_gate_cs6a.py`, `tests/test_live_gate_cs6a_review.py` | 130 passed in 1.08s |
| `tests/test_morning_state_cs6b1.py`, `tests/test_morning_state_cs6b1_carbon.py`, `tests/test_morning_state_cs6b1_review.py`, `tests/test_orchestrator.py` | 108 passed in 1.48s |
| `tests/test_pickpath_review_pp1.py`, `tests/test_pickpath_review_pp2.py`, `tests/test_private_mode_transport.py`, `tests/test_saved_pick_cs5a_review.py` | 119 passed, 8 skipped in 1.30s |
| `tests/test_saved_pick_paths_cs5a.py`, `tests/test_saver_cs3.py`, `tests/test_saver_cs3_review.py`, `tests/test_scheduler.py` | 379 passed, 4 skipped in 2.65s |
| `tests/test_scheduler_decision_record_integration.py`, `tests/test_scheduler_eve_of_break.py`, `tests/test_scheduler_no_games.py`, `tests/test_scheduler_skip_visibility.py` | 74 passed in 0.34s |
| `tests/test_scheduler_state_integrity.py`, `tests/test_scheduler_tail_objective.py`, `tests/test_season_end_date.py`, `tests/test_shadow_scheduler.py` | 32 passed in 75.43s (0:01:15) |
| `tests/test_stale_delivery_cs4c.py`, `tests/test_stale_delivery_cs4c_carbon.py`, `tests/test_strategy.py`, `tests/test_streak_season_cs2.py` | 379 passed in 1.53s |
| `tests/test_streak_season_cs2_review.py`, `tests/test_streak_season_cs2b_review.py`, `tests/test_tail_policy_artifact_contract.py`, `tests/test_tail_policy_r3_fixes.py` | 173 passed in 0.69s |
| `tests/test_tail_policy_strategy.py`, `tests/test_thin_slate_checks.py`, `tests/test_wr_a_calendar.py`, `tests/test_wr_a_calendar_replay_carbon.py` | 61 passed in 1.08s |
| `tests/test_wr_a_pick.py`, `tests/test_wr_a_review_carbon.py`, `tests/test_wr_a_schedule.py`, `tests/test_wr_c1_history.py` | 121 passed in 5.42s |
| `tests/test_wr_c2_carbon_v2.py`, `tests/test_wr_c2_r2_carbon.py`, `tests/test_wr_c2_receipt_identity.py`, `tests/test_wr_c2_recovery.py` | 283 passed in 2.58s |
| `tests/test_wr_c3_r2_entered_pick_blockers.py`, `tests/test_wr_c3v_delivery.py`, `tests/test_wr_e_texts.py`, `tests/test_wr_f3_skip_state_review.py` | 175 passed in 32.76s |
| `tests/test_wr_g_guard.py`, `tests/test_wr_h2_caps.py`, `tests/test_wr_h_longest.py`, `tests/test_wr_k1_r2.py` | 192 passed in 19.46s |
| `tests/test_wr_k1_refusal_dm.py`, `tests/validate/test_rare_event_mc_eval.py` | 119 passed in 1.46s |

The 79 matched pytest files yielded 2,744 passed, 17 skipped, and one expected failure. All 33 failures and 16 setup errors were in `tests/test_audit_driver.py`. They are the base's GNU rsync import failures. The audit driver and its test file match `82cd60d1` byte for byte, and a standalone rerun reproduced all 49 affected cases. The other search hits were script sources, documentation, and Python bytecode. The full fast suite was outside this check.

The tests ran serially in ET with a 110-second timeout per invocation. The longest batch took 75.43 seconds. The OS declined the requested nice priority, and the sandbox process lookup could not determine whether `victoria3` was running. The two staged files passed `git diff --cached --check`.
