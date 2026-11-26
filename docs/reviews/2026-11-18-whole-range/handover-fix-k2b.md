# K2b author handover

Commit `0697110b` on `fix/wr-k2`, based on `153f1354`.

A qualifying start queues the R74 line for a daemon writer using `os.write(2, …)`. The boundary DM, dispatch worker and receipt path are gone. The scheduler still calls the boundary check at the same point and supplies its existing positional arguments. The check only uses the date and games. The strategy file matched `6f0ab421` byte for byte and the health directory diff was empty.

## Tests before and after

| Test | Before scheduler edit | After |
| --- | --- | --- |
| Qualifying start in private, DM and public modes | All three failed with a boundary DM and a receipt | All three passed with zero sends and no receipt |
| Configured boundary journal on repeat starts | Failed on the missing R74 advisory | Passed with exact lines and counts 1, 2, 1 |
| Unconfigured boundary journal with unclassifiable rows | Failed on the missing R74 advisory | Passed with the exact line and count 1 |
| Real morning refusal at the boundary | Failed with an extra boundary DM after the expected refusal | Passed with exactly the archived base refusal |
| Stalled writer at the 10:05 and 10:10 thin-slate starts | Both passed at 10:15 ET with the game enterable | Both passed at 10:15 ET with the game enterable |
| Stalled writer at the 18:59:59 restart | Passed at 18:59:59 ET with the 19:05 game enterable | Passed at 18:59:59 ET with the game enterable |
| Startup without a warning, using descriptor capture | All eight passed | All eight passed |

The existing floor cases E1–E5 and E7, saved decision schema, raw health count, and 373,188 positive-day decision and display-bar comparisons passed before and after. The final K2 file passed all 52 tests.

## Removed cases

- The restart and count-change DM test and the DM wording test previously asserted the send count and body. The configured journal test covers the restart counts and proves a same-date final counts while postponed and canceled games do not. The new delivery-mode test covers the absence of a send and receipt.
- The old journal wording test and check-already-due reason test were superseded by the exact R74 lines and the stalled-writer restart case.
- Unconfirmed send retries, invalid receipt handling, and date-keyed receipt behavior were removed with the boundary receipt and dispatch.
- The fresh-clock lead test and the blocked transport deadline test applied to the removed lead and deadline checks. The real morning refusal test retains the comparison to the archived base refusal. The stalled-journal tests now wait on the writer directly.

## Rendered journal lines

```text
SEASON BOUNDARY BEHIND SCHEDULE — 2027-09-27 ET: 1 regular-season game(s) with that official date and a reported status other than postponed or canceled. The configured exclusive boundary is 2027-09-27, giving a raw count of zero. A fresh pick selection with an available predicted candidate uses the last-day policy. Ask the BTS lead session or Eric to check MLB's last regular-season game date, including pending makeup games, and to update the stored boundary for a later run.
SEASON BOUNDARY BEHIND SCHEDULE — 2027-09-27 ET: 2 regular-season game(s) with that official date and a reported status other than postponed or canceled. The configured exclusive boundary is 2027-09-27, giving a raw count of zero. A fresh pick selection with an available predicted candidate uses the last-day policy. Ask the BTS lead session or Eric to check MLB's last regular-season game date, including pending makeup games, and to update the stored boundary for a later run.
SEASON BOUNDARY BEHIND SCHEDULE — 2028-09-29 ET: 1 regular-season game(s) with that official date and a reported status other than postponed or canceled. The fallback exclusive boundary for the unconfigured year 2028 is 2028-09-28, giving a raw count of zero. A fresh pick selection with an available predicted candidate uses the last-day policy. Ask the BTS lead session or Eric to check MLB's last regular-season game date, including pending makeup games, and to update the stored boundary for a later run.
```

## Regression batches

The 86 search-selected test files ran serially after the final code and test edits, with ET pinned and a 110-second timeout per batch. The longest batch took 75.33 seconds. Victoria 3 was absent at each batch check. macOS declined the requested priority change and the tests ran. Total: 2,815 passed, 17 skipped and one expected failure. The audit-driver batch had 33 failures and 16 setup errors from missing GNU rsync, matching the [archived base run](docs/reviews/2026-11-18-whole-range/review-fix-k2-codex/tests-base-audit.log).

| Batch | Files | Result |
| --- | --- | --- |
| 1 | `tests/test_wr_k2_season_boundary.py` | 52 passed in 8.72s |
| 2 | `tests/test_audit_driver.py` | 33 failed, 16 errors in 1.28s |
| 3 | `tests/test_wr_h_longest.py` | 90 passed in 14.49s |
| 4 | `tests/test_wr_e_texts.py` | 65 passed in 31.37s |
| 5 | `tests/test_shadow_scheduler.py` | 8 passed in 75.33s (0:01:15) |
| 6 | `tests/experiment/test_factored_runner_bitexact.py`, `tests/experiment/test_integration.py`, `tests/experiment/test_registry.py`, `tests/experiment/test_run_selection_feature_cols.py`, `tests/experiment/test_runner.py`, `tests/experiment/test_runner_nested_split.py`, `tests/experiment/test_strategies.py` | 42 passed, 5 skipped in 0.65s |
| 7 | `tests/health/test_realized_calibration.py`, `tests/model/test_calibrate.py`, `tests/scripts/season_walk/test_season_walk.py`, `tests/scripts/test_bin_collapse_read.py`, `tests/scripts/test_leaderboard_mechanism_mining.py`, `tests/scripts/test_mining87_inference.py`, `tests/scripts/test_mining87_production.py` | 110 passed in 1.37s |
| 8 | `tests/scripts/test_mining87_units.py`, `tests/scripts/test_p05_tail_policy_audit.py`, `tests/simulate/test_exact.py`, `tests/simulate/test_mdp.py`, `tests/simulate/test_mdp_save_atomic.py`, `tests/simulate/test_monte_carlo.py`, `tests/simulate/test_rare_event_mc.py` | 68 passed in 6.74s |
| 9 | `tests/simulate/test_strategies.py`, `tests/test_alert_status_callers_cs6b2.py`, `tests/test_alert_status_cs6b2_carbon.py`, `tests/test_cli_integration.py`, `tests/test_cli_schedule_date.py`, `tests/test_cold_start_integration.py`, `tests/test_contest_dated_cs4a_review.py` | 183 passed, 1 xfailed in 2.15s |
| 10 | `tests/test_cutoff_candidate_exclusion.py`, `tests/test_decide_action.py`, `tests/test_incident_2026_08_30.py`, `tests/test_incident_register_2026.py`, `tests/test_live_gate_cs6a.py`, `tests/test_live_gate_cs6a_review.py`, `tests/test_morning_state_cs6b1.py` | 213 passed in 1.54s |
| 11 | `tests/test_morning_state_cs6b1_carbon.py`, `tests/test_morning_state_cs6b1_review.py`, `tests/test_orchestrator.py`, `tests/test_pickpath_review_pp1.py`, `tests/test_pickpath_review_pp2.py`, `tests/test_private_mode_transport.py`, `tests/test_saved_pick_cs5a_review.py` | 166 passed, 8 skipped in 1.44s |
| 12 | `tests/test_saved_pick_paths_cs5a.py`, `tests/test_saver_cs3.py`, `tests/test_saver_cs3_review.py`, `tests/test_scheduler.py`, `tests/test_scheduler_decision_record_integration.py`, `tests/test_scheduler_eve_of_break.py`, `tests/test_scheduler_no_games.py` | 414 passed, 4 skipped in 2.64s |
| 13 | `tests/test_scheduler_skip_visibility.py`, `tests/test_scheduler_state_integrity.py`, `tests/test_scheduler_tail_objective.py`, `tests/test_season_end_date.py`, `tests/test_skip_policy_shadow.py`, `tests/test_stale_delivery_cs4c.py`, `tests/test_stale_delivery_cs4c_carbon.py` | 180 passed in 1.21s |
| 14 | `tests/test_strategy.py`, `tests/test_streak_season_cs2.py`, `tests/test_streak_season_cs2_review.py`, `tests/test_streak_season_cs2b_review.py`, `tests/test_tail_policy_artifact_contract.py`, `tests/test_tail_policy_r3_fixes.py`, `tests/test_tail_policy_strategy.py` | 495 passed in 1.19s |
| 15 | `tests/test_thin_slate_checks.py`, `tests/test_wr_a_calendar.py`, `tests/test_wr_a_calendar_replay_carbon.py`, `tests/test_wr_a_pick.py`, `tests/test_wr_a_review_carbon.py`, `tests/test_wr_a_schedule.py`, `tests/test_wr_c1_history.py` | 157 passed in 5.22s |
| 16 | `tests/test_wr_c2_carbon_v2.py`, `tests/test_wr_c2_r2_carbon.py`, `tests/test_wr_c2_receipt_identity.py`, `tests/test_wr_c2_recovery.py`, `tests/test_wr_c3_r2_entered_pick_blockers.py`, `tests/test_wr_c3v_delivery.py`, `tests/test_wr_f3_skip_state_review.py` | 393 passed in 3.10s |
| 17 | `tests/test_wr_g_guard.py`, `tests/test_wr_h2_caps.py`, `tests/test_wr_k1_refusal_dm.py`, `tests/validate/test_rare_event_mc_eval.py` | 179 passed in 4.61s |

All 20 matched Python scripts passed syntax parsing.

- `scripts/audit/boundary_shadow_census.py`
- `scripts/audit/bullpen_r52_compare.py`
- `scripts/audit/c1_r4b/replay.py`
- `scripts/audit/cold_start/rehearse.py`
- `scripts/audit/dd_p_policy_value_sensitivity.py`
- `scripts/audit/mining87/production.py`
- `scripts/audit/mining87/registration.py`
- `scripts/audit/mining87/units.py`
- `scripts/audit/season_walk/walk.py`
- `scripts/audit_driver.py`
- `scripts/audit_mdp_live_aggressiveness.py`
- `scripts/canonicalize_realized_picks.py`
- `scripts/diff_audit_outputs.py`
- `scripts/dr_mdp_gap_measure.py`
- `scripts/gate_b_production_metric_rebaseline.py`
- `scripts/leaderboard_mechanism_mining.py`
- `scripts/rebuild_tail_policy.py`
- `scripts/run_falsification_harness.py`
- `scripts/run_realized_picks_fdr.py`
- `scripts/strategy_grid_search.py`

`scripts/cron-setup.sh` passed `bash -n`. The search also matched `scripts/audit/season_walk/README.md`. The season walk ran through its selected tests.

## Handover status

Staging and the single commit received separate approvals. No scope-related stop was needed. The journal still shares descriptor 2 with ordinary scheduler output, and the writer can drop a line if a nonblocking sink returns EAGAIN.
