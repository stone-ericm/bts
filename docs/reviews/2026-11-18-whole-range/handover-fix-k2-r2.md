# K2 round 2 handover

Commit `6f0ab421` on `fix/wr-k2`, based on Amendment 4 at `70d14651`. Staging and the commit received separate approvals.

The scheduler enqueues its warning lines. A daemon writes those lines to descriptor 2. A separate daemon runs eligible dispatches, and the scheduler joins only that daemon within the existing deadline. The deadline assertion still verifies that the journal line appears while transport is blocked. The tests coordinate asynchronous output with bounded waits.

## New tests before and after

| Case | Current HEAD before production changes | After |
| --- | --- | --- |
| Thin-slate first check with a stalled warning journal | Failed at 10:17 ET | Passed at the planned 10:15 ET with the game available |
| Restart just before the entry cutoff | Failed at 19:00:01 ET with the 19:05 game unavailable | Passed at 18:59:59 ET with the game still enterable |
| DM reported-status sentence | Failed on “eligible regular-season game” | Passed with the prescribed wording |
| Journal reported-status count | Failed on “1 eligible regular-season game(s)” | Passed with the prescribed wording |
| Journal reason for a check due now | Failed on “due within 10 minutes” | Passed with the prescribed wording |
| 2027-09-27 raw-zero loader regression | Passed with a degraded single | Passed with a degraded single |

The two stalled-journal tests model backpressure at the warning write, with a pinned ET clock and bounded event waits. A scratch copy of `src` removed only the early `min_days=1` argument. The raw-zero case failed with the escaping `TypeError: broken loader`, and the existing September 26 case passed. The scratch directory was cleaned up.

## Captured warning text

```text
BTS health WARN attention:
- [season_boundary] Season boundary behind the schedule for 2027-09-27 ET. At scheduler startup MLB listed at least one regular-season game with that official date and a reported status other than postponed or canceled. The configured exclusive boundary is 2027-09-27, giving a raw count of zero. With the raw count at zero, a fresh pick selection with an available predicted candidate uses the last-day policy. Please send this advisory to the BTS lead or Eric. Check the last regular-season game date on MLB's updated schedule, including pending makeup games, before arranging an update for a later run. Keep any pick already sent or entered as recorded.
BTS health WARN attention:
- [season_boundary] Season boundary behind the schedule for 2028-09-29 ET. At scheduler startup MLB listed at least one regular-season game with that official date and a reported status other than postponed or canceled. The fallback exclusive boundary for the unconfigured year 2028 is 2028-09-28, giving a raw count of zero. With the raw count at zero, a fresh pick selection with an available predicted candidate uses the last-day policy. Please send this advisory to the BTS lead or Eric. Check the last regular-season game date on MLB's updated schedule, including pending makeup games, before arranging an update for a later run. Keep any pick already sent or entered as recorded.
```

```text
SEASON BOUNDARY BEHIND SCHEDULE — 2027-09-27 ET: 1 regular-season game(s) with that official date and a reported status other than postponed or canceled. The configured exclusive boundary is 2027-09-27, giving a raw count of zero.
SEASON BOUNDARY BEHIND SCHEDULE — 2027-09-27 ET: 1 regular-season game(s) with that official date and a reported status other than postponed or canceled. The configured exclusive boundary is 2027-09-27, giving a raw count of zero. Season boundary warning not sent: The first planned lineup check is due now or has already passed.
SEASON BOUNDARY BEHIND SCHEDULE — 2028-09-29 ET: 1 regular-season game(s) with that official date and a reported status other than postponed or canceled. The fallback exclusive boundary for the unconfigured year 2028 is 2028-09-28, giving a raw count of zero.
SEASON BOUNDARY BEHIND SCHEDULE — 2028-09-29 ET: 1 regular-season game(s) with that official date and a reported status other than postponed or canceled. The fallback exclusive boundary for the unconfigured year 2028 is 2028-09-28, giving a raw count of zero. Season boundary warning not sent: The first planned lineup check is due now or has already passed.
```

The two `CLAUDE.md` sentences read “Fresh selection uses at least one policy day once it has a primary candidate, even when more than one makeup day remains.” and “Standalone policy calls and health checks use the raw day count from the configured or fallback exclusive boundary.”

## Regression batches after the last edit

All 86 search-selected pytest files ran serially with a 110-second bound per batch and a pinned ET timezone. There were 2,827 passes, 17 skips and one expected failure. The audit driver reproduced 33 failures and 16 setup errors for missing GNU rsync. The round-1 review recorded the same results at the base. The OS denied the requested priority reduction and the commands continued. The sandbox could not inspect the `victoria3` process list. The 20 matched Python script files passed syntax parsing, and the matched shell script passed `bash -n`. The matched season-walk README is documentation. The season walk ran through its selected tests.

| Batch | Files | Result |
| --- | --- | --- |
| 1 | `tests/test_audit_driver.py` | 33 failed, 16 errors in 1.38s |
| 2 | `tests/test_wr_h_longest.py` | 90 passed in 14.88s |
| 3 | `tests/test_wr_k2_season_boundary.py` | 64 passed in 8.72s |
| 4 | `tests/test_shadow_scheduler.py` | 8 passed in 75.35s (0:01:15) |
| 5 | `tests/test_wr_k1_refusal_dm.py` | 97 passed in 1.05s |
| 6 | `tests/test_scheduler_no_games.py` | 1 passed in 0.20s |
| 7 | `tests/test_scheduler_state_integrity.py` | 7 passed in 0.20s |
| 8 | `tests/test_wr_c3v_delivery.py` | 9 passed in 0.99s |
| 9 | `tests/test_wr_f3_skip_state_review.py` | 92 passed in 0.64s |
| 10 | `tests/validate/test_rare_event_mc_eval.py` | 22 passed in 0.58s |
| 11 | `tests/experiment/test_factored_runner_bitexact.py`, `tests/experiment/test_integration.py`, `tests/experiment/test_registry.py`, `tests/experiment/test_run_selection_feature_cols.py`, `tests/experiment/test_runner.py`, `tests/experiment/test_runner_nested_split.py`, `tests/experiment/test_strategies.py` | 42 passed, 5 skipped in 0.58s |
| 12 | `tests/health/test_realized_calibration.py`, `tests/model/test_calibrate.py`, `tests/scripts/season_walk/test_season_walk.py`, `tests/scripts/test_bin_collapse_read.py`, `tests/scripts/test_leaderboard_mechanism_mining.py`, `tests/scripts/test_mining87_inference.py`, `tests/scripts/test_mining87_production.py` | 110 passed in 1.46s |
| 13 | `tests/scripts/test_mining87_units.py`, `tests/scripts/test_p05_tail_policy_audit.py`, `tests/simulate/test_exact.py`, `tests/simulate/test_mdp.py`, `tests/simulate/test_mdp_save_atomic.py`, `tests/simulate/test_monte_carlo.py`, `tests/simulate/test_rare_event_mc.py` | 68 passed in 6.80s |
| 14 | `tests/simulate/test_strategies.py`, `tests/test_alert_status_callers_cs6b2.py`, `tests/test_alert_status_cs6b2_carbon.py`, `tests/test_cli_integration.py`, `tests/test_cli_schedule_date.py`, `tests/test_cold_start_integration.py`, `tests/test_contest_dated_cs4a_review.py` | 183 passed, 1 xfailed in 2.25s |
| 15 | `tests/test_cutoff_candidate_exclusion.py`, `tests/test_decide_action.py`, `tests/test_incident_2026_08_30.py`, `tests/test_incident_register_2026.py`, `tests/test_live_gate_cs6a.py`, `tests/test_live_gate_cs6a_review.py`, `tests/test_morning_state_cs6b1.py` | 213 passed in 1.63s |
| 16 | `tests/test_morning_state_cs6b1_carbon.py`, `tests/test_morning_state_cs6b1_review.py`, `tests/test_orchestrator.py`, `tests/test_pickpath_review_pp1.py`, `tests/test_pickpath_review_pp2.py`, `tests/test_private_mode_transport.py`, `tests/test_saved_pick_cs5a_review.py` | 166 passed, 8 skipped in 1.51s |
| 17 | `tests/test_saved_pick_paths_cs5a.py`, `tests/test_saver_cs3.py`, `tests/test_saver_cs3_review.py`, `tests/test_scheduler.py`, `tests/test_scheduler_decision_record_integration.py`, `tests/test_scheduler_eve_of_break.py`, `tests/test_scheduler_skip_visibility.py` | 452 passed, 4 skipped in 2.83s |
| 18 | `tests/test_scheduler_tail_objective.py`, `tests/test_season_end_date.py`, `tests/test_skip_policy_shadow.py`, `tests/test_stale_delivery_cs4c.py`, `tests/test_stale_delivery_cs4c_carbon.py`, `tests/test_strategy.py`, `tests/test_streak_season_cs2.py` | 431 passed in 1.66s |
| 19 | `tests/test_streak_season_cs2_review.py`, `tests/test_streak_season_cs2b_review.py`, `tests/test_tail_policy_artifact_contract.py`, `tests/test_tail_policy_r3_fixes.py`, `tests/test_tail_policy_strategy.py`, `tests/test_thin_slate_checks.py`, `tests/test_wr_a_calendar.py` | 220 passed in 1.49s |
| 20 | `tests/test_wr_a_calendar_replay_carbon.py`, `tests/test_wr_a_pick.py`, `tests/test_wr_a_review_carbon.py`, `tests/test_wr_a_schedule.py`, `tests/test_wr_c1_history.py`, `tests/test_wr_c2_carbon_v2.py`, `tests/test_wr_c2_r2_carbon.py` | 174 passed in 5.77s |
| 21 | `tests/test_wr_c2_receipt_identity.py`, `tests/test_wr_c2_recovery.py`, `tests/test_wr_c3_r2_entered_pick_blockers.py`, `tests/test_wr_g_guard.py`, `tests/test_wr_h2_caps.py` | 313 passed in 5.84s |
| 22 | `tests/test_wr_e_texts.py` | 65 passed in 31.43s |

## Known limits

The queued journal bytes can still fill the shared pipe and delay a later scheduler line. The shared dispatcher's receipt or transport failure line still uses `sys.stderr` on the dispatch thread, where a stalled consumer can hold the stderr writer lock. A dispatch can finish after its deadline, and an unconfirmed send can repeat. Changing the stored boundary after an acknowledgment can send one additional warning on that slate date when the body changes.
