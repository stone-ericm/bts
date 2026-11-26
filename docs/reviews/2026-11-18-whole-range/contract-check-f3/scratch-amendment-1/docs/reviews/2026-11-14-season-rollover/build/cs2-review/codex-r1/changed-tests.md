# Existing changed tests, compared individually

`audit_tests.py` compares original and current ASTs and all assert statements in order. All 848 existing assert statements are identical. No marker, expected exception, oracle, helper implementation, or test body was weakened. The only non-fixture changes are the named missing-marker test rename and the valid-date loop below.

## tests/test_check_results_wait.py

- L127 `TestShadowReconcileOnEveryExit.test_production_pending_still_reconciles_shadow`: season fixture only; 5 → 5 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L148 `TestShadowReconcileOnEveryExit.test_production_resolver_error_still_reconciles_shadow`: season fixture only; 3 → 3 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L174 `TestStaleScoringGuard.test_old_unresolved_scoreable_production_never_scores`: season fixture only; 5 → 5 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L199 `TestStaleScoringGuard.test_allow_stale_scoring_override_scores`: season fixture only; 3 → 3 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L217 `TestStaleScoringGuard.test_yesterday_scores_without_override`: season fixture only; 3 → 3 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L237 `TestWaitDeadline.test_wait_retries_until_production_resolves`: season fixture only; 4 → 4 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L256 `TestWaitDeadline.test_deadline_is_a_hard_bound`: season fixture only; 4 → 4 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L277 `TestWaitDeadline.test_default_is_single_attempt_no_sleep`: season fixture only; 3 → 3 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L296 `TestWaitDeadline.test_wait_already_past_deadline_single_attempt`: season fixture only; 4 → 4 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L332 `TestWaitDeadline.test_wait_settles_immediately_when_already_resolved`: season fixture only; 3 → 3 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L364 `TestWaitDeadline.test_vanished_pick_file_stays_pending`: season fixture only; 5 → 5 assertions. Preserves original subject; assertions unchanged and non-vacuous.

## tests/test_cli_integration.py

- L187 `TestBtsRun.test_run_uses_fresh_contest_streak_for_live_action`: season fixture only; 4 → 4 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L228 `TestBtsPreview.test_preview_uses_fresh_contest_streak_for_projected_pick`: season fixture only; 5 → 5 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L316 `TestBtsCheckResults.test_check_results_hit_updates_streak`: season fixture only; 3 → 3 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L336 `TestBtsCheckResults.test_check_results_miss_resets_streak`: season fixture only; 3 → 3 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L356 `TestBtsCheckResults.test_check_results_none_warns_scratched`: season fixture only; 3 → 3 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L379 `TestBtsCheckResults.test_check_results_voids_primary_and_scores_double_once`: season fixture only; 6 → 6 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L418 `TestBtsCheckResults.test_check_results_voids_double_and_scores_primary_once`: season fixture only; 6 → 6 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L457 `TestBtsCheckResults.test_check_results_all_void_keeps_streak`: season fixture only; 6 → 6 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L493 `TestBtsCheckResults.test_check_results_skips_already_resolved`: season fixture only; 3 → 3 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L516 `TestBtsCheckResults.test_check_results_resolves_shadow_when_production_already_resolved`: season fixture only; 5 → 5 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L556 `TestBtsCheckResults.test_check_results_writes_shadow_status_artifact`: season fixture only; 5 → 5 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L591 `TestBtsCheckResults.test_check_results_shadow_status_failure_does_not_crash`: season fixture only; 3 → 3 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L624 `TestBtsCheckResults.test_check_results_skips_already_resolved_shadow`: season fixture only; 3 → 3 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L654 `TestBtsCheckResults.test_check_results_resolves_shadow_with_unresolved_production`: season fixture only; 4 → 4 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L687 `TestBtsCheckResults.test_check_results_shadow_double_down_is_dd_aware`: season fixture only; 3 → 3 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L769 `TestBtsCheckResults.test_check_results_skips_unscoreable_skip_record`: season fixture only; 3 → 3 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L783 `TestBtsCheckResults.test_check_results_scores_scoreable_decision`: season fixture only; 1 → 1 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L794 `TestBtsCheckResults.test_check_results_missing_decision_falls_back_to_delivered`: season fixture only; 1 → 1 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L803 `TestBtsCheckResults.test_check_results_missing_decision_undelivered_not_scored`: season fixture only; 1 → 1 assertions. Preserves original subject; assertions unchanged and non-vacuous.

## tests/test_contest_state.py

- L16 `test_model_state_used_when_no_contest_state`: season fixture only; 4 → 4 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L34 `test_required_contest_state_missing_raises`: season fixture only; 1 → 1 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L51 `test_fresh_contest_state_drives_live_decision`: season fixture only; 5 → 5 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L79 `test_lagged_contest_uses_contest_value_keeps_doubles`: season fixture only; 4 → 4 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L97 `test_lagged_contest_model_does_not_inflate`: season fixture only; 4 → 4 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L195 `test_missing_source_date_is_stale_but_keeps_doubles`: season fixture only; 3 → 3 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L208 `test_model_never_inflates_decision_streak`: season fixture only; 4 → 4 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L228 `test_two_pick_gap_is_stale`: season fixture only; 1 → 1 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L241 `test_unconfirmed_local_miss_marks_stale`: season fixture only; 2 → 2 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L254 `test_unexpired_override_wins_over_auto_through_decision`: season fixture only; 3 → 3 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L275 `test_saver_conservative_when_reached_zone_but_unconfirmed`: season fixture only; 2 → 2 assertions. Preserves original subject; assertions unchanged and non-vacuous.

## tests/test_incident_register_2026.py

- L259 `test_l01_check_results_single_pick`: season fixture only; 3 → 3 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L281 `test_l01_check_results_hit_plus_pass`: season fixture only; 2 → 2 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L298 `test_l01_check_results_pass_plus_pass_preserves`: season fixture only; 2 → 2 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L314 `test_l01_pass_at_saver_streak_keeps_the_saver`: season fixture only; 2 → 2 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L329 `test_l01_absent_player_stays_pending`: season fixture only; 3 → 3 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L382 `test_l02_same_day_hit_survives_two_late_reconciles`: season fixture only; 0 → 0 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L393 `test_l02_same_day_saver_consumption_survives_two_late_reconciles`: season fixture only; 0 → 0 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L404 `test_l02_unplayed_undelivered_preview_is_excluded`: season fixture only; 1 → 1 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L460 `test_l04_hit_applied_once_across_a_crash_and_restart`: season fixture only; 0 → 0 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L471 `test_l04_saver_miss_applied_once_across_a_crash_and_restart`: season fixture only; 0 → 0 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L483 `test_l04_polling_death_then_the_cron_scorer_applies_once`: season fixture only; 4 → 4 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L513 `test_l04_control_without_the_fault_a_rerun_is_a_no_op`: season fixture only; 3 → 3 assertions. Preserves original subject; assertions unchanged and non-vacuous.

## tests/test_pickpath_review_pp1.py

- L23 `test_crash_reconcile_restart_keeps_applied_hit`: season fixture only; 6 → 6 assertions. Preserves original subject; assertions unchanged and non-vacuous.

## tests/test_pickpath_review_pp2.py

- L239 `test_delivered_hit_plus_void_double_adds_exactly_one`: season fixture only; 7 → 7 assertions. Preserves original subject; assertions unchanged and non-vacuous.

## tests/test_picks.py

- L665 `TestReconcileResults.test_no_corrections_when_results_match`: season fixture only; 1 → 1 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L688 `TestReconcileResults.test_corrects_hit_to_miss`: season fixture only; 4 → 4 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L715 `TestReconcileResults.test_preview_pick_for_tomorrow_does_not_break_streak_walk`: season fixture only; 1 → 1 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L771 `TestReconcileResults.test_partial_void_hit_counts_as_single_streak_day`: season fixture only; 2 → 2 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L803 `TestReconcileResults.test_shadow_pick_files_are_ignored_in_streak_walk`: season fixture only; 1 → 1 assertions. Preserves original subject; assertions unchanged and non-vacuous.

## tests/test_reconcile_cutoff.py

- L24 `_plant_hit`: season fixture only; 0 → 0 assertions. Preserves original subject; assertions unchanged and non-vacuous.

## tests/test_reconcile_receipt.py

- L36 `plant`: season fixture only; 0 → 0 assertions. Preserves original subject; assertions unchanged and non-vacuous.

## tests/test_scheduler.py

- L986 `TestPollResults.test_run_result_polling_voids_primary_and_scores_double_once`: season fixture only; 4 → 4 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L1012 `TestPollResults.test_run_result_polling_uses_schedule_void_when_live_feed_preview`: season fixture only; 4 → 4 assertions. Preserves original subject; assertions unchanged and non-vacuous.

## tests/test_scoring_lock.py

- L64 `test_check_results_skips_when_peer_scored_mid_flight`: season fixture only; 3 → 3 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L136 `test_check_results_fails_closed_when_pick_vanishes`: season fixture only; 3 → 3 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L160 `test_check_results_adopts_fresh_metadata_before_scoring`: season fixture only; 3 → 3 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L187 `test_reconcile_results_saves_under_scoring_lock`: season fixture only; 2 → 2 assertions. Preserves original subject; assertions unchanged and non-vacuous.

## tests/test_streak_applied_dates.py

- L15 `test_the_same_date_is_applied_once`: season fixture only; 3 → 3 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L22 `test_a_repeated_saver_miss_does_not_reset_the_streak`: season fixture only; 3 → 3 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L29 `test_different_dates_each_apply`: season fixture only; 2 → 2 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L42 `test_other_writers_of_the_streak_keep_the_marker`: season fixture only; 2 → 2 assertions. Preserves original subject; assertions unchanged and non-vacuous.
- L50 `test_a_tagged_streak_file_without_a_marker_still_works`: season fixture only; 2 → 2 assertions. Original covered tagless carry-over and absence of a marker together. Now explicitly tagged: preserves the missing-marker subject, intentionally drops obsolete tagless carry-over; new test_row3_tagless_three_starts_at_one_instead_of_carrying_over covers its replacement.
- L56 `test_the_marker_list_is_bounded`: date loop now uses valid ISO dates; 1 → 1 assertions. Original generated 2027-04-31. New timedelta dates are valid, still 35 updates against a 30-marker cap; identical assertion remains meaningful.

## Incident register line-by-line check

The only twelve changed lines are save_streak fixture calls at current lines 264, 286, 304, 322, 338, 386, 398, 410, 464, 476, 489, 517. Each adds exactly season=2026, matching DATE and all L01/L02/L04 contest clocks. Removing these keywords makes the entire module AST identical to base. All 46 assert statements, the dedicated incident exception classes, strict oracle, bad/required tuples, HTTP call assertions, BaseException death point, and E77/L03 portions are unchanged. L01 still routes real feeds through the scorer; L02 still replays twice; L04 still dies between writes and exercises a real restart. The file passes on both commits with each version's own fixtures. Frozen scripts/audit/incident_register is unchanged.
