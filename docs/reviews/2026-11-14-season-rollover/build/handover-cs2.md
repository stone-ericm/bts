# CS2 handover

Implementation commit: `9d4353a330552a881dcc022ccec423a8e8e72f7c` — `fix(picks): scope model streak state to its contest season`.
Branch: `sb-cs2`. Base: `d8eb50d0` (the kickoff commit). No push, network, box, or contest-account operation was performed. `HANDOVER-CS2.md` and `.codex-review/cs2/` are intentionally uncommitted.

The model streak is now scoped to the contest date's calendar year. Only a JSON object whose season has `type(value) is int` and equals that year matches. Reads do not migrate files. Dated scoring and successful season replay stamp the year; dateless operations retain legacy behavior and preserve an existing tag verbatim. The live contest observation and the saver flag authority are unchanged.

## Contract evidence

Locations are for the implementation commit above. All test names in this table are in `tests/test_streak_season_cs2.py` unless another file is named. Parameterized tests cover multiple cases under the same test function.

| Row | Implementation / unchanged boundary | Tests |
| --- | --- | --- |
| 1 | `src/bts/picks.py:516`, `src/bts/picks.py:528`, `src/bts/picks.py:540`, `src/bts/picks.py:555`. Keyword-only season; strict integer match; fresh defaults; default None follows the original parsing/error paths. | `tests/test_streak_season_cs2.py:34` (`test_row1_readers_require_an_exact_integer_season_without_writing`)<br>`tests/test_streak_season_cs2.py:47` (`test_row1_dateless_corruption_behavior_is_unchanged`)<br>`tests/test_streak_season_cs2.py:56` (`test_row1_nonobject_or_unparseable_file_cannot_match_season`)<br>`tests/test_streak_season_cs2.py:63` (`test_row1_missing_file_and_missing_keys_keep_defaults`) |
| 2 | `src/bts/picks.py:568`. Seasonal saves preserve only matching saver/markers; explicit saver wins. Dateless saves retain tags including null, bool, string and float values. Legacy explicit-saver overwrite tolerance is retained. | `tests/test_streak_season_cs2.py:72` (`test_row2_save_resets_only_preserved_state_on_mismatch`)<br>`tests/test_streak_season_cs2.py:88` (`test_row2_dateless_save_preserves_tag_verbatim`) |
| 3 | `src/bts/picks.py:628`; dated scoring callers: `src/bts/cli.py:2437`, `src/bts/scheduler.py:2563,2609`. Strict ISO date validation precedes all state I/O; its year is passed to all three reads and every save. | `tests/test_streak_season_cs2.py:101` (`test_row3_legacy_rollover_then_same_date_retry`)<br>`tests/test_streak_season_cs2.py:112` (`test_row3_matching_season_retains_saver_forgiveness`)<br>`tests/test_streak_season_cs2.py:119` (`test_row3_tagless_three_starts_at_one_instead_of_carrying_over`)<br>`tests/test_streak_season_cs2.py:126` (`test_row3_invalid_date_fails_before_io`) |
| 4 | `src/bts/picks.py:628` and `src/bts/picks.py:568`. No date means season=None, no reset and no once-per-date guard. Existing season tag survives. | `tests/test_streak_season_cs2.py:135` (`test_row4_dateless_update_preserves_tag_and_state`)<br>`tests/test_streak_season_cs2.py:145` (`test_row4_tagless_dateless_update_keeps_legacy_state`) |
| 5 | `src/bts/picks.py:1368` through the save at line 1372 stamps replay_day.year. `src/bts/picks.py:685`; its line 707 marker read deliberately has no season. A None replay still saves nothing. | `tests/test_streak_season_cs2.py:161` (`test_row5_reconcile_stamps_replayed_season_and_discards_old_markers`)<br>`tests/test_streak_season_cs2.py:169` (`test_row5_tagless_same_day_marker_still_blocks_incomplete_replay`) |
| 6 | `src/bts/cli.py:2438`; `src/bts/scheduler.py:2561,2605,2610`. Each display read passes the scored date year, including void and peer-scored paths. | `tests/test_streak_season_cs2.py:179` (`test_row6_cli_void_display_reads_scored_season`)<br>`tests/test_streak_season_cs2.py:191` (`test_row6_scheduler_display_reads_scored_season`) |
| 7 | `src/bts/cli.py:2539,2543,2546`. The local now_et() converts receipt.clock() to ET; receipt.clock is the same callable passed to reconcile_results, whose own now_et() performs the same conversion. No second wall-clock source was introduced. | `tests/test_streak_season_cs2.py:227` (`test_row7_reconcile_summary_uses_replay_et_clock`) |
| 8 | `src/bts/scheduler.py:1899`. Shadow model streak comes from the shadow date year. | `tests/test_streak_season_cs2.py:244` (`test_row8_shadow_selection_uses_own_date_season`) |
| 9 | `src/bts/strategy.py:573`. Saver fallback uses select_pick's date year; explicit saver remains authoritative. | `tests/test_streak_season_cs2.py:261` (`test_row9_strategy_saver_fallback_uses_selection_date`) |
| 10 | `src/bts/contest_state.py:308`; year supplied at `src/bts/contest_state.py:345,347`. Readable mismatches give (0,True,None). Non-object/bad-JSON/non-UTF-8/unreadable model data still follows CS1 error handling before the season comparison. Contest observation precedence and saver_state reads are untouched. | `tests/test_streak_season_cs2.py:276` (`test_row10_model_diagnostics_reset_without_changing_contest_or_saver_authority`) |
| 11 | `src/bts/picks.py:516,528,540,555,628,1372` and `src/bts/contest_state.py:308,347`. These read/write paths implement this without migration or inference from updated/applied dates/pick files. | `tests/test_streak_season_cs2.py:296` (`test_row11_tagless_reads_do_not_migrate_then_first_dated_write_stamps`) |
| 12 | `src/bts/web.py:114,1089`, `src/bts/health/streak_validation.py:27`, `src/bts/state/export.py:21,60`, `src/bts/state/verify.py:65`, `src/bts/state/regenerate.py:387,395`, `src/bts/data/backup.py:317,327`. No edits to these direct readers/writers. Their file formats remain compatible; relevant existing tests pass in the streak-focused run. | `tests/test_streak_season_cs2.py:307` (`test_row12_tagged_file_still_validates_and_exports`) |

The row 10 regression runs with and without a contest observation; it checks stale/tagless/current tags plus bool/string/float impostors. It asserts the contest streak remains 7 when present and that the uninitialized saver flag still makes the decision saver unavailable. The existing CS1 defensive/corrupt-precedence tests passed unchanged. Row 12 checks both no validation alert and exported streak/saver values from a file tagged 2027.

Season-specific scorer loaders regard a non-object/unparseable file as unable to match a season; the dateless loaders retain their original corrupt-file behavior. This does not weaken the separate contest model-diagnostics corruption path.

## Existing tests changed and why

No existing assertion was deleted or relaxed. The one former tagless carry-over test is renamed and seeded with an explicit season so it continues testing the absence of an applied-date marker. A separate new regression asserts the intentional tagless 3 -> 1 reset. All other fixture edits preserve the tests' original scoring, saver, cutoff, concurrency, or observation expectations.


### `tests/test_check_results_wait.py`

Tag the pre-existing streak as 2026 so stale-score guards, waits, shadow handling and deadlines keep exercising their in-season scenario.

- `TestShadowReconcileOnEveryExit.test_production_pending_still_reconciles_shadow` — `tests/test_check_results_wait.py:127`
- `TestShadowReconcileOnEveryExit.test_production_resolver_error_still_reconciles_shadow` — `tests/test_check_results_wait.py:148`
- `TestStaleScoringGuard.test_old_unresolved_scoreable_production_never_scores` — `tests/test_check_results_wait.py:174`
- `TestStaleScoringGuard.test_allow_stale_scoring_override_scores` — `tests/test_check_results_wait.py:199`
- `TestStaleScoringGuard.test_yesterday_scores_without_override` — `tests/test_check_results_wait.py:217`
- `TestWaitDeadline.test_wait_retries_until_production_resolves` — `tests/test_check_results_wait.py:237`
- `TestWaitDeadline.test_deadline_is_a_hard_bound` — `tests/test_check_results_wait.py:256`
- `TestWaitDeadline.test_default_is_single_attempt_no_sleep` — `tests/test_check_results_wait.py:277`
- `TestWaitDeadline.test_wait_already_past_deadline_single_attempt` — `tests/test_check_results_wait.py:296`
- `TestWaitDeadline.test_wait_settles_immediately_when_already_resolved` — `tests/test_check_results_wait.py:332`
- `TestWaitDeadline.test_vanished_pick_file_stays_pending` — `tests/test_check_results_wait.py:364`

### `tests/test_cli_integration.py`

Tag the model/scoring fixture as 2026 so CLI action, scoring, void, shadow and scoreability checks retain their intended in-season preconditions.

- `TestBtsRun.test_run_uses_fresh_contest_streak_for_live_action` — `tests/test_cli_integration.py:187`
- `TestBtsPreview.test_preview_uses_fresh_contest_streak_for_projected_pick` — `tests/test_cli_integration.py:228`
- `TestBtsCheckResults.test_check_results_hit_updates_streak` — `tests/test_cli_integration.py:316`
- `TestBtsCheckResults.test_check_results_miss_resets_streak` — `tests/test_cli_integration.py:336`
- `TestBtsCheckResults.test_check_results_none_warns_scratched` — `tests/test_cli_integration.py:356`
- `TestBtsCheckResults.test_check_results_voids_primary_and_scores_double_once` — `tests/test_cli_integration.py:379`
- `TestBtsCheckResults.test_check_results_voids_double_and_scores_primary_once` — `tests/test_cli_integration.py:418`
- `TestBtsCheckResults.test_check_results_all_void_keeps_streak` — `tests/test_cli_integration.py:457`
- `TestBtsCheckResults.test_check_results_skips_already_resolved` — `tests/test_cli_integration.py:493`
- `TestBtsCheckResults.test_check_results_resolves_shadow_when_production_already_resolved` — `tests/test_cli_integration.py:516`
- `TestBtsCheckResults.test_check_results_writes_shadow_status_artifact` — `tests/test_cli_integration.py:556`
- `TestBtsCheckResults.test_check_results_shadow_status_failure_does_not_crash` — `tests/test_cli_integration.py:591`
- `TestBtsCheckResults.test_check_results_skips_already_resolved_shadow` — `tests/test_cli_integration.py:624`
- `TestBtsCheckResults.test_check_results_resolves_shadow_with_unresolved_production` — `tests/test_cli_integration.py:654`
- `TestBtsCheckResults.test_check_results_shadow_double_down_is_dd_aware` — `tests/test_cli_integration.py:687`
- `TestBtsCheckResults.test_check_results_skips_unscoreable_skip_record` — `tests/test_cli_integration.py:769`
- `TestBtsCheckResults.test_check_results_scores_scoreable_decision` — `tests/test_cli_integration.py:783`
- `TestBtsCheckResults.test_check_results_missing_decision_falls_back_to_delivered` — `tests/test_cli_integration.py:794`
- `TestBtsCheckResults.test_check_results_missing_decision_undelivered_not_scored` — `tests/test_cli_integration.py:803`

### `tests/test_contest_state.py`

Tag the model diagnostics fixture as 2026 so contest precedence/freshness/no-inflation and model-fallback assertions keep seeing the stated model streak.

- `test_model_state_used_when_no_contest_state` — `tests/test_contest_state.py:16`
- `test_required_contest_state_missing_raises` — `tests/test_contest_state.py:34`
- `test_fresh_contest_state_drives_live_decision` — `tests/test_contest_state.py:51`
- `test_lagged_contest_uses_contest_value_keeps_doubles` — `tests/test_contest_state.py:79`
- `test_lagged_contest_model_does_not_inflate` — `tests/test_contest_state.py:97`
- `test_missing_source_date_is_stale_but_keeps_doubles` — `tests/test_contest_state.py:195`
- `test_model_never_inflates_decision_streak` — `tests/test_contest_state.py:208`
- `test_two_pick_gap_is_stale` — `tests/test_contest_state.py:228`
- `test_unconfirmed_local_miss_marks_stale` — `tests/test_contest_state.py:241`
- `test_unexpired_override_wins_over_auto_through_decision` — `tests/test_contest_state.py:254`
- `test_saver_conservative_when_reached_zone_but_unconfirmed` — `tests/test_contest_state.py:275`

### `tests/test_incident_register_2026.py`

Tag the model fixture as 2026 so the L01 grading, L02 replay and L04 crash/retry regressions remain in-season cases. No frozen incident-register script was edited.

- `test_l01_check_results_single_pick` — `tests/test_incident_register_2026.py:259`
- `test_l01_check_results_hit_plus_pass` — `tests/test_incident_register_2026.py:281`
- `test_l01_check_results_pass_plus_pass_preserves` — `tests/test_incident_register_2026.py:298`
- `test_l01_pass_at_saver_streak_keeps_the_saver` — `tests/test_incident_register_2026.py:314`
- `test_l01_absent_player_stays_pending` — `tests/test_incident_register_2026.py:329`
- `test_l02_same_day_hit_survives_two_late_reconciles` — `tests/test_incident_register_2026.py:382`
- `test_l02_same_day_saver_consumption_survives_two_late_reconciles` — `tests/test_incident_register_2026.py:393`
- `test_l02_unplayed_undelivered_preview_is_excluded` — `tests/test_incident_register_2026.py:404`
- `test_l04_hit_applied_once_across_a_crash_and_restart` — `tests/test_incident_register_2026.py:460`
- `test_l04_saver_miss_applied_once_across_a_crash_and_restart` — `tests/test_incident_register_2026.py:471`
- `test_l04_polling_death_then_the_cron_scorer_applies_once` — `tests/test_incident_register_2026.py:483`
- `test_l04_control_without_the_fault_a_rerun_is_a_no_op` — `tests/test_incident_register_2026.py:513`

### `tests/test_pickpath_review_pp1.py`

Tag the starting streak as 2026 so the same-day crash/reconcile/restart test still begins at 1 and ends at 2.

- `test_crash_reconcile_restart_keeps_applied_hit` — `tests/test_pickpath_review_pp1.py:23`

### `tests/test_pickpath_review_pp2.py`

Tag the starting streak as 2026 so all eight hit-plus-void permutations still check 1 -> 2 through CLI/polling.

- `test_delivered_hit_plus_void_double_adds_exactly_one` — `tests/test_pickpath_review_pp2.py:239`

### `tests/test_picks.py`

Tag only reconcile fixtures as 2026, matching their pick dates; legacy dateless update/saver tests are untouched.

- `TestReconcileResults.test_no_corrections_when_results_match` — `tests/test_picks.py:665`
- `TestReconcileResults.test_corrects_hit_to_miss` — `tests/test_picks.py:688`
- `TestReconcileResults.test_preview_pick_for_tomorrow_does_not_break_streak_walk` — `tests/test_picks.py:715`
- `TestReconcileResults.test_partial_void_hit_counts_as_single_streak_day` — `tests/test_picks.py:771`
- `TestReconcileResults.test_shadow_pick_files_are_ignored_in_streak_walk` — `tests/test_picks.py:803`

### `tests/test_reconcile_cutoff.py`

The shared _plant_hit helper tags the seed with int(day[:4]), including its moving real-clock case; cutoff assertions stay unchanged.

- `_plant_hit` — `tests/test_reconcile_cutoff.py:24`

Tests using that shared fixture: `test_post_cutoff_rescoring_does_not_flip_a_settled_hit`, `test_pre_cutoff_correction_is_still_applied`, `test_cutoff_is_0800_eastern_the_next_day`, `test_today_is_the_eastern_calendar_day`, `test_feed_answer_arriving_after_the_cutoff_is_discarded`, `test_write_landing_after_the_cutoff_is_refused`, `test_late_feed_answer_is_discarded_even_if_the_clock_steps_back`, `test_replay_boundary_is_read_under_the_lock`.

### `tests/test_reconcile_receipt.py`

The shared plant helper tags the seed as 2026, matching receipt/replay dates; receipt content/behavior assertions stay unchanged.

- `plant` — `tests/test_reconcile_receipt.py:36`

Tests using that shared fixture: `test_a_pre_cutoff_correction_is_observed_and_applied`, `test_an_unchanged_result_is_coverage_even_with_no_corrections`, `test_dates_outside_the_window_or_without_a_graded_pick_are_skipped_with_a_reason`, `test_a_pending_slot_is_not_coverage`, `test_a_pending_first_slot_leaves_the_double_down_not_attempted`, `test_a_schedule_void_slot_records_its_basis_without_a_feed`, `test_a_status_fetch_failure_is_recorded`, `test_a_feed_failure_is_recorded_and_still_raises`, `test_a_feed_answer_after_the_cutoff_is_a_late_observation`, `test_a_write_landing_after_the_cutoff_is_refused_and_recorded`, `test_the_receipt_changes_no_request_or_result`, `test_a_receipt_method_failure_never_changes_the_run`, `test_a_response_after_the_cutoff_is_never_coverage_even_if_the_clock_steps_back`, `test_d1_a_lost_feed_observation_is_never_coverage`, `test_d2_a_late_response_is_not_erased_by_a_later_step_back`, `test_d2_a_late_selected_feed_then_an_on_time_fallback_is_not_coverage`, `test_d3_the_basis_comes_from_the_consumed_payloads`, `test_d3_a_feed_naming_another_game_is_not_the_selected_game`, `test_d4_a_failed_save_is_never_recorded_as_applied`, `test_d4_the_written_selection_is_recorded_separately`, `test_d5_receipt_processing_cannot_move_the_run_past_its_cutoff`, `test_d1_a_day_level_hook_failure_uncovers_its_slots`, `test_d2_an_out_of_order_clock_before_the_cutoff_is_still_not_coverage`, `test_r3_d3_a_feed_without_its_own_game_id_is_unqualified`, `test_r3_d3_a_suspended_void_must_be_the_selected_game`, `test_r3_d3_a_name_only_batter_match_is_unqualified`, `test_r3_d1_a_lost_earlier_leg_observation_uncovers_the_later_leg`, `test_r3_d6_a_write_hook_failure_is_unavailable_evidence_on_its_own_date`, `test_r4_d3_id_presence_cannot_qualify_another_players_grade`, `test_r4_d3_id_presence_cannot_qualify_a_name_contaminated_suspended_grade`.

### `tests/test_scheduler.py`

Tag both polling fixtures as 2026 so partial-void scoring still checks 3 -> 4.

- `TestPollResults.test_run_result_polling_voids_primary_and_scores_double_once` — `tests/test_scheduler.py:986`
- `TestPollResults.test_run_result_polling_uses_schedule_void_when_live_feed_preview` — `tests/test_scheduler.py:1012`

### `tests/test_scoring_lock.py`

Tag direct model fixtures as 2026 so peer-scoring, vanished-file, metadata and locked-reconcile checks use their own contest season.

- `test_check_results_skips_when_peer_scored_mid_flight` — `tests/test_scoring_lock.py:64`
- `test_check_results_fails_closed_when_pick_vanishes` — `tests/test_scoring_lock.py:136`
- `test_check_results_adopts_fresh_metadata_before_scoring` — `tests/test_scoring_lock.py:160`
- `test_reconcile_results_saves_under_scoring_lock` — `tests/test_scoring_lock.py:187`

### `tests/test_streak_applied_dates.py`

Tag in-season dated seeds as 2027. The dateless control and the later dateless marker-preserving save remain dateless. Rename the old tagless carry-over test to test_a_tagged_streak_file_without_a_marker_still_works, preserving its +1 assertion. Use date + timedelta for retention dates, replacing the formerly generated invalid April 31 without weakening the list-length assertion.

- `test_the_same_date_is_applied_once` — `tests/test_streak_applied_dates.py:15`
- `test_a_repeated_saver_miss_does_not_reset_the_streak` — `tests/test_streak_applied_dates.py:22`
- `test_different_dates_each_apply` — `tests/test_streak_applied_dates.py:29`
- `test_other_writers_of_the_streak_keep_the_marker` — `tests/test_streak_applied_dates.py:42`
- `test_a_tagged_streak_file_without_a_marker_still_works` — `tests/test_streak_applied_dates.py:50`
- `test_the_marker_list_is_bounded` — `tests/test_streak_applied_dates.py:56`

## Calls and year propagation

The inventory below includes the six public functions (three loaders, save, update, reconcile), plus replay and the private model-diagnostics reader. It groups identical call policies by file and lists every direct Python call line in tracked source, tests and archived probes. Test assertions invoking loaders without a season intentionally inspect the bytes/state produced or pin the legacy API. Archived probes are historical artifacts, not production callers, and were not migrated. Dynamic test-loader loops are covered by row 1's parameterized tests.

Production policy:

- update_streak's reads and all three save branches receive the year derived from its validated date, or None for legacy dateless calls.
- save_streak passes its own season to load_applied_dates. Its dateless path preserves any existing season tag.
- _replay_season_streak's applied-date lookup stays dateless because the date identifies its own year, including tagless same-day crash markers.
- reconcile_results calls replay with replay_day.year and saves with that same year, only on a non-None replay.
- CLI scoring and both scheduler scoring branches pass date to update_streak; it derives the year. CLI/scheduler readbacks and shadow reads pass the scored/selected date year.
- Both CLI reconcile summary reads use now_et().year from receipt.clock, also used by replay.
- select_pick passes its date year to the saver fallback only when no explicit saver was provided.
- load_decision_streak_state passes its already computed ET now_year to _load_model_diagnostics.
- web.load_streak is a separate dashboard reader, with its one dashboard fallback call intentionally unchanged (row 12).

| File | Function called | Call lines | Season / clock policy |
| --- | --- | --- | --- |
| `docs/reviews/2026-11-14-gemini-known-answer/g2-probe_streak_reconcile.py` | `load_applied_dates` | 101, 162 | `season=None: legacy/default API` |
| `docs/reviews/2026-11-14-gemini-known-answer/g2-probe_streak_reconcile.py` | `load_saver_available` | 132, 138, 145 | `season=None: legacy/default API` |
| `docs/reviews/2026-11-14-gemini-known-answer/g2-probe_streak_reconcile.py` | `load_streak` | 100, 109, 198, 228 | `season=None: legacy/default API` |
| `docs/reviews/2026-11-14-gemini-known-answer/g2-probe_streak_reconcile.py` | `reconcile_results` | 185 | `clock=lambda: datetime(2027, 5, 2, 23, 30, tzinfo=ET); replay year read under lock` |
| `docs/reviews/2026-11-14-gemini-known-answer/g2-probe_streak_reconcile.py` | `reconcile_results` | 196 | `clock=lambda: datetime(2027, 5, 4, 2, 0, tzinfo=ET); replay year read under lock` |
| `docs/reviews/2026-11-14-gemini-known-answer/g2-probe_streak_reconcile.py` | `reconcile_results` | 99, 137, 225 | `clock=lambda: now_reconcile; replay year read under lock` |
| `docs/reviews/2026-11-14-gemini-known-answer/g2-probe_streak_reconcile.py` | `save_streak` | 84, 125, 158, 180, 209 | `season=None: legacy/default API` |
| `docs/reviews/2026-11-14-gemini-known-answer/g2-probe_streak_reconcile.py` | `update_streak` | 92, 181 | `date='2027-05-02'; year derived internally when dated` |
| `docs/reviews/2026-11-14-gemini-known-answer/g2-probe_streak_reconcile.py` | `update_streak` | 131 | `date='2027-05-11'; year derived internally when dated` |
| `docs/reviews/2026-11-14-gemini-known-answer/g2-probe_streak_reconcile.py` | `update_streak` | 161 | `date=d; year derived internally when dated` |
| `docs/reviews/2026-11-14-season-rollover/build/cs1-review/carbon-r1/probe_cs1_review.py` | `load_applied_dates` | 365 | `season=None: legacy/default API` |
| `docs/reviews/2026-11-14-season-rollover/build/cs1-review/carbon-r1/probe_cs1_review.py` | `load_saver_available` | 364 | `season=None: legacy/default API` |
| `docs/reviews/2026-11-14-season-rollover/build/cs1-review/carbon-r1/probe_cs1_review.py` | `load_streak` | 363, 449 | `season=None: legacy/default API` |
| `docs/reviews/2026-11-14-season-rollover/build/cs1-review/codex-confirm/probe.py` | `_load_model_diagnostics` | 271, 316 | `now_year=not supplied: archived historical-snapshot API` |
| `docs/reviews/2026-11-14-season-rollover/build/cs1-review/codex-r1/probe.py` | `load_saver_available` | 270 | `season=None: legacy/default API` |
| `docs/reviews/2026-11-14-season-rollover/build/cs1-review/codex-r1/probe.py` | `load_streak` | 269 | `season=None: legacy/default API` |
| `docs/reviews/2026-11-14-season-rollover/build/cs1-review/codex-r2/probe_r2.py` | `load_streak` | 138 | `season=None: legacy/default API` |
| `docs/reviews/2026-11-14-season-rollover/probes-audit/run_probes.py` | `load_saver_available` | 582 | `season=None: legacy/default API` |
| `docs/reviews/2026-11-14-season-rollover/probes-audit/run_probes.py` | `load_streak` | 582 | `season=None: legacy/default API` |
| `docs/reviews/2026-11-14-season-rollover/probes-audit/run_probes.py` | `reconcile_results` | 580 | `clock=lambda: datetime(2027, 3, 4, 2, 0, tzinfo=ET); replay year read under lock` |
| `docs/reviews/2026-11-14-season-rollover/probes-r2/carbon_probe_r2_state_saver.py` | `save_streak` | 105 | `season=None: legacy/default API` |
| `docs/reviews/2026-11-14-season-rollover/probes-r2/carbon_probe_r2_state_saver.py` | `update_streak` | 107 | `date='2027-03-24'; year derived internally when dated` |
| `docs/reviews/2026-11-14-season-rollover/probes-r2/carbon_probe_s1_s2.py` | `reconcile_results` | 233 | `clock=lambda: datetime(2027, 3, 25, 6, 0, tzinfo=timezone.utc); replay year read under lock` |
| `docs/reviews/2026-11-14-season-rollover/probes-r2/carbon_probe_s1_s2.py` | `update_streak` | 192, 208 | `date='2027-03-24'; year derived internally when dated` |
| `docs/reviews/2026-11-14-season-rollover/probes-r2/codex_probe_r2_state.py` | `update_streak` | 117 | `date=day; year derived internally when dated` |
| `docs/reviews/2026-11-14-season-rollover/probes-r2/codex_probe_scheduler.py` | `load_streak` | 150 | `season=None: legacy/default API` |
| `docs/reviews/2026-11-14-season-rollover/probes-r2/codex_probe_state.py` | `load_streak` | 80, 81 | `season=None: legacy/default API` |
| `docs/reviews/2026-11-14-season-rollover/probes-r2/codex_probe_state.py` | `reconcile_results` | 79 | `clock=lambda: NOW; replay year read under lock` |
| `docs/reviews/2026-11-14-season-rollover/probes-r2/codex_probe_state.py` | `update_streak` | 73 | `date=DATE; year derived internally when dated` |
| `docs/reviews/2026-11-14-season-rollover/probes/codex_r2_notes.py` | `_replay_season_streak` | 33 | `season=2027` |
| `docs/reviews/2026-11-14-season-rollover/probes/codex_r2_notes.py` | `load_applied_dates` | 31, 32 | `season=None: legacy/default API` |
| `docs/reviews/2026-11-14-season-rollover/probes/codex_r2_notes.py` | `update_streak` | 32 | `date='2027-03-24'; year derived internally when dated` |
| `docs/reviews/2026-11-14-season-rollover/probes/gemini_r2_season_start_notes.py` | `load_streak` | 152, 163 | `season=None: legacy/default API` |
| `docs/reviews/2026-11-14-season-rollover/probes/gemini_r2_season_start_notes.py` | `reconcile_results` | 151, 162 | `clock=lambda: datetime(2027, 3, 25, 2, 0, tzinfo=et); replay year read under lock` |
| `docs/reviews/2026-11-14-season-rollover/probes/gemini_r2_season_start_notes.py` | `update_streak` | 148 | `date='2027-03-24'; year derived internally when dated` |
| `src/bts/cli.py` | `load_streak` | 2438 | `season=date_type.fromisoformat(date).year` |
| `src/bts/cli.py` | `load_streak` | 2543, 2546 | `season=now_et().year` |
| `src/bts/cli.py` | `reconcile_results` | 2529 | `clock=receipt.clock; replay year read under lock` |
| `src/bts/cli.py` | `update_streak` | 2437 | `date=date; year derived internally when dated` |
| `src/bts/contest_state.py` | `_load_model_diagnostics` | 347 | `now_year=now_year` |
| `src/bts/picks.py` | `_replay_season_streak` | 1369 | `season=replay_day.year` |
| `src/bts/picks.py` | `load_applied_dates` | 707 | `season=None: legacy/default API` |
| `src/bts/picks.py` | `load_applied_dates` | 593, 655 | `season=season` |
| `src/bts/picks.py` | `load_saver_available` | 654 | `season=season` |
| `src/bts/picks.py` | `load_streak` | 653 | `season=season` |
| `src/bts/picks.py` | `save_streak` | 1372 | `season=replay_day.year` |
| `src/bts/picks.py` | `save_streak` | 660, 665, 668 | `season=season` |
| `src/bts/scheduler.py` | `load_streak` | 1899, 2561, 2605, 2610 | `season=date_type.fromisoformat(date).year` |
| `src/bts/scheduler.py` | `update_streak` | 2563, 2609 | `date=date; year derived internally when dated` |
| `src/bts/strategy.py` | `load_saver_available` | 573 | `season=date_type.fromisoformat(date).year` |
| `src/bts/web.py` | `load_streak` | 1089 | `none: separate dashboard fallback, explicitly unchanged by row 12` |
| `tests/test_check_results_wait.py` | `load_streak` | 142, 167, 193, 213, 231, 251, 271, 292, 311, 350, 389 | `season=None: legacy/default API` |
| `tests/test_check_results_wait.py` | `save_streak` | 133, 154, 183, 205, 223, 243, 264, 283, 302, 342, 373 | `season=2026` |
| `tests/test_cli_integration.py` | `load_streak` | 375, 413, 452, 490, 512, 552, 778, 809 | `season=None: legacy/default API` |
| `tests/test_cli_integration.py` | `save_streak` | 200, 241, 321, 341, 362, 392, 431, 470, 500, 524, 564, 598, 630, 662, 695, 772, 786, 797, 807 | `season=2026` |
| `tests/test_contest_state_defensive_reads.py` | `load_applied_dates` | 229 | `season=None: legacy/default API` |
| `tests/test_contest_state_defensive_reads.py` | `load_saver_available` | 229 | `season=None: legacy/default API` |
| `tests/test_contest_state_defensive_reads.py` | `load_streak` | 229 | `season=None: legacy/default API` |
| `tests/test_decision_saver_fallback.py` | `save_streak` | 17, 25, 36 | `season=None: legacy/default API` |
| `tests/test_incident_register_2026.py` | `load_saver_available` | 239, 377 | `season=None: legacy/default API` |
| `tests/test_incident_register_2026.py` | `load_streak` | 239, 377 | `season=None: legacy/default API` |
| `tests/test_incident_register_2026.py` | `reconcile_results` | 373 | `clock=lambda: now; replay year read under lock` |
| `tests/test_incident_register_2026.py` | `save_streak` | 264, 286, 302, 318, 333, 386, 398, 410, 464, 476, 489, 517 | `season=2026` |
| `tests/test_pickpath_review_pp1.py` | `load_applied_dates` | 40 | `season=None: legacy/default API` |
| `tests/test_pickpath_review_pp1.py` | `load_streak` | 39, 44, 47 | `season=None: legacy/default API` |
| `tests/test_pickpath_review_pp1.py` | `reconcile_results` | 43 | `clock=lambda: now; replay year read under lock` |
| `tests/test_pickpath_review_pp1.py` | `save_streak` | 28 | `season=2026` |
| `tests/test_pickpath_review_pp2.py` | `load_streak` | 290, 292 | `season=None: legacy/default API` |
| `tests/test_pickpath_review_pp2.py` | `reconcile_results` | 291 | `clock=lambda: H.AFTER_NORMAL_GAMES; replay year read under lock` |
| `tests/test_pickpath_review_pp2.py` | `save_streak` | 245 | `season=2026` |
| `tests/test_picks.py` | `load_saver_available` | 465, 479, 485, 488 | `season=None: legacy/default API` |
| `tests/test_picks.py` | `load_streak` | 427, 431, 437, 443, 713, 766, 768, 801, 833, 834 | `season=None: legacy/default API` |
| `tests/test_picks.py` | `reconcile_results` | 685, 709, 798 | `clock=_cron_clock(yesterday); replay year read under lock` |
| `tests/test_picks.py` | `reconcile_results` | 762, 831 | `clock=_cron_clock(yesterday_iso); replay year read under lock` |
| `tests/test_picks.py` | `save_streak` | 681, 705, 758, 792, 827 | `season=2026` |
| `tests/test_picks.py` | `save_streak` | 430, 434, 440, 446, 451, 456, 462, 469, 471, 476, 482 | `season=None: legacy/default API` |
| `tests/test_picks.py` | `update_streak` | 435, 441, 447, 452, 457, 463, 470, 472, 477, 483 | `date=None; year derived internally when dated` |
| `tests/test_reconcile_cutoff.py` | `load_streak` | 185 | `season=None: legacy/default API` |
| `tests/test_reconcile_cutoff.py` | `reconcile_results` | 53 | `clock=default aware ET clock; replay year read under lock` |
| `tests/test_reconcile_cutoff.py` | `reconcile_results` | 190 | `clock=lambda: datetime(2026, 8, 21, 2, 0); replay year read under lock` |
| `tests/test_reconcile_cutoff.py` | `save_streak` | 32 | `season=int(day[:4])` |
| `tests/test_reconcile_receipt.py` | `reconcile_results` | 191, 414 | `clock=SimClock(AT_0200); replay year read under lock` |
| `tests/test_reconcile_receipt.py` | `reconcile_results` | 91 | `clock=clock; replay year read under lock` |
| `tests/test_reconcile_receipt.py` | `save_streak` | 44 | `season=2026` |
| `tests/test_scheduler.py` | `load_streak` | 1006, 1033 | `season=None: legacy/default API` |
| `tests/test_scheduler.py` | `save_streak` | 993, 1019 | `season=2026` |
| `tests/test_scheduler_decision_record_integration.py` | `load_streak` | 258 | `season=None: legacy/default API` |
| `tests/test_scheduler_decision_record_integration.py` | `save_streak` | 246 | `season=None: legacy/default API` |
| `tests/test_scoring_lock.py` | `reconcile_results` | 224 | `clock=lambda: datetime(2026, 8, 21, 2, 0, tzinfo=ZoneInfo('America/New_York')); replay year read under lock` |
| `tests/test_streak_applied_dates.py` | `load_applied_dates` | 19, 33, 39, 46, 52, 60 | `season=None: legacy/default API` |
| `tests/test_streak_applied_dates.py` | `load_saver_available` | 26 | `season=None: legacy/default API` |
| `tests/test_streak_applied_dates.py` | `load_streak` | 19, 26 | `season=None: legacy/default API` |
| `tests/test_streak_applied_dates.py` | `save_streak` | 16, 23, 30, 43, 57 | `season=2027` |
| `tests/test_streak_applied_dates.py` | `save_streak` | 37, 45 | `season=None: legacy/default API` |
| `tests/test_streak_applied_dates.py` | `update_streak` | 17, 18, 24, 25, 31, 44, 47, 53 | `date='2027-04-02'; year derived internally when dated` |
| `tests/test_streak_applied_dates.py` | `update_streak` | 32 | `date='2027-04-03'; year derived internally when dated` |
| `tests/test_streak_applied_dates.py` | `update_streak` | 59 | `date=(date(2027, 4, 1) + timedelta(days=day)).isoformat(); year derived internally when dated` |
| `tests/test_streak_applied_dates.py` | `update_streak` | 38, 39 | `date=None; year derived internally when dated` |
| `tests/test_streak_replay.py` | `_replay_season_streak` | 63, 68, 74, 81 | `season=2026` |
| `tests/test_streak_season_cs2.py` | `_replay_season_streak` | 173 | `season=2027` |
| `tests/test_streak_season_cs2.py` | `load_applied_dates` | 41, 59, 68, 301 | `season=2027` |
| `tests/test_streak_season_cs2.py` | `load_applied_dates` | 38, 52 | `season=None: legacy/default API` |
| `tests/test_streak_season_cs2.py` | `load_saver_available` | 40, 58, 67, 300 | `season=2027` |
| `tests/test_streak_season_cs2.py` | `load_saver_available` | 37 | `season=None: legacy/default API` |
| `tests/test_streak_season_cs2.py` | `load_streak` | 40, 58, 67, 299 | `season=2027` |
| `tests/test_streak_season_cs2.py` | `load_streak` | 37 | `season=None: legacy/default API` |
| `tests/test_streak_season_cs2.py` | `reconcile_results` | 174 | `clock=lambda: datetime(2027, 3, 24, 23, tzinfo=ET); replay year read under lock` |
| `tests/test_streak_season_cs2.py` | `reconcile_results` | 164 | `clock=lambda: datetime(2027, 3, 25, 2, tzinfo=ET); replay year read under lock` |
| `tests/test_streak_season_cs2.py` | `save_streak` | 74 | `season=2027` |
| `tests/test_streak_season_cs2.py` | `save_streak` | 81 | `season=2028` |
| `tests/test_streak_season_cs2.py` | `save_streak` | 93 | `season=None: legacy/default API` |
| `tests/test_streak_season_cs2.py` | `update_streak` | 103, 108, 114, 141, 303 | `date='2027-03-24'; year derived internally when dated` |
| `tests/test_streak_season_cs2.py` | `update_streak` | 121 | `date='2027-04-02'; year derived internally when dated` |
| `tests/test_streak_season_cs2.py` | `update_streak` | 137, 147 | `date=None; year derived internally when dated` |
| `tests/test_streak_season_cs2.py` | `update_streak` | 130 | `date=day; year derived internally when dated` |

## Validation and exact commands

Every pytest invocation went through `/Users/eric/bts-runs/yield_run.sh` with a log under this worktree's `.codex-review/cs2/`, and used `env TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider ...` (the full gate puts the no-cache flag at the end). The wrapper paused/resumed for Victoria 3. No pattern kill was used. The initial sandbox launch could not set nice priority, so it was rerun with the required permission; that failed launch did not run pytest.

Discovery before and after:

```sh
rg -l 'streak\.json|load_streak|save_streak|update_streak|load_saver_available|load_applied_dates|reconcile_results' tests
```

The before list is `.codex-review/cs2/mentioned-tests.txt`; the final list is `.codex-review/cs2/after-mentioned-tests.txt`. Every discovered file was run, not merely the kickoff's likely-file list. For the focused runs, the paths from this discovery were supplied to pytest with `-q`.

Full fast gate before and after:

```sh
/Users/eric/bts-runs/yield_run.sh .codex-review/cs2/<before-or-after>-fast.log env TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -m 'not slow' --ignore=tests/simulate --ignore=tests/model --ignore=tests/experiment --ignore=tests/validate -q -p no:cacheprovider
```

Run results:

| Log in `.codex-review/cs2/` | Result / purpose |
| --- | --- |
| `before-mentioned.log` | 845 passed, 8 skipped. Original code and original focused tests. |
| `before-fast.log` | 38 failed, 4,971 passed, 19 skipped, 6 deselected, 1 xfailed, 16 errors (27m24s). All 53 expected environment failures/errors plus the one baseline artifact explained below. |
| `before-boundary-confirm.log` | 1 passed: original tree's boundary check with the newly authored, uncommitted test file temporarily outside tests/. |
| `red.log` | 48 failed, 6 passed. Initial new regression batch; test-seam corrections followed before implementation. |
| `red-corrected.log` | 46 failed, 8 passed. Corrected policy seam and diagnostic attribute against unchanged production code. |
| `red-additional.log` | 52 failed, 8 passed. Added missing/default/non-object season-match cases, still before implementation. |
| `red-final.log` | 62 failed, 8 passed. Added dateless exact-tag preservation, still before implementation. |
| `red-tagless-three.log` | 1 failed: exact tagless 3 -> 1 regression before implementation (old result 4). |
| `red-cli-display-corrected.log` | 2 failed: corrected exact CLI text, run with the original CLI read restored temporarily; stale/tagless output was 13 rather than 0. Season-aware CLI immediately restored afterward. |
| `after-code-mentioned.log` | 39 failed, 877 passed, 8 skipped. 37 existing in-season fixture/date cases plus 2 new CLI assertions with incorrect text. |
| `after-mentioned.log` | 916 passed, 8 skipped after fixture updates and corrected CLI assertion. All 71 new parameterized CS2 cases pass. |
| `after-fast.log` | 37 failed, 5043 passed, 19 skipped, 6 deselected, 1 xfailed, 16 errors in 1486.01s (0:24:46) |

The baseline's extra failure was `tests/scripts/c2_framing/test_screen.py::test_the_boundary_admits_the_real_suite`. I authored the new test file while the baseline gate was still running. That check scans the live tree and rejects uncommitted Python modules, so it reported `tests/test_streak_season_cs2.py: not committed`; the baseline had no production edits. Moving only that file temporarily out of the suite made the isolated original boundary check pass. The baseline count above is reported as actually observed, not relabeled as a clean 53-failure run. The final full gate runs after committing all implementation/test Python files to avoid this artifact.

Final failing-set comparison: exactly matches the 53 reference IDs; no unexpected or missing failures. The reference is `/Users/eric/bts-runs/gate-failed.txt` (53 IDs: 37 failed + 16 errors). Exact comparisons are recorded in `.codex-review/cs2/gate-comparison.txt`. No environment dependencies were installed or bypassed.

`git diff --check` passed. The frozen `scripts/audit/incident_register/` tree has no changes. The only committed production files are the five named by the design. No existing state file or migration was added.

## Known limit and unresolved items

By design, the first dated write after deployment finds a tagless file and starts from season-fresh state. Deployment before the season gives the desired reset. Deployment in mid-season makes readers see model streak 0 until the 02:00 reconcile can replay the season from the pick files and stamp the season. The live decision uses the contest observation when one exists. This was not worked around by inferring a tagless season or writing a migration.

All 12 contract rows are implemented and pinned. No row was rejected or substituted. The only validation caveat before the final result is the disclosed baseline boundary artifact; final gate status is stated above. The CS1 corruption tests remain unchanged. Saver mapping/state, contest files/fetch, alerts, CLI surface and dashboard implementation were not changed.

