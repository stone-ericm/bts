# CS2B handover

Branch: `sb-cs2b`. Base for the WHOLE change set: `d8eb50d0`. Starting point for this attempt: `ee9fb870`. No push, external-service access, box, or contest-account operations were performed; the required test suite uses local loopback socket fixtures. The only new commit is `dd915ac9` (`fix: resolve contest date seasons without breaking legacy reads`), containing exactly:

- `src/bts/picks.py`
- `src/bts/scheduler.py`
- `src/bts/cli.py`
- `src/bts/strategy.py`
- `tests/test_streak_season_cs2.py`

The initial proposed commit containing review artifacts was declined before execution; the index was checked empty, and only the five authorized paths were staged and committed. Nothing under `.codex-review/` is committed. This handover is also uncommitted, as instructed. Inherited commits, unchanged: `3b4196be`, `e6d6f0a9`, `4cf73de7`, `a26958dc`, `ee9fb870`.

## Implementation and proof, L1–L4

The sole new date-to-season function is `src/bts/picks.py:516`, `season_of_date(date)`. Non-strings return None. Strings try `date.fromisoformat` first, then `datetime.strptime(..., "%Y-%m-%d")`, returning the successful calendar year, otherwise None. It catches parser exceptions, does not strip, and adds no formats or regular expression. All 18 mandated interpreter results agree with the table. The existing replay filename regex remains unchanged.

| Rule | Production sites at dd915ac9 | Tests in tests/test_streak_season_cs2.py |
|---|---|---|
| L1 shadow | scheduler.py:1899 | `test_row8_shadow_selection_uses_own_date_season` at line 335: 18 values through the real entry point |
| L1 all-void | scheduler.py:2610 | `test_row6_scheduler_display_reads_scored_season` at line 280, mode `void`, 18 values |
| L1 final peer | scheduler.py:2605 | Same test, mode `final_peer`, 18 values |
| L1 early peer | scheduler.py:2561 | Same test, mode `midgame_peer`, 18 values; checks the actual reply's streak |
| L2 CLI void read | cli.py:2438 | Existing `test_row6_cli_void_display_reads_scored_season` at line 259 retained; new `test_row6_cli_date_matrix_preserves_age_guard` at line 468, 18 values |
| L3 saver fallback | strategy.py:573 | Existing row9 date/season tests retained; `test_row9_saver_read_never_rejects_date` at line 446, 18 values |
| L4 dated write | picks.py:685, before the first state read at line 688 | Helper table at line 32; six parseable strings score once with original markers at line 37; invalid values preserve bytes with all state readers/writer and Path.open forbidden at line 200; existing date-object rejection at line 126 retained |

`_season_of` was deleted from scheduler.py. `_now_et`, the CLI age guard, and strategy's unrelated date restrictions were not changed. `date=None` remains the existing dateless mode: it is NOT an invalid dated write.

All four scheduler paths can reach their season read for every row, including the date object, None, bytes and int. Fixture persistence uses the string form of the lookup key (the production file lookup itself uses an f-string); the entry points receive the original typed value. The scorer fixture has a delivered pick and an ungraded result, and peer fixtures really write the peer result during resolution. No season parser or target entry point is mocked.

The CLI's text option cannot supply typed non-string values. Its real callback was separately invoked with all arguments for those four rows; the base and branch both raise TypeError in decision-path construction before reaching the age guard/season read. For string rows, only padded, compact, and week dates reach the void read; the other eleven strings raise the same ValueError at the unchanged age guard. Strategy reaches its saver read for every row; padded and the three unpadded strings complete selection, compact/week keep their later base ValueErrors, and unknown-season strings/non-strings keep their existing later ValueError/TypeError. Exact messages and outcomes are in the recorded matrix.

## Recorded comparison with the base

Extracted with `git archive d8eb50d0 src | tar -x -C .codex-review/cs2b/base`; no checkout over the tracked tree. The uncommitted `.codex-review/cs2b/date_matrix.py` launches isolated base and branch workers with their respective src on PYTHONPATH and repository test fixture helpers. Its temporary directories are inside `.codex-review/cs2b/`. The clock is pinned to 2027-03-24 23:00 ET, including writer timestamps, fixture decision-receipt timestamps, and selection return timestamps. External prediction, result and posting calls are mocked. Posting mocks return a fixed URI so their object IDs cannot create false output differences.

Executed `.venv/bin/python .codex-review/cs2b/date_matrix.py`, followed by `.venv/bin/python .codex-review/cs2b/summarize_matrix.py`. Both completed successfully against the final source. Raw results are in `matrix-base.json` and `matrix-branch.json` in that directory: every row/path records return or exception type/message, captured output, each observed state read's season/value, the pick file's final result, and streak bytes before/after as text and hex. The summary verifier checks all 126 pairs, exact byte preservation for read paths, identical pre-existing exceptions, no state reads for rejected dated writes, and the six successful seasonal writes.

### Contract inconsistency that cannot be classified as (a) or (b)

For `update_streak(..., date=None)`, the matrix's 2026-tagged fixture returns 14 at both base and branch. The base removes the season tag when saving; rows B and D expressly require the branch to preserve it. This difference predates CS2B and remains required. It is neither (a), since None is not a parseable date, nor (b), since this is a dateless call and must not raise. It is marked `D` below, rather than falsely claiming every difference fits (a)/(b). Removing it would violate rows B/D. No other unclassified difference was found. This qualification also applies to row B's phrase “behaves as at the base”: preserving a tag is the explicit specified exception.

# Recorded date matrix

Base `d8eb50d0` versus the CS2B branch, 18 values × 7 real entry paths.
Raw records include return/exception, output, observed reads, pick result, and exact streak bytes (hex).
All clocks fixed to 2027-03-24 23:00 ET; external services mocked.

`a`: season-scoped read/write; `b`: unknown dated write rejected; `D`: required dateless tag preservation (outside the requested a/b classes).

| Value | Shadow | Void | Final peer | Early peer | CLI | Strategy | Update |
|---|---|---|---|---|---|---|---|
| padded: `"2027-03-24"` | a | a | a | a | a | a | a |
| unpadded-month: `"2027-3-24"` | a | a | a | a | = | a | a |
| unpadded-day: `"2027-03-4"` | a | a | a | a | = | a | a |
| unpadded-both: `"2027-3-4"` | a | a | a | a | = | a | a |
| compact: `"20270324"` | a | a | a | a | a | a | a |
| week: `"2027-W12-3"` | a | a | a | a | a | a | a |
| leading-space: `" 2027-03-24"` | = | = | = | = | = | = | b |
| trailing-space: `"2027-03-24 "` | = | = | = | = | = | = | b |
| time: `"2027-03-24T10:00"` | = | = | = | = | = | = | b |
| slash: `"2027/03/24"` | = | = | = | = | = | = | b |
| dot: `"2027.03.24"` | = | = | = | = | = | = | b |
| impossible: `"2027-02-30"` | = | = | = | = | = | = | b |
| junk: `"junk"` | = | = | = | = | = | = | b |
| empty: `""` | = | = | = | = | = | = | b |
| date-object: `date(2027, 3, 24)` | = | = | = | = | = | = | b |
| none: `None` | = | = | = | = | = | = | D |
| bytes: `b"2027-03-24"` | = | = | = | = | = | = | b |
| integer: `20270324 (int)` | = | = | = | = | = | = | b |

Counts: {'a': 39, '=': 75, 'b': 11, 'D': 1}. `=` means every recorded field is identical between base and branch.

For a: scheduler streak 13 → 0, CLI void display 13 → 0 where its age guard permits entry; strategy saver False → True (later base date exceptions remain identical); scorer 14 → 1, saver False → True, season 2027, original marker retained.
For b: base returns 14 except date/bytes objects raise TypeError; branch raises ValueError before reads and preserves all bytes.
For D: date=None is dateless. Both return 14, but the base removes the tag and the branch preserves season 2026, explicitly required by rows B/D. No row-L implementation can remove this difference while preserving those rows.

## Every differing pair

| Value | Path | Class | Differing recorded fields |
|---|---|---|---|
| padded | shadow | a | outcome, reads |
| padded | void | a | output, reads |
| padded | final_peer | a | output, reads |
| padded | midgame_peer | a | outcome, reads |
| padded | cli | a | outcome, reads |
| padded | strategy | a | outcome, reads |
| padded | update | a | outcome, reads, after, after_hex |
| unpadded-month | shadow | a | outcome, reads |
| unpadded-month | void | a | output, reads |
| unpadded-month | final_peer | a | output, reads |
| unpadded-month | midgame_peer | a | outcome, reads |
| unpadded-month | strategy | a | outcome, reads |
| unpadded-month | update | a | outcome, reads, after, after_hex |
| unpadded-day | shadow | a | outcome, reads |
| unpadded-day | void | a | output, reads |
| unpadded-day | final_peer | a | output, reads |
| unpadded-day | midgame_peer | a | outcome, reads |
| unpadded-day | strategy | a | outcome, reads |
| unpadded-day | update | a | outcome, reads, after, after_hex |
| unpadded-both | shadow | a | outcome, reads |
| unpadded-both | void | a | output, reads |
| unpadded-both | final_peer | a | output, reads |
| unpadded-both | midgame_peer | a | outcome, reads |
| unpadded-both | strategy | a | outcome, reads |
| unpadded-both | update | a | outcome, reads, after, after_hex |
| compact | shadow | a | outcome, reads |
| compact | void | a | output, reads |
| compact | final_peer | a | output, reads |
| compact | midgame_peer | a | outcome, reads |
| compact | cli | a | outcome, reads |
| compact | strategy | a | reads |
| compact | update | a | outcome, reads, after, after_hex |
| week | shadow | a | outcome, reads |
| week | void | a | output, reads |
| week | final_peer | a | output, reads |
| week | midgame_peer | a | outcome, reads |
| week | cli | a | outcome, reads |
| week | strategy | a | reads |
| week | update | a | outcome, reads, after, after_hex |
| leading-space | update | b | outcome, reads, after, after_hex |
| trailing-space | update | b | outcome, reads, after, after_hex |
| time | update | b | outcome, reads, after, after_hex |
| slash | update | b | outcome, reads, after, after_hex |
| dot | update | b | outcome, reads, after, after_hex |
| impossible | update | b | outcome, reads, after, after_hex |
| junk | update | b | outcome, reads, after, after_hex |
| empty | update | b | outcome, reads, after, after_hex |
| date-object | update | b | outcome, reads |
| none | update | D | after, after_hex |
| bytes | update | b | outcome, reads |
| integer | update | b | outcome, reads, after, after_hex |

## Tests changed in this attempt and why

Only the existing `tests/test_streak_season_cs2.py` was edited; no test file was added and no test function was deleted.

- Added `DATE_CASES` and `test_season_of_date_table`: the exact 18-row contract, including all four unpadded/padded combinations, compact/week, invalid strings and typed values.
- Added `test_row3_parseable_date_scores_once_with_original_marker`: all six parseable strings now score season-fresh, stamp 2027, retain the original spelling, and retry without changing bytes.
- Changed `test_row3_invalid_date_fails_before_io`: removed compact/unpadded strings from its rejection parameters because row L now expressly requires them to succeed; the new scoring test covers them. Added every unknown-season value except the dateless None sentinel, retaining the original invalid leap day/junk/time-suffix examples. Replaced the empty-directory assertion with a seeded file's exact-byte preservation assertion, and strengthened the no-I/O guard to forbid every state reader/writer and Path.open. This assertion change implements the requested stronger preservation proof, not a concession to a failing implementation.
- Extended `test_row6_scheduler_display_reads_scored_season`: 2 date values → all 18, for each of the three existing real polling modes. Original padded/unpadded assertions remain 0; newly unparseable values must display 13. Added the unchanged-streak assertion.
- Extended `test_row8_shadow_selection_uses_own_date_season`: 2 values → all 18, keeping its real shadow entry point and streak capture; unknown seasons must read 13. Explicitly pinned its clock.
- Changed fixture helper `resolved_pick`: stringify only the persisted lookup key and create its parent path, allowing typed date arguments and slash-containing lookup keys without asking JSON to serialize date/bytes objects. Existing string fixtures keep their values and behavior; the original typed date is passed to each entry point.
- Added `test_row9_saver_read_never_rejects_date`: observe the real saver fallback for all 18 values, including whether selection subsequently raises an existing base date error. The decision call must receive the correct saver in every case.
- Added `test_row6_cli_date_matrix_preserves_age_guard`: all 18 rows through the real Click command or typed callback; preserve upstream exceptions and verify read-only bytes. This test is already green on the starting branch because all rows reaching that branch's CLI read are supported by its existing fromisoformat call; it documents that fact rather than bypassing the guard to invent a regression.
- Imports add the date class solely for the required typed test row. Other test functions and the earlier adversarial test file remain unchanged.

Clocks affecting expected results are supplied or pinned. The matrix also pins persistence timestamps. No new assertion depends on today's date. The first attempt's fixtures, clock pins, and other assertions are preserved; a full inherited test-name audit is appended below.

## Every date parse or season derivation remaining in the whole source diff

Checked `git diff d8eb50d0 HEAD -- src`, not merely this commit:

- `picks.py:516`/525/529: the single `season_of_date` helper, date.fromisoformat then datetime.strptime, returning `.year` or None.
- `picks.py:685`: dated update derives its season only by that helper. Dateless None bypasses the derivation. All subsequent reads/writes carry that season; no marker normalization occurs.
- `scheduler.py:1899`, 2561, 2605, 2610: all four added reads use only `season_of_date(date)`.
- `cli.py:2438`: the void read uses only `season_of_date(date)`.
- `strategy.py:573`: the saver fallback uses only `season_of_date(date)`.
- `picks.py:1407`: replay save derives the season from `replay_day.year`. Its unchanged producer at lines 1403–1404 uses the timezone-aware reconcile clock converted to ET (`now_et`, line 1308), then `.date()`, `.year`, and `.isoformat()`. No date-string parser was added to reconcile. The unchanged replay marker check at line 742 calls `load_applied_dates` without a season, as required.
- `cli.py:2539`, 2543, 2546: summary converts `receipt.clock()` to ET and uses `.year`. It still samples the receipt's clock again after replay (accepted existing display-only boundary behavior, see below).
- `contest_state.py:345–347`: the existing ET `now_year` calculation (`now_dt.astimezone(ZoneInfo("America/New_York")).year`) now also supplies `_load_model_diagnostics`, which compares the exact integer tag at line 322. This code compares years and parses no date string.
- Reader/save helpers compare an explicit season argument and/or retain an existing tag; they do not infer a season from `updated`, markers, filenames, or another field. `_require_int_season` validates type, not dates.

The CLI's unchanged age guard uses `_date_cls.fromisoformat(date)` at line 2388. Strategy retains its unrelated strptime-based days-remaining logic. The replay's unchanged filename regex and ISO lexical boundary tests remain. None is a new season derivation in the diff. No `_season_of` remains.

## Verification

Every pytest command used the shared-machine wrapper, a log under `.codex-review/cs2b/`, `env TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider` (or the equivalent option order for the fast suite). No pattern kills were used. The wrapper watcher recorded no hold/pause during the completed baseline runs.

The targeted selection was generated before edits with:

```
rg -l 'streak\.json|load_streak|save_streak|update_streak|load_saver_available|load_applied_dates|reconcile_results|select_pick|run_result_polling' tests
```

The before and after lists are `baseline-files.txt` and `final-files.txt`: identical sets of 45 files. The wrapper commands passed their file lists as pytest arguments, followed by `-q`.

| Run | Log under .codex-review/cs2b/ | Result |
|---|---|---|
| Initial sandboxed targeted baseline | baseline-targeted.log | 1 failed, 1118 passed, 8 skipped; the sandbox blocked nice priority and the local HTTP socket in test_web_saver's socket wiring test |
| Targeted baseline rerun outside sandbox | baseline-targeted-retry.log | 1119 passed, 8 skipped, 6.66s |
| Fast baseline, before test edits | baseline-fast.log | 37 failed, 5094 passed, 19 skipped, 6 deselected, 1 xfailed, 16 errors, 1526.49s; EXACT known set of 53 failures/errors |
| RED, expanded CS2 file on unchanged ee9fb870 source | red.log | 93 failed, 146 passed, 2.51s |
| GREEN, expanded CS2 file after implementation | green.log | 239 passed, 0.60s |
| Final targeted selection | final-targeted.log | 1252 passed, 8 skipped, 6.72s |
| Final fast suite at dd915ac9 | final-fast.log | 37 failed, 5227 passed, 19 skipped, 6 deselected, 1 xfailed, 16 errors, 1485.93s; EXACT known set of 53 failures/errors |

RED breakdown: helper table 18 failures; dated scorer 5; shadow 14; polling 42 (14 per mode); strategy saver read 14. The remaining 146 parameter cases passed. The failure log was read before the source was changed. All inherited CS2 tests remained in the focused run. The targeted increase is 133 passing cases.

The exact fast command, before and after:

```
~/bts-runs/yield_run.sh .codex-review/cs2b/<baseline-fast.log or final-fast.log> env TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -m "not slow" --ignore=tests/simulate --ignore=tests/model --ignore=tests/experiment --ignore=tests/validate -q -p no:cacheprovider
```

The uncommitted `compare_gate.py` compares each failing/error node ID against `~/bts-runs/gate-failed.txt`. Baseline: 53 actual, 53 expected, zero extra/missing. Final: 53 actual, 53 expected, zero extra/missing. Both comparisons passed. The final fast suite adds exactly the same 133 passing cases as the targeted suite; all other counts are unchanged. The final wrapper exited rc=1 for the expected failing set and recorded no pause.

`git diff --check` passed before the commit. No forbidden row-I source was changed, and `contest_state.py` was not touched in this attempt. The commit includes exactly the user's five authorized files. Review scripts, raw comparisons, logs, and this handover remain uncommitted.

## Remaining obligations and limitations

- Confirmation B2 is closed: compact/week dates pass all three real polling branches and shadow selection; six parseable spellings score and retain their markers.
- Other previously regressed read inputs are closed: all 18 rows reach every scheduler read, with season 2027 reading 0 and unknown seasons reading the base's 13; the matrix compares the actual base package. No non-string scheduler row needed an upstream exception exemption.
- The dateless None matrix classification conflict is documented above. It is the only difference outside (a)/(b) and is required by rows B/D. It was not silently altered to force the matrix green.
- STOP round 2 N3: a saver-zone miss supplies an explicit saver, so corruption introduced after successful reads can be replaced. The base behaves the same way. Left unchanged because rows A–K must stay as they are and row B preserves the explicit-saver rebuild behavior; no broader concurrency/corruption redesign was attempted.
- STOP round 1 N1: state/export.py and state/regenerate.py drop the tag. Restored state consequently reads season-fresh until complete replay. Deferred as recorded; row I expressly keeps those files unchanged.
- STOP round 2 N2: reconcile summary samples the clock again after replay, so an ET New Year boundary can show a different season from the replay. Kept under the lead's recorded display-only ruling; the receipt-clock/ET test remains.
- Row K remains: a mid-season deploy reads model streak 0 until the 02:00 reconcile.
- Corrupt-state behavior, season type checks, tagless reads, first dated stamps, once-per-date/saver rules, replay marker protection, and diagnostics behavior are covered by the preserved targeted/adversarial tests. No further discrepancy in A–K was observed beyond the explicit dateless/base wording conflict and the listed accepted limitations.

## Inherited changed-test audit

Inherited test changes (d8eb50d0 → ee9fb870), preserved by this attempt:

- `tests/test_check_results_wait.py`:
  Season-tagged fixtures and/or pinned decision clocks; previous behavioral assertions retained:
  - `TestShadowReconcileOnEveryExit.test_production_pending_still_reconciles_shadow`
  - `TestShadowReconcileOnEveryExit.test_production_resolver_error_still_reconciles_shadow`
  - `TestStaleScoringGuard.test_old_unresolved_scoreable_production_never_scores`
  - `TestStaleScoringGuard.test_allow_stale_scoring_override_scores`
  - `TestStaleScoringGuard.test_yesterday_scores_without_override`
  - `TestWaitDeadline.test_wait_retries_until_production_resolves`
  - `TestWaitDeadline.test_deadline_is_a_hard_bound`
  - `TestWaitDeadline.test_default_is_single_attempt_no_sleep`
  - `TestWaitDeadline.test_wait_already_past_deadline_single_attempt`
  - `TestWaitDeadline.test_wait_settles_immediately_when_already_resolved`
  - `TestWaitDeadline.test_vanished_pick_file_stays_pending`

- `tests/test_cli_integration.py`:
  Season-tagged fixtures and/or pinned decision clocks; previous behavioral assertions retained:
  - `TestBtsRun.test_run_uses_fresh_contest_streak_for_live_action`
  - `TestBtsPreview.test_preview_uses_fresh_contest_streak_for_projected_pick`
  - `TestBtsCheckResults.test_check_results_hit_updates_streak`
  - `TestBtsCheckResults.test_check_results_miss_resets_streak`
  - `TestBtsCheckResults.test_check_results_none_warns_scratched`
  - `TestBtsCheckResults.test_check_results_voids_primary_and_scores_double_once`
  - `TestBtsCheckResults.test_check_results_voids_double_and_scores_primary_once`
  - `TestBtsCheckResults.test_check_results_all_void_keeps_streak`
  - `TestBtsCheckResults.test_check_results_skips_already_resolved`
  - `TestBtsCheckResults.test_check_results_resolves_shadow_when_production_already_resolved`
  - `TestBtsCheckResults.test_check_results_writes_shadow_status_artifact`
  - `TestBtsCheckResults.test_check_results_shadow_status_failure_does_not_crash`
  - `TestBtsCheckResults.test_check_results_skips_already_resolved_shadow`
  - `TestBtsCheckResults.test_check_results_resolves_shadow_with_unresolved_production`
  - `TestBtsCheckResults.test_check_results_shadow_double_down_is_dd_aware`
  - `TestBtsCheckResults.test_check_results_skips_unscoreable_skip_record`
  - `TestBtsCheckResults.test_check_results_scores_scoreable_decision`
  - `TestBtsCheckResults.test_check_results_missing_decision_falls_back_to_delivered`
  - `TestBtsCheckResults.test_check_results_missing_decision_undelivered_not_scored`

- `tests/test_contest_state.py`:
  Season-tagged fixtures and/or pinned decision clocks; previous behavioral assertions retained:
  - `test_model_state_used_when_no_contest_state`
  - `test_required_contest_state_missing_raises`
  - `test_fresh_contest_state_drives_live_decision`
  - `test_lagged_contest_uses_contest_value_keeps_doubles`
  - `test_lagged_contest_model_does_not_inflate`
  - `test_missing_source_date_is_stale_but_keeps_doubles`
  - `test_model_never_inflates_decision_streak`
  - `test_two_pick_gap_is_stale`
  - `test_unconfirmed_local_miss_marks_stale`
  - `test_unexpired_override_wins_over_auto_through_decision`
  - `test_saver_conservative_when_reached_zone_but_unconfirmed`

- `tests/test_incident_register_2026.py`:
  Season-tagged fixtures and/or pinned decision clocks; previous behavioral assertions retained:
  - `test_l01_check_results_single_pick`
  - `test_l01_check_results_hit_plus_pass`
  - `test_l01_check_results_pass_plus_pass_preserves`
  - `test_l01_pass_at_saver_streak_keeps_the_saver`
  - `test_l01_absent_player_stays_pending`
  - `test_l02_same_day_hit_survives_two_late_reconciles`
  - `test_l02_same_day_saver_consumption_survives_two_late_reconciles`
  - `test_l02_unplayed_undelivered_preview_is_excluded`
  - `test_l04_hit_applied_once_across_a_crash_and_restart`
  - `test_l04_saver_miss_applied_once_across_a_crash_and_restart`
  - `test_l04_polling_death_then_the_cron_scorer_applies_once`
  - `test_l04_control_without_the_fault_a_rerun_is_a_no_op`

- `tests/test_pickpath_review_pp1.py`:
  Season-tagged fixtures and/or pinned decision clocks; previous behavioral assertions retained:
  - `test_crash_reconcile_restart_keeps_applied_hit`

- `tests/test_pickpath_review_pp2.py`:
  Season-tagged fixtures and/or pinned decision clocks; previous behavioral assertions retained:
  - `test_delivered_hit_plus_void_double_adds_exactly_one`

- `tests/test_picks.py`:
  Season-tagged fixtures and/or pinned decision clocks; previous behavioral assertions retained:
  - `TestReconcileResults.test_no_corrections_when_results_match`
  - `TestReconcileResults.test_corrects_hit_to_miss`
  - `TestReconcileResults.test_preview_pick_for_tomorrow_does_not_break_streak_walk`
  - `TestReconcileResults.test_partial_void_hit_counts_as_single_streak_day`
  - `TestReconcileResults.test_shadow_pick_files_are_ignored_in_streak_walk`

- `tests/test_reconcile_cutoff.py`:
  Season-tagged fixtures and/or pinned decision clocks; previous behavioral assertions retained:
  - `_plant_hit`

- `tests/test_reconcile_receipt.py`:
  Season-tagged fixtures and/or pinned decision clocks; previous behavioral assertions retained:
  - `plant`

- `tests/test_scheduler.py`:
  Season-tagged fixtures and/or pinned decision clocks; previous behavioral assertions retained:
  - `TestPollResults.test_run_result_polling_voids_primary_and_scores_double_once`
  - `TestPollResults.test_run_result_polling_uses_schedule_void_when_live_feed_preview`

- `tests/test_scoring_lock.py`:
  Season-tagged fixtures and/or pinned decision clocks; previous behavioral assertions retained:
  - `test_check_results_skips_when_peer_scored_mid_flight`
  - `test_check_results_fails_closed_when_pick_vanishes`
  - `test_check_results_adopts_fresh_metadata_before_scoring`
  - `test_reconcile_results_saves_under_scoring_lock`

- `tests/test_streak_applied_dates.py`:
  Season-tagged fixtures and/or pinned decision clocks; previous behavioral assertions retained:
  - `test_the_same_date_is_applied_once`
  - `test_a_repeated_saver_miss_does_not_reset_the_streak`
  - `test_different_dates_each_apply`
  - `test_other_writers_of_the_streak_keep_the_marker`
  - `test_a_tagged_streak_file_without_a_marker_still_works`
  - `test_the_marker_list_is_bounded`

- `tests/test_streak_season_cs2.py`:
  New contract/adversarial tests for season isolation, corruption, idempotency, saver boundaries, rollover and replay:
  - `plant`
  - `snapshot`
  - `test_row1_readers_require_an_exact_integer_season_without_writing`
  - `test_row1_dateless_corruption_behavior_is_unchanged`
  - `test_row1_a_corrupt_file_is_an_error_with_a_season_too`
  - `test_row3_dated_update_on_a_corrupt_file_raises_and_preserves_it`
  - `test_row2_seasonal_save_on_a_corrupt_file`
  - `test_row3_a_date_object_is_a_value_error_before_any_io`
  - `test_row1_missing_file_and_missing_keys_keep_defaults`
  - `test_row2_save_resets_only_preserved_state_on_mismatch`
  - `test_row2_dateless_save_preserves_tag_verbatim`
  - `test_row3_legacy_rollover_then_same_date_retry`
  - `test_row3_matching_season_retains_saver_forgiveness`
  - `test_row3_tagless_three_starts_at_one_instead_of_carrying_over`
  - `test_row3_invalid_date_fails_before_io`
  - `test_row4_dateless_update_preserves_tag_and_state`
  - `test_row4_tagless_dateless_update_keeps_legacy_state`
  - `resolved_pick`
  - `test_row5_reconcile_stamps_replayed_season_and_discards_old_markers`
  - `test_row5_tagless_same_day_marker_still_blocks_incomplete_replay`
  - `test_row6_cli_void_display_reads_scored_season`
  - `test_row6_scheduler_display_reads_scored_season`
  - `test_row7_reconcile_summary_uses_replay_et_clock`
  - `test_row8_shadow_selection_uses_own_date_season`
  - `test_row9_strategy_saver_fallback_uses_selection_date`
  - `test_row10_model_diagnostics_reset_without_changing_contest_or_saver_authority`
  - `test_row11_tagless_reads_do_not_migrate_then_first_dated_write_stamps`
  - `test_row12_tagged_file_still_validates_and_exports`
  - `test_row9_strategy_accepts_the_unpadded_date_it_accepted_before`
  - `test_a_season_argument_that_is_not_an_int_is_a_type_error`
  - `test_row5_reconcile_rebuilds_a_deeply_nested_file_as_it_did_before`

- `tests/test_streak_season_cs2_review.py`:
  New contract/adversarial tests for season isolation, corruption, idempotency, saver boundaries, rollover and replay:
  - `_pick`
  - `_plant_daily`
  - `test_b1_deeply_nested_streak_json_isolated_by_season_readers_and_reconcile`
  - `test_sequence_2026_to_2027_hit_racing_scorers_double_crash_and_reconcile`
  - `test_sequence_opening_night_miss_does_not_trigger_2026_saver_zone`
  - `test_sequence_opening_night_void_keeps_legacy_file_untouched_then_day2_stamps`
  - `test_mid_season_saver_exact_boundaries_on_tagged_file`
  - `test_mid_season_second_miss_after_spent_saver_resets_to_zero`
  - `test_corrupt_streak_json_stops_the_scorer_and_is_left_intact`
  - `test_contest_dates_dec31_to_jan01_boundary`
  - `test_reconcile_summary_end_to_end_at_utc_et_new_year_boundary`
  - `test_row12_backup_verify_and_export_accept_season_tagged_streak`
