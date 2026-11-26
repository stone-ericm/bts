# PP-C1 author handover

Implemented the lead's narrower contract. No commit, push, stash, branch change, deployment, account access, or agent delegation.

## Files and functions
- `src/bts/scheduler.py`: `_keep_playable_leg` now mutates the caller's DailyPick, records the reduction, and saves scheduler state only. Removed archive/unlink, separate pick save, WARN dispatch, and replacement-object return. Removed unused config/date arguments.
- `_deliver_and_lock_pick` calls that helper without rebinding `daily`; retains `selection = None`. The existing cutoff and delivery branches are unchanged. All three `run_day` shadow sites consequently see the surviving primary through their existing object references.
- `src/bts/health/late_delivery.py::check`: reduction WARN says `kept <batter>` without a delivery claim.
- `tests/test_pickpath_review_pp2.py`: new regressions adapted from the supplied reviewer probes; imports resolve within this worktree. Adds caller identity, disk/save ordering in DM/public/private, no archive, no alert, record/health checks, and state-save/delivery-save deaths plus real run_day restart and inline shadow comparison for either survivor.
- `tests/test_postponed_delivery_guard.py`: updated `test_a_postponed_double_down_leg_is_dropped_and_the_primary_delivered_alone` to require no archive and no alert dispatch; updated `test_the_health_check_words_each_refusal_by_its_reason` to require neutral kept-leg wording. These old assertions pinned behavior the contract removes. No tests deleted.
- This required handover is the only added worktree file outside source/tests. Full temporary execution transcripts are `/private/tmp/pp-c1-*.log`.

## Test names used below
All T1–T8 are in `tests/test_pickpath_review_pp2.py`:
- T1: `test_reduction_updates_the_callers_object` (either surviving leg).
- T2: `test_reduction_does_not_archive_the_double`.
- T3: `test_reduction_does_not_dispatch_an_alert_dm` (also exactly one reduction stderr line).
- T4: `test_only_normal_delivery_saves_replace_the_double` (DM/public/private).
- T5: `test_reduction_records_one_warning_without_a_delivery_claim`.
- T6: `test_death_before_delivery_save_keeps_double_and_restart_delivers_single` (two survivors × state-save death, delivery-save death, no fault).
- T7: `test_survivor_file_commit_capture_grade_and_replay_are_single` (two survivors × CLI/daemon scorers).
- T8: `test_failed_single_delivery_is_not_reported_as_delivered`.
In `tests/test_postponed_delivery_guard.py`:
- T9: `test_the_decision_record_describes_the_single_not_the_selection_that_chose_a_double`.
- T10: `test_the_kept_leg_still_faces_the_cutoff`.

## Obligations and contract: RED → GREEN
Existing required focused set, before edits: `625 passed in 15.89s`.
Before production edits, new tests: `14 failed, 5 passed in 3.33s` (`pp-c1-red.log`).
Updated old assertions before production edits: `2 failed, 11 passed in 1.42s` (`pp-c1-existing-red.log`).
After implementation, new + postponed files: `32 passed in 2.71s` (`pp-c1-green.log`).
Final focused run below passes every named test; GREEN counts below are cases in that run.

| Obligation / contract | Test and RED evidence | GREEN |
| --- | --- | --- |
| C1; item 2, death/restart | T6: both delivery-save deaths ended `assert (None is not None)`; state-save deaths failed byte-identical double retention. Real restarted run_day, failed prediction cascades, normal fallback. | 6 passed; cached double bytes preserved before save, one single DM after restart |
| Item 1; shadow note | T1: `assert daily.pick is kept and daily.double_down is None` failed twice. T6 `[None-True]`: expected shadow `McNeil`, actual `Kwan`. | T1 2 passed; T6 6 passed, including actual inline shadow argument |
| Item 2, no archive/save | T2: archive existed. T4: first save saw `FileNotFoundError`; old helper had unlinked the double. | T2 1 passed; T4 3 passed, only normal path's save sequence |
| Item 3; WARN-dispatch note | T3: `Expected 'dispatch_dm_for_health_alerts' to not have been called. Called 1 times.` | 1 passed; no alert call, one pick DM, one reduction stderr line |
| Item 4, record | T5: `assert 'archive' not in record` failed. Duplicate/kept/void-field mutations below each fail T5. | 1 passed; one entry per reduction call with reason, void games/states, kept batter; persisted WARN |
| Item 4; health-wording note | T8: old health message `Kwan was delivered alone` after `OSError`; false-delivery assertion failed. Updated planted-record wording test also failed. | T8 1 passed; undelivered pick, one neutral reduction WARN |
| Item 5, commit selection | T9 already green before production edits; stale-selection mutation gives `1 failed in 1.41s`. | 1 passed; single action, no DD, real kept candidate rather than stale selection |
| Item 6, cutoff/path | T10 already green before production edits; moved-cutoff mutation gives `assert (True is False)`, `1 failed in 1.95s`. | 1 passed; retained leg refused after its own cutoff |
| Required scorer regressions | T7 initially 4 passed; retained-DD mutant produces `4 failed in 1.82s` (delivery/capture contract fails before grading). | 4 passed; each real scorer increments 1→2, one hit slot, increment exactly 1; replay stays 2 |

T7 also reads the actual saved pick with the production snapshot loader and cached shadow evaluator: both see one slot. These are synthetic fixture artifacts, not historical outcomes or a real transport.

## Source mutations, executed sequentially and restored
Each row names a source change actually executed against the named tests; all failures were assertions, not collection errors. Full output: `/private/tmp/pp-c1-mutant-<name>.log`.

| Mutant | Killed by | RED last line |
| --- | --- | --- |
| `copy_instead_of_in_place`: copy daily at call site | T1 both; T6 `[None-True]` | `3 failed in 2.76s` |
| `archive_and_unlink`: restore archive/removal before mutation | T2; T6 `[delivery_save-False]` | `2 failed in 1.98s` |
| `separate_save`: add helper save before normal delivery | T4 all modes | `3 failed in 1.93s` |
| `alert_dispatch`: add separate WARN dispatch | T3 | `1 failed in 2.06s` |
| `duplicate_record`: append the reduction twice | T5 | `1 failed in 1.69s` |
| `missing_kept_batter`: store kept=None | T5 | `1 failed in 1.85s` |
| `missing_void_games`: store void_games={} | T5 | `1 failed in 2.61s` |
| `false_delivery_claim`: restore old health success wording | T8 | `1 failed in 3.51s` |
| `wrong_warning_level`: render reduction CRITICAL | T5 | `1 failed in 1.97s` |
| `stale_selection`: omit selection reset | T9 | `1 failed in 1.41s` |
| `no_reduction_log`: remove stderr line | T3 | `1 failed in 1.96s` |
| `retain_double_down`: omit clearing DD | T7 all cases | `4 failed in 1.82s` |
| `bypass_kept_cutoff`: move chokepoint cutoff one day later | T10 | `1 failed in 1.95s` |

The first cutoff-mutation harness refused a non-unique source anchor before changing source; reran with the unique cutoff-computation anchor above. No mutant remains.

## Final verification and limits
Ran exactly the kickoff's focused set plus `tests/test_pickpath_review_pp2.py`, with `TZ=America/New_York .venv/bin/python -m pytest ... -q -p no:cacheprovider`.
Last line: `============================= 644 passed in 18.79s =============================`
`git diff --check` passed. Only the four source/test files above differ from the clean starting tree (plus this ignored handover).
No full suite, remote gate, new review round, model training, real send, or prohibited directory/outcome reads. The lead lands/reviews the change.
One entry is appended per reduction invocation; a crash after the state save can leave that diagnostic and cause another reduction record on restart. Pick recovery does not depend on deduplicating those records. Existing send-attempt crash semantics after the normal delivery save are unchanged.
No unresolved implementation obligation identified. Crash tests inject process-death exceptions at write boundaries; they do not exercise OS power-loss durability.
