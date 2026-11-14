# PP-C1 kickoff 2 handover

Implemented the replacement contract: no leg reduction. No commit, push, stash, branch change, deployment, real-account access, or delegation.

## Files and functions
- `src/bts/scheduler.py::_deliver_and_lock_pick`: the status guard refuses only when all distinct committed games are void. A partial-void double gets one stderr line naming game/status, then follows the existing cutoff/delivery path unchanged, with its original SelectionResult.
- Deleted `_playable_leg` and `_keep_playable_leg`, including mutation, archive/save, reduction record, alert dispatch and selection reset. `_refuse_void_game_delivery`, `_archive_and_remove_pick`, cutoff and transport branches are unchanged.
- `src/bts/health/late_delivery.py::check`: deleted the `leg_dropped_game_not_playable` branch.
- `tests/test_postponed_delivery_guard.py`: changed surviving-leg and selection assertions to unchanged doubles; retained the original earliest cutoff; removed the planted reduction WARN. Expanded single/all-void refusal coverage across DM/public/private and same-game doubles, retaining archives, CRITICAL game/status alerts and no batter names.
- `tests/test_pickpath_review_pp1.py::test_postponed_dd_does_not_erase_playable_primary`: keeps the real day-level B2 path, now requiring delivery and decision as a double (with and without the guard control).
- `tests/test_pickpath_review_pp2.py`: replaced reduction-only tests with the contract matrix, caught-send health checks, reference-compared restart/error days, corrected-status case and both real scorers for hit+Pass doubles.
- This required handover is the only new worktree file outside source/tests. Scratch reference source, loader and execution logs are `/private/tmp/ppc2-*`.

## Deleted or replaced pp2 tests
Deleted the old `test_reduction_updates_the_callers_object`, `test_reduction_does_not_archive_the_double`, `test_reduction_does_not_dispatch_an_alert_dm`, `test_only_normal_delivery_saves_replace_the_double`, and `test_reduction_records_one_warning_without_a_delivery_claim`: they asserted a reduction that no longer exists. The useful no-alert/no-extra-save checks moved into the unchanged-double matrix.
Replaced `test_death_before_delivery_save_keeps_double_and_restart_delivers_single` with the reference-compared restart matrix; `test_survivor_file_commit_capture_grade_and_replay_are_single` with real hit+void double scoring; and `test_failed_single_delivery_is_not_reported_as_delivered` with caught-send failure checks requiring the undelivered double and no reduction health alert.

## Test names / RED → GREEN
Aliases below refer to `tests/test_pickpath_review_pp2.py` unless a file is specified.
- A = `test_partial_void_double_is_delivered_without_reduction` (either void leg × DM/public/private).
- B = `test_caught_send_failure_keeps_double_and_has_no_reduction_health` (either void leg × DM/public).
- C = `test_cached_double_restart_matches_pre_guard_outcome` (either void leg × mode × equal/earlier void start × state-save death, delivery-save death, contest-state error or caught send failure).
- D = `test_corrected_status_after_pre_send_error_matches_pre_guard_loss`.
- E = `test_delivered_hit_plus_void_double_adds_exactly_one` (either void leg × CLI/daemon scorer).
- F = `tests/test_postponed_delivery_guard.py::test_the_decision_record_keeps_the_original_double_selection`.
- G = `tests/test_postponed_delivery_guard.py::test_a_postponed_or_cancelled_game_is_refused_archived_and_alerted`.
- H = `tests/test_postponed_delivery_guard.py::test_a_double_with_both_legs_postponed_is_refused`.

Before any changes, existing focused set: `644 passed in 17.08s` (`ppc2-baseline.log`).
After writing the replacement tests, before production edits: `47 failed, 25 passed, 8 skipped in 8.94s` (`ppc2-red.log`). A/B fail because the reducer removes a slot; F sees a single decision; B2 and the delivered C/E cases see singles or a missing pick rather than the required double. C/D's reference-lost cases already pass. The final A save callback additionally asserts original slot identities before comparing their values.
After production edits, changed files: `83 passed, 8 skipped in 5.96s` (`ppc2-green.log`). All remain green in the final set below.

| Contract item / obligation | Test pins | RED evidence | Final GREEN |
| --- | --- | --- | --- |
| 1: partial-void double unchanged in every mode | A; pp1 B2; C | A six failures; B2 guard-on fails double assertion; delivered C cases fail double or missing-file assertion | A 6; B2 2; C 40 passed, 8 skipped |
| 1/2: no slot mutation, extra save, refusal record or alert | A/B | A/B lose DD under old reducer; separate mutations below isolate each invariant | A 6; B 4 passed; same slot identities/values, normal save counts, no refusal/dispatch |
| 1/2: original double decision and selection | A/F | F action is `single` before edits | A 6; F 1 passed, including source `mdp` |
| 1: one stderr game/status line | A | Old log says reduction; log-removal mutation fails all six cases | 6 passed |
| 2: no reduction health wording | A/B; planted non-reduction refusal test | B saves a single and old reducer produces reduction state; record mutation fails A | 10 passed; health returns no alerts on these ample-margin paths |
| 3: all-void refusals, cutoff ordering, unreadable schedule | G/H; pp1 delayed-private-cutoff; earliest-cutoff and unreadable tests | Already passing behavior retained; refusal mutations below kill G/H | G 9; H 6 passed; all existing cutoff/outage checks green |
| Pass scoring matches a single | E | Old reducer leaves no DD, so double prerequisite fails | 4 passed; real slot results hit+void, streak 1→2, increment exactly 1; replay stays 2 |

## Per-scenario reference comparison
Before production edits, exported the ENTIRE scheduler with `git show 4199360:src/bts/scheduler.py` to `/private/tmp/ppc2-reference-scheduler.py`; the scratch runner executes it into the imported scheduler module, then runs the same fixtures/tests. Other application modules are the unchanged worktree versions. No source checkout or branch change.
Reference command: `TZ=America/New_York .venv/bin/python /private/tmp/ppc2-run-reference.py tests/test_pickpath_review_pp1.py::test_postponed_dd_does_not_erase_playable_primary tests/test_pickpath_review_pp2.py::test_cached_double_restart_matches_pre_guard_outcome tests/test_pickpath_review_pp2.py::test_corrected_status_after_pre_send_error_matches_pre_guard_loss -q -s -p no:cacheprovider --tb=short`.
Reference: `43 passed, 8 skipped in 3.01s` (`ppc2-reference.log`). Current: same targets through `.venv/bin/python -m pytest`, `43 passed, 8 skipped in 3.68s` (`ppc2-current-scenarios.log`). Compared all 43 printed outcome tuples: exact equality, including final file/decision and successful transport call counts.

| Scenario (both void-leg orientations unless stated) | Cases | Reference → current | Interpretation |
| --- | --- | --- | --- |
| pp1 B2 failed-refresh final fallback, guard/control | 2 | delivered double → delivered double | Delivery preserved |
| Equal 19:10 starts, death before normal pick save | 6 | committed double → committed double | Original death-before-save restart preserved |
| Equal starts, death at normal state save | 6 | committed double → committed double | Preserved |
| Equal starts, caught pre-send contest-state error | 4 | delivered double → delivered double | Delivery preserved |
| Equal starts, caught transport failure | 4 | delivered double on retry → same | Delivery preserved |
| Void 13:40 / playable 19:10, pick-save death; restart 13:36 | 6 | no pick/decision/send → same | Pre-existing stale-cutoff archival loss |
| Earlier void cutoff, state-save death, DM/public | 4 | delivered double retained → same | Normal state save is after send; restart does not resend |
| Earlier void cutoff, state-save death, private | 2 | no pick/decision → same | Pre-existing loss before private decision is saved |
| Earlier void cutoff, caught pre-send contest-state error | 4 | no pick/decision/send → same | Pre-existing loss; no survivor-only recovery added |
| Earlier void cutoff, caught transport failure | 4 | failed attempt, no successful retry → same | Pre-existing loss |
| Corrected status/start after pre-send error (primary void, DM) | 1 | no pick/decision/send → same | Pre-existing cached-time cutoff loss |

Eight private contest-state/transport cases skip because private has neither operation. The reduction's pre-send state-save boundary was deleted; the state-save injections now target the existing normal lock save, exactly as they do in the reference. Pick-save injections still die before any normal pick save writes.

## Mutations (all restored)
Logs: `/private/tmp/ppc2-mutant-<name>.log`. Each source mutation was executed sequentially, its named tests failed, then original source was restored.

| Mutation | Killed by | RED last line |
| --- | --- | --- |
| `refuse_any_void`: refuse a partial-void double | A | `6 failed in 2.62s` |
| `never_refuse_void`: remove all-void refusal | G/H | `15 failed in 1.55s` |
| `count_slots_not_games`: compare against 2 slots rather than distinct games | H same-game cases | `3 failed, 3 passed in 1.25s` |
| `reduce_double`: restore active-slot reduction | A | `6 failed in 3.20s` |
| `add_refusal_record`: add partial-void refusal state | A | `6 failed in 1.96s` |
| `dispatch_alert`: add partial-void WARN dispatch | A | `6 failed in 2.16s` |
| `extra_save`: separately save the partial-void double | A | `6 failed in 1.87s` |
| `reset_selection`: clear the original SelectionResult | F | `1 failed in 1.20s` |
| `omit_stderr`: remove the partial-void line | A | `6 failed in 1.94s` |

## Final focused run / limits
Ran the kickoff's exact focused set plus `tests/test_pickpath_review_pp2.py`, using `TZ=America/New_York .venv/bin/python -m pytest ... -q -p no:cacheprovider` (`ppc2-focused.log`).
Last line: `======================= 691 passed, 8 skipped in 15.16s ========================`
`git diff --check` passed. Only scheduler, late_delivery and the three named test files differ; reduction helpers/reason/reset are absent from production source.
Known costs retained exactly as instructed: DM/decision name the void leg; entry checks may flag it; cutoff uses the original earliest leg, including a void leg. No attempt to repair those costs or the measured pre-existing day losses.
No whole suite, remote gate, new review round, model training, historical outcome read, prohibited-directory read or real transport. Artifacts and clocks are synthetic; crash exceptions test write interleavings, not OS power-loss durability. No unresolved implementation obligation identified.
