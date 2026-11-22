# W3 handover

Base commit `ceab9854`. Implementation commit `f7a8f6011c58a5e0081950e3e2cf946d6e1ee04d`.
Save-hook follow-up commit `59b72c044b7e6969e35c72c1ae3a02e852c7e462`.

`focused-post-loop.log` reports 331 passes and 4 skips in one process, including `test_lightgbm_optional.py` and `test_web_saver.py`. The wrapper exited successfully. Eric ran the separate fast-suite gate for the initial commit.

## Implementation

- `src/bts/picks.py:405` scopes byte capture to the scheduler's original call to `save_pick`. The save observes the same ASCII JSON serialization passed to `atomic_write_text` once the write succeeds and before the audit append. The observer accepts the first write of that candidate. A replacement during the append cannot change the captured bytes. The public `save_pick` still returns a path.
- `src/bts/picks.py:591` loads and parses the fallback pick from one byte read. The opt-in regular-file reader uses a nonblocking open and verifies the opened descriptor before reading. The initial fallback reload keeps the existing file behavior.
- `src/bts/scheduler.py:1953` returns the original `LockDecision` and saved bytes on successful unlocked saves. `run_day` at line 3009 consumes the result from the current check at or after its fallback deadline, with no fallback sleep and with matching same-date pick bytes.
- `src/bts/scheduler.py:905` checks the selected rows against the actual committed slots before using selection metadata. A single can retain its same-cycle policy runner-up. A cached pick supplies its own rows and persisted policy metadata.
- `src/bts/scheduler.py:922` reads and validates the decision record on the same-byte path. A scoreable record wins and gates result polling on slot agreement. Recovery uses the durable flags on the latest pick. The extra reads after reuse polling protect new evidence before the scheduler archives or delivers a candidate.
- The final ruling keeps the existing readers and eligibility for the post-loop fallback, missed-pick alert and result polling. LW3-4 therefore applies to those reads. When reuse discovers an authoritative scoreable record, result polling still requires slot agreement.
- `src/bts/scheduler.py:981` shares a two-second elapsed-time budget between pending lineup confirmations and the defer-only status lookup. Each request publishes its own result within the budget. Daemon threads may finish an already started request after the scheduler proceeds. The scheduler ignores those results. A reuse-based defer keeps overdue scheduled checks available.
- `src/bts/scheduler.py:1293` adds the keyword-only `reused_check_cascade: bool = False`. Only the in-loop reuse delivery passes `True`. The same absolute cutoff is checked again after persisting an attempt. Live contest state is read again if the first read crosses into a new ET day. Transport checks the validated ET date again and clears a known-unsent marker on refusal.

## Baseline test results

`baseline-final.log` used the `ceab9854` package source from a temporary Git archive and the final test files from this worktree. It reported 34 failures, 296 passes and 4 skips. “Red” below refers to the specific assertions on that base. Several controls check both the legacy outcome and that the updated scheduler reached it through reuse.

| Row | Test and result on `ceab9854` |
| --- | --- |
| M1 | `test_real_overrun_reuses_the_same_candidate_before_cutoff` was red. The real 18:25 check ran a second cascade at 19:00. |
| M2 | `test_fallback_reload[replacement,audit_replacement,whitespace]` passed for a changed batter, a write during the audit append, and a whitespace-only change. `test_warmup_cached_selection_commits_saved_pick_rows` was red at the required reuse assertion and verifies the committed row on reuse. |
| M3 | `test_fallback_reload[sleep]` and `test_kwan_delivered_at_13_20` passed. `test_fallback_reload[equality]` and `test_reuse_defer_keeps_an_overdue_confirmed_check_available` were red. The latter verifies that the next check runs independently after defer. |
| M4 | Both cases of `test_standing_mdp_skip_survives_cache_and_yields_to_genuine_pick` passed. `test_run_day_genuine_mdp_skip_still_writes_a_skip` also passed. |
| M5 | `test_fallback_reload[late_read]` passed. `test_pending_window_replans_without_losing_later_checks[reload_cutoff]` and both channels of `test_reuse_refuses_when_attempt_write_reaches_cutoff` were red. The attempt-write test pairs the current path with an empty legacy refresh that would consume the remaining time. |
| M6 | `test_fallback_reload[best,saver]` passed. The updated real-check consumer in M12 exercises refusal after checking the produced fields. Existing saved-pick refusal and missed-pick controls ran in the same suite. |
| M7 | Both channels in `test_reuse_confirms_stale_automatic_reading` and both cases in `test_reuse_warns_about_stale_reading_only_after_confirmed_send` were red at the reuse assertion. Existing stale warning and recovery controls passed. The new tests cover a missing transport identifier and a caught failure that clears its attempt and retries. |
| M8 | `test_fallback_reload[void]` passed. `test_pending_window_replans_without_losing_later_checks[void]` was red. `test_reuse_keeps_both_slots_when_only_one_leg_is_void` was red at the reuse assertion after verifying both committed legs. |
| M9 | `test_fallback_reload[absence,dangling,json,shape,invalid_time,unreadable_pick,directory,fifo]` passed. |
| M10 | `test_fallback_reload[initial_attempt]` passed. `test_fallback_reload[scoreable,matching_scoreable]` and the `attempt_deliver` and `attempt_defer` pending-window cases were red. Existing delivered, attempted and committed start-of-check controls passed. |
| M11 | All four `test_early_check_returns_keep_their_shape` cases passed, including propagation of a failed candidate write. The standing-skip test also checks that the surfaced skip carries neither field. |
| M12 | The changed `test_check_overrunning_deadline_reuses_its_cascade` fixture passed. Previously it supplied only the decision. It now also supplies the exact bytes from the fixture save. It still asserts no refresh, one DM and reuse in history. `test_overrun_reloads_the_same_cycle_choice_and_checks_it_as_saved` was red at the new producer assertion. Previously it injected a decision. It now asserts that the real check returned a decision and bytes matching the saved file before applying the same dated correction. Its reuse, refusal, no-send and no-commit assertions remain in place. The slept-fallback and duration assertions passed. |

The final ruling adds `test_missed_pick_alert_survives_abandoned_reuse`. It changes the saved bytes before the extra read and corrects the dated observation. It verifies that the independent post-loop refresh runs, the contest-state matcher refuses the pick, and the existing missed-pick alert still fires. The existing FIFO mutation test asserts only that the in-loop deferral is abandoned and a later check can recover. That test required no adjustment.

## Save-hook regression

The gate for `f7a8f601` found that the first W3 implementation bypassed the `picks.save_pick` hook used by both fresh-delivery observation correction cases in `tests/test_wr_a_pick.py` and by the scheduler midnight case in `tests/test_wr_a_review_carbon.py`. The guard read the old observation because the correction or time change never ran. The follow-up restores the original save call and observes the bytes inside that call. The three existing regression tests retain their assertions.

`focused-save-hook.log` reports 541 passes and 4 skips. The wrapper exited successfully. This run included the focused W3 set, all `tests/test_wr_a_*.py` and `tests/test_wr_b_*.py` files, and the three reported regressions. The fast suite was not run for the follow-up.

## Review round 1, Amendment 1

Amendment 1 commit `82acab73718fbf85d0e298d33dac8c730720747e`. The fast-suite gate is assigned to Eric.

- R1 extends the pending-window test with a scoreable `locked_unconfirmed` record and an invalid decision record, each published alongside an attempt during the added poll. Both were red on `59b72c04`. The scoreable case checks that the record bytes survive and that a second call to the alert helper is deduplicated. The invalid case exercises recovery from the pick's attempt marker.
- R2 adds `test_reuse_replans_gap_after_status_read` with and without a best-only correction. Both were red on `59b72c04`. The status read advances the clock past the last feasible gap window. The updated scheduler replans with the final live time and can deliver the primary. With the correction, `contest_state` and `missed_pick` both fire.
- R3 extends the cutoff and ET rollover cases with a transient write failure during marker clear. The clear retries once within the refusal path. A persistent failure propagates before transport. Both cutoff channels and both ET rollover dates were red on `59b72c04`. The cutoff cases were verified on the review archive of that source.
- The existing pending-window test joins the delayed worker and checks the live confirmation set after completion. Its new shared-budget case measures the total wait for a delayed confirmation followed by a blocked status body. Both checks passed on `59b72c04`.
- The existing abandoned-reuse test was red on `59b72c04` when it checked the journal. The reuse line now prints after the extra checks pass. It reports that this in-loop fallback used the check's cascade. A subsequent independent final fallback can still run if delivery is refused.

The Amendment 1 focused run, including both whole-range test families, reported 550 passes and 4 skips in `round1-focused.log`. In `round1-mutations.log`, the late-result mutation failed after it added game 103 to the confirmation set, and the split-budget mutation failed at 3.22 seconds.

## Review round 1, Amendment 1b

Amendment 1b commit `c0917aa750005ac8f3b89c0c414f2539508f1eb3`.

- R4 adds both cases of `test_reused_postponed_pick_takes_void_refusal_at_cutoff`. Both were red on `82acab73`, where they recorded `past_submission_cutoff`. The deliver case now uses the standard delivery guard. The defer case honors the known all-void status when the scheduler-state save crosses the cutoff. Both record `game_not_playable`, send the existing `postponed_pick` alert without naming the batter, and leave no delivery or commit.
- R5 adds inline shadow assertions to `test_fallback_reload[matching_scoreable]` and the four pending-window attempt cases. All five were red on `82acab73`. Early recovery now uses the same inline shadow trigger and existing “Pick locked. Stopping lineup checks.” journal line as other in-loop fallback locks. The line reports the lock, including recovery of an earlier attempt, and gives no instruction to enter a pick.

`round1b-red.log` reports seven failures across R4 and R5. `round1b-focused.log` reports 552 passes and 4 skips, including the LightGBM and web-saver controls and both whole-range test families. The wrapper exited successfully. Eric owns the fast-suite gate.

## Second contract check

| Item | Test and result on `ceab9854` |
| --- | --- |
| LW3-1 | Both cases of `test_reuse_refuses_when_attempt_write_reaches_cutoff` were red. Both channels take the existing late refusal after clearing the known-unsent marker. |
| LW3-2 | All six cases of `test_reuse_checks_live_day_at_midnight` were red. They cover the rollover before the read, during the read, and during the attempt write at ordinary midnight and New Year, with first pitch after midnight. |
| Item 4 decision read | The `scoreable` and `matching_scoreable` rows of `test_fallback_reload` were red. The decision FIFO, unreadable, invalid, wrong-date, empty, and dangling cases passed as legacy controls. M1 covers true absence. |
| Item 4 and LW3-5 | The `initial_attempt` control passed for changed bytes at the first reload. The `attempt_deliver`, `attempt_defer` and `changed_decision_fifo` pending-window cases were red. The FIFO case verifies that the next scheduled check can proceed once the file is corrected. |
| Items 5 and 6 added read budget | The `confirm`, `slow_status` and `partial` pending-window cases and the overdue-check replay were red. The slow lookup fixtures release their worker with a bounded wait after verifying that the later check could send while the lookup was still pending. |
| M9 wording | The M9 controls passed. Absence and a dangling link return absence. A link to a readable regular file is followed. Invalid JSON, unsupported shape, read failures and directories propagate their existing errors. An invalid slot time surfaces the existing timestamp error. The legacy FIFO reader can block. The test bounds it with a process alarm and checks that the path survives. The additional reuse reads use the separate regular-file check. |

## Standing rows

- Both `KeyboardInterrupt` and `SystemExit` escape before the attempt marker, including when the added poll raises. The two poll interruption controls passed on the base. All six `test_restart_uses_durable_delivery_state` cases were red at the reuse-history assertion. A restart after an unconfirmed attempt recovers `locked_unconfirmed` and issues the existing unknown-outcome alert. A restart after durable confirmation recovers the pick. The scoreable record controls verify preservation of existing committed slots. A before-marker restart uses the identical saved bytes and persisted reuse history to verify that telemetry cannot authorize reuse.
- The existing 4c catch still applies to interruptions in its own pre-send telemetry block. LW3-1 and LW3-2 remain on existing delivery paths. The M9 behavior and the initial changed-byte refresh remain as specified for LW3-4 and LW3-5.
- The skipped second prediction can miss a new lineup, replacement selection or new skip. A standing genuine skip still suppresses cached delivery until a genuine choice supersedes it. The final durable reads leave the existing race with unsynchronized writers after those reads.

## Operator text

The existing reuse, late refusal, postponed refusal and unknown-outcome messages are reached on the new paths. The only added message text is “ET date changed before contest state could be validated” and “ET date changed after the final contest-state read”. Both use the existing contest-state refusal and alert. The first can occur when a live date cannot be validated during the read and its single retry. The second can occur if the date changes after the final read, including while the attempt is being saved. A pick is withheld and any known-unsent attempt from this call is cleared. Neither sentence asks the operator to take an action.
