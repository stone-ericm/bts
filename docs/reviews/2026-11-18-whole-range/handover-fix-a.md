# Whole-range fix set A

Branch `fix/wr-a` from `9e0ec28a`. Single fix commit `9cb84536` incorporates amendments 1–3.

## Changes and base evidence

- **A1** — `src/bts/strategy.py:40`, `CLAUDE.md:63`, and checklist B5 at `docs/ops/2027-season-start.md:32`. The exclusive boundary is September 27 for the September 26, 2027 finale. The unchanged shared subtraction yields 187 on March 24 and 1/0 on September 26/27. `tests/test_wr_a_calendar.py:19` replays the bundled reach-57 single and tail double through delivery. Both original Slice 2 base replays skipped. The RED pytest run also caught the separate September 25 streak-55 change from double to skip and the shift of the health CRITICAL window. Its September 20 fixture initially lacked trusted provenance and got a single on the base. The corrected fixture supplies trusted automatic provenance, as the original base replay did.
- **A2** — `src/bts/scheduler.py:1280`, `src/bts/cli.py:1215,1267`, and `src/bts/orchestrator.py:396,446`. After a successful required final read, both fresh and saved pick-post paths call the same `saved_pick_refusal` on the original stamp. The scheduler returns False and attempts its existing contest-state dispatch. The CLIs exit 1 with the existing refusal and without a CRITICAL DM. The on-disk selection and check telemetry survive. The named cases at `tests/test_wr_a_pick.py:67,118` cover Slice 3's post-save timing and Slice 1's in-request timing. The base sent the obsolete pick on DM, public, both CLIs and the fresh fallback. The corrected scheduler tests inspect the refused attempt at the delivery boundary before allowing a later successful fallback or lineup check.
- **A5** — `src/bts/scheduler.py:284` validates the full hydrated response shape before filtering and reports bad fields or JSON with `ValueError`. `tests/test_wr_a_schedule.py:17,59,70` exercises actual parser bytes, the scheduler, and the watchdog. The same malformed body produced a false off-day on both base paths. Empty lists and regular-season filtering have controls.
- R45 leaves the A3 and A4 producers and timing out of this commit. The 5a injected A3 consumer and the existing eligible missed-pick scenario ran as controls. The after-final-read race and W11 retry behavior remain accepted limits.

## Standing hostile-input rows

| Case | A1 | A2 | A5 |
|---|---|---|---|
| Interrupt mid-step | The calendar subtraction is read-only and persists no state. | `KeyboardInterrupt` and `SystemExit` during comparison leave no attempt, send, lock or decision. The 4c restart cases cover the existing post-marker recovery. | Interrupting the read with either exception propagates. The restarted read fetches the original date again. |
| Malformed or non-regular file | Health checks use missing, corrupt, directory and wrong-pair policy fixtures. The existing fallback is shared by the unchanged policy loader. | The final read with malformed JSON or a directory at the dated observation path refuses. Invalid, missing and non-required selection stamps refuse separately. | The changed parser reads HTTP bytes. The local evidence readers retain their existing behavior. Malformed raw JSON and invalid nested shapes are exercised. |
| Midnight ET and year change | Each calculation uses its supplied date. The parity test walks every 2026 and 2027 date and independently pins the deployed 2026 formula. The health test crosses the calendar-year boundary with explicit dates. | A real final reader clock advances past midnight and New Year. The New Year's Eve test stamps a synthetic stored pick because the actual policy has no regular-season pick then. Neither can deliver under the expired observation. | An interrupted December 31 read advances into January 1. The next run refetches December 31 and subsequently asks for January 1 to plan wake-up. |
| Failed or unconfirmed send | Calendar calculation has no send. | The 4c failure/retry and unconfirmed recovery tests exercise the existing ordering. The 5a and 6b-1 controls exercise dispatch, dedup and the missed-pick and morning alarms. Confirmed stale automatic delivery retains its warning. | A watchdog DM failure still reports `schedule_unverifiable` and exits 1. |
| Restart mid-step | Calendar calculation has no durable step. | An interrupted comparison leaves a reusable saved file. Existing 4c restart tests exercise unconfirmed and confirmed markers. A later real check uses new predictions and the corrected stamp. | The interrupted-run test verifies refetch with no partial scheduler slate persisted. |

## A5 caller behavior

| Reader/caller | Existing exception behavior retained |
|---|---|
| `scheduler.run_day` morning fetch | Propagates the new parser error just as it propagates an exhausted transport error. Only retryable transport errors retry inside `retry_urlopen`. A persistent malformed feed can cause service restarts in place of false off-day idle. |
| `scheduler._next_day_wakeup` | Logs a fetch error and hands off in 15 minutes. |
| `data.lineup_collect.collect_for_date` and `bts data collect-lineup-times` | Propagate before polling or writing JSONL. |
| `day_outcome.gather` and `bts day-outcome` | With valid intent and readable evidence, report `schedule_unverifiable`, exit 1 and attempt the configured DM. Preserve error and intent precedence. |
| Scheduler delivery-status guard | Uses the independent unhydrated reader. Unreadable and missing statuses allow delivery. All-void refuses and alerts. A double with one playable leg is sent intact. |
| `model.predict._fetch_game_slots` | Uses an independent hydrated reader. The error object and `{"dates":[]}` yield no slots. `{"dates":null}` raises `TypeError`. The model reader is outside A5. |

## Existing tests edited

| Test | Before | After |
|---|---|---|
| `test_season_end_date.py::test_2027_opening_night_has_a_full_season_ahead` | Boundary September 26 and 186 days. | Boundary September 27 and 187 days. |
| `test_season_end_date.py::test_2027_last_day_and_after` | September 25 counted 1 and September 26 counted 0. | September 26 counts 1 and September 27 counts 0. The later October assertion is retained. |
| `test_saved_pick_paths_cs5a.py::test_scheduler_uses_same_cycle_object_and_policy_decision` | Fresh DM/public with a best-only correction was formatted, sent and committed. | Returns False and alerts without formatting, sending or committing. Asserts the original saved bytes, including the first-read stamp. Matching dated identity still formats, sends and commits. |
| `test_saved_pick_paths_cs5a.py::test_fresh_choices_stamp_the_selection_read[dated-run]` | Exit 0 and a post showing the changed streak of 8, with the stamp from streak 7. | Exit 1 with `no pick`, no post and the same first-read stamp assertion. The automatic sibling still exits 0 and posts streak 8. |
| `test_stale_delivery_cs4c.py::test_attempt_snapshot_and_confirmed_warning` | Fresh cases used an unstamped synthetic selection. | Same send and warning assertions with a valid required automatic stamp. |
| `test_stale_delivery_cs4c.py::test_final_cutoff_is_a_no_compute_control` | An unstamped fresh selection reached the cutoff and refused without computing a stale snapshot. | Same refusal assertions with a valid stamp. |
| `test_stale_delivery_cs4c.py::test_missing_recipient_amendment` | Unstamped fresh DM refused without a recipient and public delivered. | Same outcomes and warning assertions with a valid stamp. |
| `test_stale_delivery_cs4c.py::test_other_stale_warning_copy` | Unstamped fresh delivery emitted the configured stale warning detail. | Same assertion with a valid stamp. |
| `test_stale_delivery_cs4c.py::test_warning_failures_cannot_affect_pick` | Unstamped fresh delivery survived stale-warning failures. | Same assertions with a valid stamp. |
| `test_stale_delivery_cs4c.py::test_warning_dedup_with_two_independent_same_day_deliveries` | The shared helper sent unstamped fresh selections and deduped warnings. | Same assertions with the shared helper supplying valid stamps. |
| `test_stale_delivery_cs4c.py::test_warning_status_is_independent_of_critical_status` | The shared helper sent unstamped fresh selections and kept warning and CRITICAL status independent. | Same assertions with valid stamps. |
| `test_stale_delivery_cs4c.py::test_failed_warning_retries_without_resurface_banner` | The shared helper sent unstamped fresh selections and retried the failed warning. | Same assertions with valid stamps. |
| `test_alert_status_cs6b2_carbon.py::test_lane_callers_recover_corrupt_and_nested_status[delivery_stale_warn-…]` | The two synthetic fresh cases relied on the exemption to reach stale-warning dispatch. | Both use a required automatic stamp from their own fixture reader. All delivery and warning assertions are intact. |
| `test_stale_delivery_cs4c_carbon.py::test_warning_passes_explicit_now_et_date_independent_of_alert_clock` | Two unstamped picks reached delivery and checked the warning date. | Same assertions. Each pick is stamped from its own automatic state fixture. |
| `test_stale_delivery_cs4c_carbon.py::test_send_returning_no_id_and_reentry` | Four parameter cases sent unstamped picks and checked no-ID behavior and reentry. | Same assertions with a valid automatic stamp for the fresh attempt. |
| `test_stale_delivery_cs4c_carbon.py::test_closed_stderr_and_base_exception_after_commit_preserve_delivery` | Two fresh picks in the test reached the warning work with no stamp. | Both are stamped from their own automatic state fixtures. All assertions are intact. |
| `test_stale_delivery_cs4c_review.py::test_base_exceptions_in_the_stale_work_cannot_affect_the_pick` | 32 cases used the fresh helper without an observation. | Each passes the fixture world to the helper, which stamps the pick from its required automatic read. All exception, delivery and warning assertions are intact. |
| `test_stale_delivery_cs4c_review.py::test_no_warning_without_a_confirmation_identifier` | Ten cases supplied an unstamped fresh pick to check the missing-identifier warning behavior. | Same assertions with a required automatic stamp from the form of the fresh helper that takes the fixture world. |
| `test_stale_delivery_cs4c_review.py::test_warning_day_is_passed_to_the_dispatcher_explicitly` | The fresh helper supplied an unstamped pick to verify the dispatcher's date. | Same assertion with the helper taking the fixture world to produce the stamp. |
| `test_saver_cs3.py::test_reserved_saver_changes_the_real_policy_action` | Context August 29, 2027. Asserts MDP skip without the reserved saver and double with it. | Context August 30 to preserve the original policy day count. Both assertions are identical. |
| `test_saver_cs3_review.py::test_mistaken_use_then_undo_never_earned_behavior_at_streaks_5_10_16` | Policy context August 23, 2027. Asserts single with saver spent and double with saver available at streak 16. | Only the policy context moves to August 24. The August 23 observation, clock, dashboard and health checks and all assertions are intact. |

## Runs

- Base Slice 1 late and early correction replays, Slice 2 last-day and September 20 bundled replays, and Slice 3 correction race and schedule replay completed. Outputs are `.codex-review/wr-a/slice{1,2,3}-base.log`.
- RED `.codex-review/wr-a/red.log` had 40 failed and 18 passed. This includes the original interrupted-read fixture's incomplete request expectation. It was corrected to include the subsequent next-day wake-up read.
- First focused run `.codex-review/wr-a/focused.log` had 593 passed, 4 skipped and 7 failed. Five failures were regression fixtures. Amendment 1 resolved the existing 5a test conflict. The other failure was the localhost bind in the web-saver test under sandboxing.
- Amendment focused run `.codex-review/wr-a/focused-amendment.log` had 598 passed, 4 skipped and 2 failed. The recovery fixture expected a different batter, but the actual later check recomputed and sent the earlier batter with its new probability. The fixture now asserts that probability and the corrected stamp and delivery time.
- Final focused run `.codex-review/wr-a/focused-final.log` had 599 passed, 4 skipped and the sole sandbox-denied localhost bind. Both Slice correction tests and both optional-import control files ran in this process.
- The unsandboxed fast gate on commit `c196ca5a` completed in 25:42 with 9,238 passed, 90 failed, 16 errors, 25 skipped, 6 deselected and 1 xfailed. All 53 entries in `~/bts-runs/gate-failed.txt` were present. There were 53 extra failures. The full list and tracebacks are in `.codex-review/wr-a/fast.log`.
- The gate exposed 51 cases requiring synthetic stamps. They were two `delivery_stale_warn` cases in `test_alert_status_cs6b2_carbon.py`, six cases in `test_stale_delivery_cs4c_carbon.py`, and 43 cases in `test_stale_delivery_cs4c_review.py`. The latter also needed to pass the fixture world to the shared stamping helper, which takes it as an argument.
- The other two initial gate failures were saver-policy assertions at the old fixture dates. Work stopped under amendment 2, then amendment 3 authorized moving only those two policy-context dates. Both tests passed with their original assertions in `.codex-review/wr-a/saver-amendment.log` before the 51 fixture stamps were applied.
- The focused amendment verification in `.codex-review/wr-a/fixture-amendment.log` had 77 passed, including the 51 affected stamp cases and both saver-policy assertions.
- The final unsandboxed fast gate on `9cb84536` took 26:27 and reported 9,291 passed, 37 failed, 16 errors, 25 skipped, 6 deselected and 1 xfailed. The 53 failure and error IDs matched `~/bts-runs/gate-failed.txt` exactly, with zero extra or missing IDs. The full log is `.codex-review/wr-a/fast-amended.log`.
- Combined A/B verification belongs on the merged head under the contract's parallel integration rule. This worktree holds only fix set A.
