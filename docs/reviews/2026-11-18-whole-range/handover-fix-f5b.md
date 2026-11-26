# F5b author handover

Commit `e66cc74cb6c9b32eb5b207e26822ab02b880329f` on `fix/wr-f5b`, with parent `9f342ca051133af39213b464ef771de57130bcfa`. The index and commit approvals were separate. No push was made.

## Watchdog tests

The full base run gave 47 failures and 13 passes. A follow-up base run failed all three cases with tightened assertions for replacement tiers and late confirmation. The final candidate passed all 60 cases in 28.39 seconds. Tests use the real installer with crontab shims, the real CLI with captured account and DM transports, sealed receipts, the real health source and runner, and the real morning command.

| Row | Before on base | After |
|---|---|---|
| R1 | Fail | Pass. Pitch correction leaves the original slot warnings and morning unverifiable state. |
| R2 | Fail | Pass. Replacement commit leaves the same original slot result. |
| R3 | Fail | Pass. Early alerts and a failed early send leave the original initial warning available. |
| R4 | Fail | Pass. The previous date's live CRITICAL survives early confirmation. |
| R5 | Fail for 10:06, 10:15 and 10:20 | Pass for all three. Each defers to 10:00. |
| R6 | Fail | Pass. Wrong-game binding at 10:00 produces the warning and morning unverifiable state. |
| R7 | Fail | Pass. A manual crosswalk gap leaves the original one-minute warning available. |
| R8 | Fail | Pass. The manual check re-reads the account and warns after early confirmation. |
| R9 | Fail | Pass. One early wrong-game warning and one early all-clear. Zero DMs at the live recheck. |
| R10 | Early-only absence fails. Bound manual absence passes. | Pass for both. The FIFO results are ENTRY UNVERIFIABLE and NOT ENTERED respectively, each with exit 1 and a DM. |
| R11 | Pass | Pass. The real health runner dispatches a CRITICAL for the live UnicodeDecodeError. |
| R12 | Fail | Pass. A December 31 manual confirmation survives January 1 early checks. |
| R13 | Pass | Pass. The manual T−30 reminder uses the live tiers. A same-date live confirmation takes precedence in health. |
| N1 | Pass | Pass. Research has no watchdog and the fetch lines have no DM recipient. |
| N2 | Pass | Pass. Four bad intents refuse before changing crontab. |
| N3 | Foreign, removal, and refusal checks pass. The new schedule assertion fails. | Pass. Installer retains foreign lines and handles old or commented lines, missing crontab, and unreadable crontab. |
| N4 | Pass | Pass. All marked lines are removed. |
| N5 | Pass | Pass. Other generated lines match across enter and research with the existing fetch recipient rule. |
| N6 | Pass | Pass. Local gates and terminal confirmation make no account request. |
| N7 | Pass | Pass. A 19:10 first pitch has five eligible evening slots. |
| N8 | Nominal clock count assertion fails. | Pass. Counts are 92, 96 and 100. Wrapper contention and interruption pass with shims. |
| M1 | Fail in show and install | Pass in both. Generated line carries the all-day cadence and cron flag. |
| M2 | Fail | Pass. 06:10 missing-entry replay checks four times after delivery. |
| M3 | Fail | Pass. 10:05 missing-entry replay checks four times before cutoff. |
| M4 | Fail | Pass. The old line is replaced once. |
| M5 | Fail | Pass. Early check defers and the original slot reports wrong-game evidence. |
| M6 | Fail | Pass. The original schedule still reports a removed entry. |
| M7 | Fail for both windows | Pass for both. One account observation confirms and subsequent eligible runs stay local. |

| R70 case | Before on base | After |
|---|---|---|
| Same-day replacement after confirmation or spent tiers | Fail in both after tightening the tiers assertion | Pass in both |
| Later manual absence | Fail | Pass. The morning DM says NOT ENTERED. |
| Manual fix after the last early slot | Fail | Pass. The early warning can remain without an all-clear. |
| Independent warnings and all-clears | Fail | Pass. Each ledger can send one warning and one all-clear. |
| Corrected cutoff and genuine late confirmation | Fail in both | Pass in both. The real morning command reports ENTERED LATE. |
| Early failure across adjacent dates | Fail | Pass. The previous early failure remains available during the next date's early checks. |
| Stale fallback after live recovery | Fail | Pass. Both a newer live date and a later sealed live recovery prevent the old early failure from resurfacing. |
| Invalid early marker destination | Fail | Pass. Stderr and the sealed receipt report the failed write while exit 1 is preserved. |
| Early health with a non-regular pick | Fail | Pass. Health reports the failed send with its observation time and unknown-selection text. |
| Delayed holder at the sole original slot | Fail | Pass. Real generated wrapper with shims skips the locked 10:00 start. |

| Standing case | Before on base | After |
|---|---|---|
| Interrupt before or after replace | Both fail | Both pass. The early marker keeps the old or new bytes respectively. |
| Malformed or non-regular inputs | Early marker/decision cases fail. Live invalid non-UTF-8 behavior passes. | Pass. FIFO, non-object, bad encoding, oversize, and invalid tiers are covered. The decision FIFO uses the delivered-pick fallback. |
| Midnight ET and year boundary | New date without a pick passes. Independent marker and rollover cases fail. | Pass. Missing new-date pick stays local. Dated early files and the live recovery stay independent. |
| Failed or unconfirmed send | Retry and cutoff alert case passes through the base ledger. | Pass through the early ledger. A successful retry changes the end-of-day alert from CRITICAL to WARN. |
| Restart and lock | No early slot to retain the lock. | Pass. The next run uses the previously published marker and a retained wrapped lock blocks the next wrapped invocation. |
| Pitch-only correction | Reopening and distinct early wrong-game history fail. | Pass. Rechecks a corrected pitch and keeps unresolved wrong-game evidence for matching legs. |

## Existing checks after the final code change

All 29 matching existing test files passed their runnable cases. Only the authorized hour assertions at `tests/test_entry_intent.py:169` and `tests/test_wr_f1b_lock.py:59` changed. The two FIFO tests in the wrong-game file ran in its own batch with the other cases in that file. The bounded batches took 40.58, 59.94, 1.74, 6.45 and 28.62 seconds.

| File | Passed | Skipped |
|---|---:|---:|
| `tests/test_wr_f1_wrong_game.py` | 70 | 0 |
| `tests/test_wr_f1b_lock.py` | 21 | 4 |
| `tests/test_entry_intent.py` | 52 | 0 |
| `tests/test_entry_receipt.py` | 49 | 0 |
| `tests/health/test_attention.py` | 17 | 0 |
| `tests/health/test_pick_entry_source.py` | 6 | 0 |
| `tests/leaderboard/test_scraper_stealth.py` | 11 | 0 |
| `tests/test_audit_f6_review_probes.py` | 3 | 0 |
| `tests/test_automatic_reading_cs4c.py` | 71 | 0 |
| `tests/test_cli_integration.py` | 70 | 0 |
| `tests/test_contest_fetch.py` | 54 | 0 |
| `tests/test_day_outcome.py` | 56 | 0 |
| `tests/test_day_outcome_entry_sequence.py` | 35 | 0 |
| `tests/test_live_gate_cs6a_review.py` | 8 | 0 |
| `tests/test_pick_chosen_on_cs5a.py` | 171 | 0 |
| `tests/test_saved_pick_paths_cs5a.py` | 97 | 4 |
| `tests/test_scheduler_decision_record_integration.py` | 24 | 0 |
| `tests/test_season_contract_cs4b2_edges.py` | 20 | 0 |
| `tests/test_submission_cutoff.py` | 5 | 0 |
| `tests/test_wr_a_review_carbon.py` | 25 | 0 |
| `tests/test_wr_a_schedule.py` | 24 | 0 |
| `tests/test_wr_b_contract.py` | 24 | 0 |
| `tests/test_wr_b_edges.py` | 73 | 0 |
| `tests/test_wr_c2_carbon_v2.py` | 21 | 0 |
| `tests/test_wr_c2_r2_carbon.py` | 18 | 0 |
| `tests/test_wr_f1_review.py` | 13 | 0 |
| `tests/test_wr_f2_saver_flag.py` | 63 | 0 |
| `tests/test_wr_f3_skip_state_review.py` | 92 | 0 |
| `tests/test_wr_f4_streak_file.py` | 62 | 0 |
| `tests/scripts/season_ledger/test_contest.py` | 9 | 0 |

The 29 matched files account for 1,255 passes and eight skips. The contest parser row is the additional check for `scripts/audit/season_ledger/sources/contest_ledger.py`. The installer passed `bash -n` and its shim tests. The synthetic 21-day run of `scripts/audit/pvr_threshold_bootstrap.py` completed all 20,000 trials.

`tests/test_audit_driver.py` reproduced its known base failure on this Mac. GNU rsync 3.x was unavailable and the file reported 33 failures and 16 setup errors. The sandbox denied the priority adjustment from `nice`. Pytest continued and produced the results above. Four native-wrapper checks skipped on macOS. The shims were used for the wrapper tests and the installer was run locally.

## Implementation notes

Health uses a successful sealed live receipt and a later response-completion timestamp as evidence of a later completed marker write. The live receipt's publication timestamp alone can follow an early observation even when its marker write came first. The live schema and marker write path retain their existing format. Morning evidence uses the final pick's cutoff for matching early observations and retains all sealed receipts in the check count.

No stop condition was reached. The full fast suite and live transports were outside this run.
