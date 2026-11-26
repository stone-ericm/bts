# F5 round 2 handover

Branch `fix/wr-f5`. Round-2 commit `84e2a0fc` on `210459a8`. The staging and commit approvals were granted. This handover is untracked.

## Resolved stops and review

The first erratum aligned the slot test with the CLI's strict cutoff and inclusive window start. Erratum 2 made the marker and receipt key true-only after `false` changed the F1 parity digests. The final parity grid passed with its original digests. All existing assertions were retained.

Before the final reader correction, both new live `[]` tests failed. The health source returned no alert and the cron command proceeded to verify and replace the marker. Both now raise `AttributeError`, as the base live reads do. A direct probe ran the base and current health checks against the same file and got the same exception. The CLI test compares the manual and cron paths. A non-object dated copy is ignored, and the regular-file guard prevents the health check from blocking on a pipe.

An early run that sent an all-clear can be followed by a second all-clear at an original-hours slot. This can occur after an early wrong-game alert and then an early verified correction. The early `confirmed` marker keeps the `wrong_game` escalation, and the original-hours run evaluates all-clear eligibility from that marker before resetting the tiers. If the early confirmation followed an ordinary missing-entry warning and carries no escalation ledger, the original-hours confirmation sends no second all-clear. This behavior was left as implemented before the final reader correction.

## Tests before production edits

On `451ce153`, the corrected test fixtures gave 11 failures and 27 passes. The 18 round-1 cases still passed. After production edits all 38 cases passed, and the final suite passed all 41 cases after adding the marker read regressions.

| Case | Result before production edits |
| --- | --- |
| R1 / 2(a) | RED. Corrected pitch still hit the early confirmation and skipped the 10:00 account read. |
| R2 | RED. Replacement commit also skipped the 10:00 read. |
| R3 / 2(b) | RED. The early alert spent the tiers, so the revised slot sent no DM. |
| R4 | RED. Prior-date health saw no CRITICAL after the live marker was replaced. |
| R5 | Passed on round 1. All three cases failed with 10:15 as the first search slot in an in-memory change. |
| R6 | Passed on round 1. The 10:20 replay deferred at 09:45 and warned at 10:00. |
| 2(c) | RED. The base-hours run sent no fresh warning for the unresolved early wrong-game observation. |
| 2(d) | The one-time all-clear already passed on round 1. The initial test failed on the since-revised receipt key requirement. The final test checks that the original-hours receipt omits the early key. |
| Morning report | RED for early confirmed and missing observations after correction to an 11:15 pitch. |
| Interrupt and restart | RED. No dated copy was available after retry. After the fix the copy exists before the marker write and keeps its original bytes on retry. F1b's lock tests ran in the bounded batch below. |
| Malformed and non-regular markers | RED for the live FIFO, which hit the 15-second subprocess limit. The malformed live-marker replacement and invalid-copy cases passed before and after. |
| Midnight and year boundary | RED. January 1 overwrote December 31's pending health marker. |
| Failed send | The early `dm_failed` case already retried at the revised slot. It passed before and after. |

## Existing regression runs

All test paths are relative to `tests/`.

| Files | Result |
| --- | --- |
| `test_wr_f1_wrong_game.py` | 70 passed. |
| `test_wr_f1_review.py` | 13 passed. |
| `test_wr_f5_watchdog_window.py` | 41 passed. |
| `test_wr_f1b_lock.py` | 21 passed, 4 skipped. |
| `test_entry_intent.py` | 52 passed. |
| `test_entry_receipt.py`, `test_day_outcome_entry_sequence.py`, `test_day_outcome.py`, `health/test_pick_entry_source.py` | 146 passed. |
| `health/test_attention.py`, `test_submission_cutoff.py`, `leaderboard/test_scraper_stealth.py` | 33 passed. |
| `test_season_contract_cs4b2_edges.py`, `test_wr_c2_carbon_v2.py`, `test_scheduler_decision_record_integration.py`, `test_wr_c2_r2_carbon.py`, `test_wr_a_review_carbon.py`, `test_audit_f6_review_probes.py`, `test_automatic_reading_cs4c.py` | 182 passed. |
| `test_wr_a_schedule.py`, `test_cli_integration.py`, `test_saved_pick_paths_cs5a.py`, `test_wr_b_contract.py`, `test_live_gate_cs6a_review.py`, `test_contest_fetch.py`, `test_wr_b_edges.py`, `test_pick_chosen_on_cs5a.py` | 521 passed, 4 skipped. |
| `test_audit_driver.py` | 33 failed, 16 setup errors. All required GNU rsync 3.x, the known base failure on this machine. |

The final pass covered every matched test file in sequential batches after the last code change. Total was 1,079 passed and 8 skipped, plus the 49 baseline audit-driver failures and errors above. The longest batch took 50.25 seconds. The search also returned three implementation files under `scripts/`. The installer passed `bash -n` and its real entry points ran with shims. Both unstaged and staged diff checks passed. The sandbox rejected the `nice` priority change. The pytest commands still ran under `timeout 110`.
