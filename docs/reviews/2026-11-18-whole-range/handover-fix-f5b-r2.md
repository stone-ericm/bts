# F5b round 2 author handover

Commit `88b0427f5f3db8dd5a1b73f1ca79fcb0852431dc` on `fix/wr-f5b`. Staging and commit had separate approvals. No push was made. Round-two production and test files matched `e66cc74c` before the new tests were added. The two authorized hour assertions from round one remain the only differences in pre-existing tests against `bf78b642`.

| Amendment 2 | Before on `e66cc74c` | After |
|---|---|---|
| B1, pending all-clear | Missing-entry and wrong-game warnings each lost their all-clear after a pitch correction. | Both sequences send one all-clear on confirmation. The next run stays terminal. |
| B1, UTC spelling | The same instant spelled with `Z` reopened the marker and read the account. | The marker stays terminal with zero account calls. |
| B2, season inventory | The genuine sealed live recovery exceeded 64 KiB. Health reported the early CRITICAL after a historical live write displaced the current marker. | The 4 MiB receipt read accepts the sealed recovery and health stays quiet. |
| B2, paused manual write | The manual response came before the early failure, but its write and publication followed it. Health reported the old CRITICAL after displacement. | Health uses the later `published_at` and stays quiet. A live marker write whose receipt is published in the milliseconds after an early observation is counted as following it. |
| B3, marker fields | A non-timestamp, a timestamp without an offset, a timestamp on the prior date, and a list-valued receipt each let the invalid confirmation short-circuit the next check. | Each invalid marker is ignored and the next check observes the account and warns. |
| B3, cutoff overflow | Morning aborted without publishing its report or DM. Early health raised `OverflowError`. | Morning reports ENTRY UNVERIFIABLE and sends its DM. Early health reports the recorded failed send with the current selection unknown. |
| Operator text | The CLI described silent skips and the architecture described unconditional terminal confirmation in the applicable ledger. | The architecture, CLI help and receipt guide carry the review's wording. |
| Tests for surviving regressions | Six new cases passed on the original code. The first-round suite had missed the five temporary changes below. | All six cases pass with the fixes. Each temporary change failed its targeted test, and moving the flag failed both gate cases. |

The temporary changes preferred early health over a same-date live marker without a seal, raised the early-pick limit, set the early receipt flag before the gate, and removed leg binding from each reader separately. All temporary changes were reverted before the final regression batches.

The early marker records pending all-clear status separately from escalation tiers. For older early markers, recorded tiers supply the available warning history. Pitch-only correction still resets the tiers and retains unresolved wrong-game evidence for matching legs. The early pick reader validates the cutoff once for the morning and early health readers.

## Existing test results

The 29 matched existing test files gave 1,255 passes and eight skips. The contest parser added nine passes. Every batch ran alone under the 110-second timeout. The sandbox denied the priority adjustment from `nice`, and pytest continued.

| Files | Result |
|---|---:|
| `tests/test_wr_f1_wrong_game.py` | 70 passed |
| `tests/test_entry_intent.py`, `tests/test_wr_f1b_lock.py` | 73 passed, 4 skipped |
| `tests/test_pick_chosen_on_cs5a.py`, `tests/test_saved_pick_paths_cs5a.py`, `tests/test_scheduler_decision_record_integration.py`, `tests/test_submission_cutoff.py` | 297 passed, 4 skipped |
| `tests/test_entry_receipt.py`, `tests/test_day_outcome.py`, `tests/test_day_outcome_entry_sequence.py`, `tests/health/test_attention.py`, `tests/health/test_pick_entry_source.py` | 163 passed |
| `tests/test_contest_fetch.py`, `tests/test_cli_integration.py`, `tests/test_wr_f1_review.py`, `tests/test_automatic_reading_cs4c.py`, `tests/test_live_gate_cs6a_review.py`, `tests/leaderboard/test_scraper_stealth.py` | 227 passed |
| `tests/test_wr_a_review_carbon.py`, `tests/test_wr_a_schedule.py`, `tests/test_wr_b_contract.py`, `tests/test_wr_b_edges.py`, `tests/test_wr_c2_carbon_v2.py`, `tests/test_wr_c2_r2_carbon.py` | 185 passed |
| `tests/test_audit_f6_review_probes.py`, `tests/test_season_contract_cs4b2_edges.py`, `tests/test_wr_f2_saver_flag.py`, `tests/test_wr_f3_skip_state_review.py`, `tests/test_wr_f4_streak_file.py` | 240 passed |
| `tests/scripts/season_ledger/test_contest.py` | 9 passed |
| `tests/test_wr_f5b_watchdog.py` | 77 passed, including all 60 round-one cases |

The 28 paired manual and original-hours cases matched the archived base after normalizing receipt IDs and fixture paths. This comparison included CLI output, exit codes, DMs, marker and receipt contents, and health and morning results for the fixtures. The installer passed `bash -n` and the installer shim tests. The audit bootstrap completed 20,000 trials over 21 synthetic days, and the backup probe confirmed that its ops set includes the early ledger.

No contract stop condition was reached. The native Linux wrapper and live transports were outside this run.
