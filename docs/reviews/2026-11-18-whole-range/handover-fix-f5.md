# F5 author handover

Branch `fix/wr-f5`. Baseline `833c2772`. F5 commit `5a80f0cc`.

## Tests first

The new watchdog suite ran against the unchanged baseline before any production edits. It reported 10 failures and 8 passes.

| Row | Baseline result |
| --- | --- |
| M1 | RED. Both `show` and `install` emitted the 10–23 hour range. |
| M2 | RED. No installed slot read the account for the 06:10 pick. |
| M3 | RED. No installed slot read the account for the 10:05 pick. |
| M4 | RED. The installer replaced the marked line with another 10–23 line. |
| M5 | RED. There was no 09:45 run or `base_slot_available` receipt in the wrong-game replay. |
| M6 | RED. There was no 09:45 run or `base_slot_available` receipt in the removed-entry replay. |
| M7 | RED. Neither early slate had its first eligible account read and confirmation. |
| N1–N5 | GREEN. Research, intent refusals, install and remove behavior, and other generated lines passed. |
| N6 | GREEN. The pinned local gates and earlier-date marker behavior passed. |
| N7 | GREEN. The evening slate retained five slots and stopped account reads after confirmation. |
| N8 | RED for the new clock-day schedule assertions. The baseline had 56 ordinary slots. F1b's existing wrapper tests also ran in the post-change installer batch. |
| Manual | GREEN. The pinned 09:45 manual check still read the account and confirmed. |

The post-change installer batch ran `test_entry_intent.py`, `test_wr_f1b_lock.py`, and `test_wr_f5_watchdog_window.py`. It reported 91 passed and 4 skipped. All 18 F5 cases passed, including the tightened cutoff and warning assertions. The first post-change F5-only run also passed 18 cases.

## Existing regression checks

An initial combined batch reached the 110-second timeout. Its files were split into shorter runs. The test commands used the requested `nice -n 19` wrapper, and this sandbox reported that it could not set process priority. Pytest still ran under the requested timeout.

All paths below are relative to `tests/`.

| Files | Result |
| --- | --- |
| `test_entry_intent.py`, `test_wr_f1b_lock.py` | 73 passed, 4 skipped, excluding the 18 new cases in the same batch. |
| `test_wr_f1_wrong_game.py`, `test_wr_f1_review.py` | 83 passed. |
| `test_entry_receipt.py`, `test_day_outcome_entry_sequence.py`, `test_day_outcome.py`, `health/test_pick_entry_source.py`, `leaderboard/test_scraper_stealth.py` | 157 passed. |
| `test_audit_driver.py` | 33 failed, 16 setup errors. All hit the missing GNU rsync 3.x check, the known baseline failure on this machine. |
| `test_season_contract_cs4b2_edges.py`, `test_wr_c2_carbon_v2.py`, `test_scheduler_decision_record_integration.py`, `test_wr_c2_r2_carbon.py`, `test_wr_a_review_carbon.py`, `test_audit_f6_review_probes.py`, `test_automatic_reading_cs4c.py`, `test_wr_a_schedule.py`, `test_cli_integration.py`, `test_saved_pick_paths_cs5a.py`, `test_wr_b_contract.py`, `test_live_gate_cs6a_review.py`, `test_contest_fetch.py`, `test_wr_b_edges.py`, `test_pick_chosen_on_cs5a.py` | 703 passed, 4 skipped. |

The three matching files under `scripts/` are implementation files. The installer passed `bash -n` and was exercised through the real script with test crontab and uv shims. `git diff --check` passed.

Total for the final selected runs was 1,034 passed and 8 skipped, plus the 49 audit-driver failures and errors described above. The change reached both approval gates and was committed. The handover is untracked.

## Known limits

- The watchdog computes the window in ET wall time. A synthetic entry window across a clock change can be off by an hour. The contract check found no relevant starts in the available 2025 regular-season feeds.
- The morning report's count of checks includes runs that made no account read.
- The clock-slot test models the ET timeline. The F1b test report records platform-specific skips.
