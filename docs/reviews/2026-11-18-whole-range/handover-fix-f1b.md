# F1b author handover

Branch `fix/wr-f1`. Commit `717ac97f2251fa074adbf6338554a48476c37abb`.

The watchdog line uses the lock and verbose timeout from Amendment 1. The scheduled interval remains 900 seconds, with 840 seconds to TERM and 30 seconds of kill grace. Amendment 5 is appended verbatim to F1's contract in `kickoff-fix-f1.md`. `kickoff-fix-f1b.md` matches `b052a5b2` byte for byte. No existing assertion or `src/` file changed.

## Tests-first evidence

| Row | Result before the cron edit or on the Carbon scratch change |
|---|---|
| K2 | The shim contender exited 0 instead of 1 on the original generated line and started its stand-in command while the holder was active. |
| K3 | Both shim run-limit rows and the retained-descriptor row failed on the missing wrapper. The final shim cases verify TERM delivery and exit 124, KILL and exit 137, signal diagnostics, lock reacquisition, and an escaped child retaining the lock until it exits. |
| Installer replacing marked jobs | The original proposed line failed the wrapper assertion. The final case replaces marked active and commented jobs, preserves foreign lines and an unmarked legacy watchdog, verifies repeated install, and verifies remove. |
| Installer without a crontab, research parity, intent refusal, live-gate refusal, unreadable crontab, filter failure | These six controls passed on the original line. They verify existing behavior. A separate byte comparison of `show` and the installed crontab against the starting HEAD under both intents matched after removing only the new wrapper. |
| Generated order and interval | The original generated line failed the ordered-duration pattern. The final test reads the durations and interval from the generated line and checks `840 + 30 < 15 * 60`. |
| Interruptions | All nine cases were red on the missing wrapper. The final cases stop the real Click command before and during fetch, after DM acceptance or send failure, during the marker temp write, during receipt temp write, between receipt rename and seal, after receipt seal and during all-clear. They check marker integrity, receipt discovery, lock release, the possible repeat after acceptance, and terminal confirmation behavior. |
| Carbon extra 7, A1 exact integer guard changed to `isinstance` | RED for `True` and `False` (2 failed, 6 passed). |
| Carbon extra 8, A1 exact integer guard changed to non-null | RED for `True`, `False`, `"101"`, and `101.0` (4 failed, 4 passed). |
| Carbon extra 9, A3 formatting guard changed to non-null | RED for all four values (4 failed, 4 passed). The rendered reminder used the invalid saved value instead of `recommended game ID unavailable`. |

The original-line run reported 15 failed, 14 passed and 4 skipped. All eight game ID cases already passed on the unmodified source. The three scratch changes were loaded one at a time in an isolated pytest import hook and the temporary hook was removed.

K4 skips by default. With `BTS_F1B_OVERLAP_DIAGNOSTIC=1` it failed at the original safety assertion because the older run erased the observation and the follow-up confirmed. This is the known limit outside the shared cron lock.

## Verification

- New files `test_wr_f1b_lock.py` and `test_wr_f1_game_pk_types.py` passed with 29 passed and 4 skipped. The harness uses a 15-second hang bound and preserves its wrapper durations and grace. The slowest test call on the rerun took 4.24 seconds. The three real-tool skips were K2 contention, K3 TERM behind real uv, and K3 KILL. This Mac has no `/usr/bin/flock`, and its BSD `/usr/bin/timeout` rejects `--version`. The fourth skip was the K4 diagnostic.
- The existing F1 files `test_wr_f1_wrong_game.py` and `test_wr_f1_review.py` passed all 83 cases.
- The existing search matches `test_audit_f6_review_probes.py`, `test_entry_intent.py`, `test_entry_receipt.py`, `test_day_outcome_entry_sequence.py`, `test_saved_pick_paths_cs5a.py`, `test_scheduler_decision_record_integration.py` and `test_contest_fetch.py` reported 314 passed and 4 skipped. The skips were in `test_saved_pick_paths_cs5a.py`.
- The existing search matches `leaderboard/test_scraper_stealth.py`, `test_cli_integration.py`, `test_wr_b_contract.py`, `test_wr_b_edges.py`, `test_wr_c2_carbon_v2.py` and `test_wr_c2_r2_carbon.py` passed all 217 cases. The remaining existing search match was `test_wr_f1_wrong_game.py`, counted in the F1 run.
- `bash -n scripts/cron-setup-hetzner.sh` and `git diff --check` passed. The requested `nice -n 19` emitted a priority warning under the local sandbox. The foreground test commands still ran under the 110-second timeout.

No existing assertion required a change. Linux real-tool verification remains for the Ubuntu gates.
